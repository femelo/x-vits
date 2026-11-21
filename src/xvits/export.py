"""``xvits-export``: an x-vits bundle -> ONNX graphs + ``runtime.json``.

Graphs (all batch 1, dynamic length; noise is an input so runs are reproducible):

  encoder.onnx     phoneme ids -> text hidden states x, mask
  prominence.onnx  x -> per-token prominence logits                (stage 4b)
  predictor.onnx   one velocity evaluation of the flow-matching predictor;
                   the runtime runs the ODE (and the guidance) around it
  durations.onnx   stochastic duration predictor, reverse pass (+ style/emphasis
                   conditioning given as an additive tensor)
  acoustic.onnx    x, prosody, frame durations, noise -> waveform
                   (prosody prior, length regulation, frame-level prosody,
                   flow, HiFi-GAN)

``--verify`` checks every graph against PyTorch on a real sentence and the full
chain end to end with identical noise.
"""

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

_LOG = logging.getLogger("xvits.export")
OPSET = 17


# ---------------------------------------------------------------------------
def interp_prosody_frames_onnx(prosody, w, n_frames):
    """Vectorised, export-friendly equivalent of
    ``piper_train.vits.models.interp_prosody_frames`` (same values)."""
    b, c, t = prosody.shape
    d = w[:, 0].to(prosody.dtype)                         # [b, t]
    centres = torch.cumsum(d, -1) - d / 2                 # [b, t]
    frames = torch.arange(n_frames, device=w.device, dtype=prosody.dtype) + 0.5
    tok = torch.arange(t, device=w.device)
    outs = []
    for ch in range(c):
        knot = d > 0
        if ch == 0 and c >= 2:
            knot = knot & (prosody[:, 1] > 0.5)
        le = (centres[:, None, :] <= frames[None, :, None]) & knot[:, None, :]       # [b, F, t]
        gt = (centres[:, None, :] > frames[None, :, None]) & knot[:, None, :]
        prev = torch.where(le, tok, torch.full_like(tok, -1)).amax(-1)                # [b, F]
        nxt = torch.where(gt, tok, torch.full_like(tok, t)).amin(-1)
        has_prev, has_next = prev >= 0, nxt < t
        p_i = torch.where(has_prev, prev, nxt).clamp(0, t - 1)
        n_i = torch.where(has_next, nxt, prev).clamp(0, t - 1)
        xp, xn = torch.gather(centres, 1, p_i), torch.gather(centres, 1, n_i)
        vals = prosody[:, ch]
        yp, yn = torch.gather(vals, 1, p_i), torch.gather(vals, 1, n_i)
        a = ((frames[None] - xp) / (xn - xp).clamp_min(1e-6)).clamp(0.0, 1.0)
        a = torch.where(p_i == n_i, torch.zeros_like(a), a)
        y = yp + a * (yn - yp)
        y = torch.where(has_prev | has_next, y, torch.zeros_like(y))
        outs.append(y)
    return torch.stack(outs, 1)


class Encoder(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.enc_p = g.enc_p

    def forward(self, ids, lengths):
        x, _, _, x_mask = self.enc_p(ids, lengths)
        return x, x_mask


class Acoustic(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x, x_mask, prosody, w_ceil, noise, spk=None):
        """spk: the speaker (emotion) embedding [1, gin, 1] of a multi-speaker voice, else None."""
        from piper_train.vits import commons

        g = self.g
        m_p, logs_p = g._prosody_prior(x, x, x, x_mask, prosody)
        n_frames = noise.size(2)
        y_lengths = torch.sum(w_ceil, [1, 2]).long()
        y_mask = commons.sequence_mask(y_lengths, n_frames).unsqueeze(1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        if g.prosody_frame is not None:
            pf = interp_prosody_frames_onnx(prosody, w_ceil, n_frames)
            m_p = m_p + g.prosody_frame(pf * y_mask) * y_mask
        z_p = m_p + noise * torch.exp(logs_p)
        z = g.flow(z_p, y_mask, g=spk, reverse=True)
        return g.dec(z * y_mask, g=spk)


class Durations(nn.Module):
    """SDP reverse pass with the noise ``z`` [b, 2, t] given (already x noise_scale_w)."""

    def __init__(self, g):
        super().__init__()
        self.dp = g.dp

    def forward(self, x, x_mask, z, h_add):
        dp = self.dp
        h = dp.pre(torch.detach(x)) + h_add
        h = dp.convs(h, x_mask)
        h = dp.proj(h) * x_mask
        flows = list(reversed(dp.flows))
        flows = flows[:-2] + [flows[-1]]
        for flow in flows:
            z = flow(z, x_mask, g=h, reverse=True)
        return z[:, :1]


class OnnxEncoderLayer(nn.Module):
    """Export-friendly re-implementation of ``nn.TransformerEncoderLayer``
    (norm_first, batch_first, relu/gelu, eval mode) with the same weights.

    The stock layer traces the sequence length into Reshape nodes as a
    constant, so the exported graph would only accept the traced length."""

    def __init__(self, layer: nn.TransformerEncoderLayer):
        super().__init__()
        assert layer.norm_first and layer.self_attn.batch_first
        self.l = layer
        self.h = layer.self_attn.num_heads

    def forward(self, x, src_key_padding_mask=None):
        l, h = self.l, self.h
        y = l.norm1(x)
        b, e = y.size(0), y.size(2)
        d = e // h
        qkv = torch.nn.functional.linear(y, l.self_attn.in_proj_weight, l.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, -1, h, d).transpose(1, 2)
        k = k.reshape(b, -1, h, d).transpose(1, 2)
        v = v.reshape(b, -1, h, d).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        if src_key_padding_mask is not None:
            att = att.masked_fill(src_key_padding_mask[:, None, None, :], -1e9)
        o = torch.matmul(torch.softmax(att, -1), v).transpose(1, 2).reshape(b, -1, e)
        x = x + l.self_attn.out_proj(o)
        y = l.norm2(x)
        return x + l.linear2(l.activation(l.linear1(y)))


def onnx_safe(m: nn.Module) -> nn.Module:
    """Swap every TransformerEncoderLayer for OnnxEncoderLayer, in place (use on
    a freshly loaded, export-only copy: weight-normed modules can't be deep-copied)."""
    for name, child in list(m.named_modules()):
        for cname, c in list(child.named_children()):
            if isinstance(c, nn.TransformerEncoderLayer):
                setattr(child, cname, OnnxEncoderLayer(c))
    return m


class PredictorStep(nn.Module):
    def __init__(self, est, has_style):
        super().__init__()
        self.est, self.has_style = est, has_style

    def forward(self, x_t, mask, cond, t, g):
        return self.est(x_t, mask, cond, t, g=g if self.has_style else None) * mask


class Prominence(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, x_mask):
        return self.head(x, x_mask)


# ---------------------------------------------------------------------------
def _export(module, args, path, inputs, outputs, axes):
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=axes, opset_version=OPSET, do_constant_folding=True, dynamo=False)
    _LOG.info("wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def export(bundle_path: Path, out: Path, verify: bool = False, text: str = None):
    from piper_train.eval.phonemize import PiperPhonemizer
    from piper_train.eval.synthesize import load_model
    from piper_train.prosody.predictor import ProsodyPredictor

    from .bundle import Bundle

    torch.backends.mha.set_fastpath_enabled(False)
    b = Bundle.load(bundle_path)
    out.mkdir(parents=True, exist_ok=True)
    g = load_model(Path(b.voice), "cpu", False).model_g.eval()
    with torch.no_grad():
        g.dec.remove_weight_norm()
    pred = ProsodyPredictor.load(b.predictor, "cpu").eval()
    pz = PiperPhonemizer(Path(b.config))
    text = text or "Hello there, this is a quick test of the exported voice."
    ids = torch.LongTensor(pz.ids(text))[None]
    L = torch.LongTensor([ids.size(1)])
    T = "phonemes"

    with torch.no_grad():
        x, x_mask = Encoder(g)(ids, L)
    C = x.size(1)
    _export(Encoder(g), (ids, L), out / "encoder.onnx", ["ids", "lengths"], ["x", "x_mask"],
            {"ids": {1: T}, "x": {2: T}, "x_mask": {2: T}})

    # durations (SDP reverse)
    ch = g.dp.pre.out_channels
    z = torch.randn(1, 2, ids.size(1))
    h_add = torch.zeros(1, ch, ids.size(1))
    _export(Durations(g), (x, x_mask, z, h_add), out / "durations.onnx", ["x", "x_mask", "z", "h_add"], ["logw"],
            {"x": {2: T}, "x_mask": {2: T}, "z": {2: T}, "h_add": {2: T}, "logw": {2: T}})

    # acoustic
    w_ceil = torch.full((1, 1, ids.size(1)), 3.0) * x_mask
    n_frames = int(w_ceil.sum())
    n_pros = g.prosody_channels  # 3, or 4 with the vocal-effort channel
    pros = torch.zeros(1, n_pros, ids.size(1)); pros[:, 1] = 1.0
    noise = torch.randn(1, g.inter_channels, n_frames) * 0.667
    multi = bool(b.speakers) and getattr(g, "emb_g", None) is not None
    ac_in, ac_names = (x, x_mask, pros, w_ceil, noise), ["x", "x_mask", "prosody", "w_ceil", "noise"]
    if multi:  # the emotion's speaker embedding enters the flow and the vocoder
        ac_in, ac_names = ac_in + (g.emb_g.weight[:1].detach().unsqueeze(-1),), ac_names + ["spk"]
    _export(Acoustic(g), ac_in, out / "acoustic.onnx", ac_names, ["audio"],
            {"x": {2: T}, "x_mask": {2: T}, "prosody": {2: T}, "w_ceil": {2: T}, "noise": {2: "frames"}, "audio": {2: "samples"}})

    # predictor step
    est = onnx_safe(pred.cfm.estimator).eval()
    has_style = est.style is not None
    cond = pred._cond(x, None)
    x_t = torch.randn(1, pred.n_targets, ids.size(1))
    t = torch.tensor([0.3])
    gid = torch.LongTensor([0])
    _export(PredictorStep(est, has_style), (x_t, x_mask, cond, t, gid), out / "predictor.onnx",
            ["x_t", "mask", "cond", "t", "style"], ["v"], {"x_t": {2: T}, "mask": {2: T}, "cond": {2: T}, "v": {2: T}})

    files = {"encoder": "encoder.onnx", "durations": "durations.onnx", "acoustic": "acoustic.onnx", "predictor": "predictor.onnx"}
    # prominence student
    if b.prominence:
        from piper_train.prosody.prominence import ProminenceHead

        ck = torch.load(b.prominence, map_location="cpu", weights_only=False)
        head = ProminenceHead(**ck["kw"]).eval(); head.load_state_dict(ck["state_dict"])
        _export(Prominence(onnx_safe(head).eval()), (x, x_mask), out / "prominence.onnx", ["x", "x_mask"], ["logits"],
                {"x": {2: T}, "x_mask": {2: T}, "logits": {1: T}})
        files["prominence"] = "prominence.onnx"

    # style/emphasis conditioning of the duration predictor: tiny, kept as arrays
    if b.durations:
        from piper_train.prosody.train_dp_style import load_style_dp

        g2 = load_model(Path(b.voice), "cpu", False).model_g.eval()
        cond_m, styles_dp = load_style_dp(b.durations, g2, "cpu")
        # the style SDP replaces the voice's SDP weights: re-export durations.onnx with them
        _export(Durations(g2), (x, x_mask, z, h_add), out / "durations.onnx", ["x", "x_mask", "z", "h_add"], ["logw"],
                {"x": {2: T}, "x_mask": {2: T}, "z": {2: T}, "h_add": {2: T}, "logw": {2: T}})
        arrs = {"style": cond_m.style.weight.detach().numpy()}
        if cond_m.emph is not None:
            arrs["emph_w"] = cond_m.emph.weight.detach().numpy(); arrs["emph_b"] = cond_m.emph.bias.detach().numpy()
        np.savez(out / "dp_cond.npz", **arrs)
        files["dp_cond"] = "dp_cond.npz"
    if multi:
        # speaker embeddings for the acoustic graph, and their (constant) contribution to the
        # duration predictor, which the runtime adds to h_add
        gdp = (g2 if b.durations else g).dp
        with torch.no_grad():
            emb = g.emb_g.weight.detach()
            dp_g = gdp.cond(emb.unsqueeze(-1))[:, :, 0]
        np.savez(out / "speakers.npz", g=emb.numpy(), dp_g=dp_g.numpy())
        files["speakers"] = "speakers.npz"

    shutil.copy(b.config, out / "config.json")
    if b.template:
        shutil.copy(b.template, out / "emphasis_template.json")
        files["template"] = "emphasis_template.json"
    manifest = {
        "name": b.name, "files": files, "config": "config.json",
        "sample_rate": json.load(open(b.config))["audio"]["sample_rate"],
        "hop_length": 256, "inter_channels": g.inter_channels, "text_channels": C,
        "predictor": {"stats": pred.stats, "styles": pred.styles, "token_feats": pred.token_feats,
                      "n_targets": pred.n_targets},
        "durations_styles": styles_dp if b.durations else [],
        "frame_prosody": g.prosody_frame is not None, "defaults": b.defaults,
        "speakers": b.speakers if multi else None,
    }
    (out / "runtime.json").write_text(json.dumps(manifest, indent=1))
    _LOG.info("runtime manifest: %s", out / "runtime.json")
    if verify:
        verify_export(out, b, text)
    return out


# ---------------------------------------------------------------------------
def verify_export(out: Path, b, text: str, tol: float = 1e-3):
    """Each graph vs PyTorch, then the whole chain with identical noise."""
    import onnxruntime as ort

    from piper_train.eval.phonemize import PiperPhonemizer
    from piper_train.eval.synthesize import load_model
    from piper_train.vits import models as vm

    from .runtime import XVits

    torch.backends.mha.set_fastpath_enabled(False)
    g = load_model(Path(b.voice), "cpu", False).model_g.eval()
    with torch.no_grad():
        g.dec.remove_weight_norm()
    if b.durations:
        from piper_train.prosody.train_dp_style import load_style_dp

        load_style_dp(b.durations, g, "cpu")
    pz = PiperPhonemizer(Path(b.config))
    ids = torch.LongTensor(pz.ids(text))[None]
    L = torch.LongTensor([ids.size(1)])
    sess = lambda n: ort.InferenceSession(str(out / n), providers=["CPUExecutionProvider"])

    # frame interpolation: vectorised vs reference
    pros = torch.randn(1, g.prosody_channels, ids.size(1)); pros[:, 1] = (torch.rand(1, ids.size(1)) > 0.3).float()
    w = torch.randint(0, 6, (1, 1, ids.size(1))).float()
    nf = int(w.sum())
    ref = vm.interp_prosody_frames(pros, w, nf)
    vec = interp_prosody_frames_onnx(pros, w, nf)
    _LOG.info("verify interp: max |diff| %.2e", (ref - vec).abs().max())
    assert torch.allclose(ref, vec, atol=1e-5)

    with torch.no_grad():
        x, x_mask = Encoder(g)(ids, L)
    ox, om = sess("encoder.onnx").run(None, {"ids": ids.numpy(), "lengths": L.numpy()})
    _LOG.info("verify encoder: max |diff| %.2e", np.abs(ox - x.numpy()).max())
    z = torch.randn(1, 2, ids.size(1)) * 0.4
    h = torch.zeros(1, g.dp.pre.out_channels, ids.size(1))
    with torch.no_grad():
        lw = g.dp(x, x_mask, reverse=True, noise_scale=0.0, h_add=h)  # noise 0 path
        lw2 = Durations(g)(x, x_mask, z, h)
    olw = sess("durations.onnx").run(None, {"x": x.numpy(), "x_mask": x_mask.numpy(), "z": z.numpy(), "h_add": h.numpy()})[0]
    _LOG.info("verify durations: max |diff| %.2e", np.abs(olw - lw2.numpy()).max())
    w_ceil = torch.ceil(torch.exp(lw2) * x_mask)
    noise = torch.randn(1, g.inter_channels, int(w_ceil.sum())) * 0.667
    spk = g.emb_g.weight[-1:].detach().unsqueeze(-1) if (b.speakers and getattr(g, "emb_g", None) is not None) else None
    with torch.no_grad():
        audio = Acoustic(g)(x, x_mask, pros.clamp(-3, 3), w_ceil, noise, spk)
    feeds = {"x": x.numpy(), "x_mask": x_mask.numpy(), "prosody": pros.clamp(-3, 3).numpy(),
             "w_ceil": w_ceil.numpy(), "noise": noise.numpy()}
    if spk is not None:
        feeds["spk"] = spk.numpy()
    oa = sess("acoustic.onnx").run(None, feeds)[0]
    _LOG.info("verify acoustic: max |diff| %.2e (audio peak %.2f)", np.abs(oa - audio.numpy()).max(), float(audio.abs().max()))

    # full chain: ONNX runtime vs the PyTorch path, same noise, several lengths
    # (dynamic shapes must not depend on the traced sentence)
    voice = XVits(out)
    style = b.styles[0] if b.styles else None
    for k, sent in enumerate([text, "Yes.", "Well, that was a *much* longer sentence than the one we used for tracing, wasn't it?"]):
        a_onnx, trace = voice.synthesize(sent, style=style, seed=1234 + k, return_trace=True)
        a_torch = torch_reference(b, g, trace)
        n = min(len(a_onnx), len(a_torch))
        diff = float(np.abs(a_onnx[:n] - a_torch[:n]).max())
        _LOG.info("verify end-to-end (%d phonemes): %d vs %d samples, max |diff| %.2e",
                  len(trace["ids"]), len(a_onnx), len(a_torch), diff)
        assert len(a_onnx) == len(a_torch), "length mismatch"
        assert diff < 5e-2, f"end-to-end mismatch {diff}"
    if voice.prom is not None:  # prominence student at another length
        from piper_train.prosody.prominence import ProminenceHead

        ck = torch.load(b.prominence, map_location="cpu", weights_only=False)
        head = ProminenceHead(**ck["kw"]).eval(); head.load_state_dict(ck["state_dict"])
        ids2 = torch.LongTensor(pz.ids("A different length for the student."))[None]
        with torch.no_grad():
            x2, m2 = Encoder(g)(ids2, torch.LongTensor([ids2.size(1)]))
            ref = head(x2, m2).numpy()
        got = voice.prom.run(None, {"x": x2.numpy(), "x_mask": m2.numpy()})[0]
        _LOG.info("verify prominence: max |diff| %.2e", np.abs(ref - got).max())
        assert np.abs(ref - got).max() < 1e-3
    _LOG.info("verification passed")


def torch_reference(b, g, trace):
    """Recompute the whole chain in PyTorch from the runtime's noise: predictor
    ODE (with style guidance), decoding, emphasis shift, durations, acoustic."""
    import math

    from piper_train.prosody.predictor import ProsodyPredictor, _Guided
    from piper_train.vits.cfm import OTCFM

    pred = ProsodyPredictor.load(b.predictor, "cpu").eval()
    t = lambda a: torch.from_numpy(np.asarray(a)).float()
    ids = torch.LongTensor(trace["ids"])[None]
    L = torch.LongTensor([ids.size(1)])
    with torch.no_grad():
        x, x_mask = Encoder(g)(ids, L)
        cond = pred._cond(x, None)
        style = torch.LongTensor([trace["style_index"]]) if trace["style_index"] is not None else None
        w = torch.as_tensor(trace["guidance"], dtype=torch.float32)
        cfm = pred.cfm
        if style is not None and bool((w != 1.0).any()):
            cfm = OTCFM(_Guided(pred.cfm.estimator, w.view(1, -1, 1) if w.dim() else w))
        y = cfm.sample(cond, x_mask, n_steps=trace["steps"], temperature=t(trace["temperature"]).view(1, -1, 1),
                       noise=t(trace["noise0"]), solver="midpoint", g=style)
        _, pros = pred.decode(y, x_mask)
        if trace.get("voicing_fix"):
            from piper_train.prosody.features import fix_voicing_np

            pros[0] = torch.from_numpy(fix_voicing_np(pros[0].numpy(), trace["symbols"]))
            pros = pred.bridge_effort(pros, x_mask)
        shift = t(trace["p_st"]) * (math.log(2) / 12) / pred.stats["lf0_std"]
        pros[:, 0] = torch.where(pros[:, 1] > 0.5, pros[:, 0] + shift[None], pros[:, 0])
        pros[:, 2] += t(trace["e_log"])[None] / pred.stats["energy_std"]
        if pros.size(1) > 3:
            pros[:, 3] += t(trace["t_log"])[None] / pred.stats["tilt_std"]
        if trace["unvoiced"]:
            pros[:, :2] = 0.0
        if trace.get("pitch_ceiling") is not None:
            from piper_train.prosody.features import soft_pitch_ceiling

            pros[0, 0] = torch.from_numpy(soft_pitch_ceiling(pros[0, 0].numpy(), pros[0, 1].numpy(), trace["pitch_ceiling"]))
        diff_p = float((pros - t(trace["prosody"])).abs().max())
        _LOG.info("verify predictor ODE + decode + emphasis: max |diff| %.2e", diff_p)
        logw = Durations(g)(x, x_mask, t(trace["z"]), t(trace["h_add"]))
        wd = torch.exp(logw) * x_mask * t(trace["d_mul"]).view(1, 1, -1)
        w_ceil = torch.ceil(wd * trace["length_scale"]) * x_mask
        if trace.get("stress_cap"):
            from piper_train.prosody.features import cap_stress_frames

            cap = float(np.ceil(trace["stress_cap"] * trace["length_scale"]))
            w_ceil[0, 0] = torch.from_numpy(cap_stress_frames(w_ceil[0, 0].numpy(), trace["symbols"], cap))
        spk = t(trace["spk"]) if trace.get("spk") is not None else None
        audio = Acoustic(g)(x, x_mask, pros, w_ceil, t(trace["noise"]), spk)
    return audio.squeeze().numpy() * np.float32(10 ** (trace.get("gain_db", 0.0) / 20))


def main(argv=None):
    p = argparse.ArgumentParser(prog="xvits-export", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path, help="xvits.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--text", help="sentence used for tracing/verification")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    export(a.bundle, a.out, a.verify, a.text)


if __name__ == "__main__":
    main()
