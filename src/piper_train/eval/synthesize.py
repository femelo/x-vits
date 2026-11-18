"""Synthesize the eval set from a Piper training checkpoint (VITS or VITS+CFM).

Defaults to CPU; pass ``--device cuda`` explicitly to use a GPU.

Examples::

    python -m piper_train.eval.synthesize --checkpoint amy.ckpt --out-dir out/baseline
    python -m piper_train.eval.synthesize --checkpoint cfm.ckpt --use-cfm \
        --cfm-n-steps 4 --out-dir out/cfm4
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..vits.lightning_model import VitsModel


def load_model(checkpoint: Path, device: str, use_cfm: bool, **cfm_kwargs) -> VitsModel:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams["dataset"] = None
    if use_cfm:
        hparams["use_cfm"] = True
    hparams.update({k: v for k, v in cfm_kwargs.items() if v is not None})
    model = VitsModel(**hparams)
    # Non-strict so a plain VITS checkpoint loads into a CFM model (new modules
    # keep their init) -- only useful for smoke tests; real CFM weights need training.
    result = model.load_state_dict(ckpt["state_dict"], strict=False)
    if result.unexpected_keys:
        print("unexpected keys:", result.unexpected_keys[:5])
    return model.eval().to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-set", type=Path, default=Path("data/ljspeech-medium/eval_set.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-cfm", action="store_true")
    p.add_argument("--cfm-n-steps", type=int)
    p.add_argument("--noise-scale", type=float, default=0.667, help="VITS noise / CFM temperature")
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--noise-scale-w", type=float, default=0.8)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--prosody",
        type=Path,
        help="prosody.pt with real per-token features: feed the real recording's "
        "pitch/energy (rows need 'utt'; model needs prosody conditioning)",
    )
    p.add_argument(
        "--gt-durations",
        action="store_true",
        help="with --prosody: also use the real recording's token durations",
    )
    p.add_argument(
        "--no-prosody-input",
        action="store_true",
        help="with --prosody: do not feed pitch/energy (e.g. only --gt-durations)",
    )
    p.add_argument(
        "--prosody-predictor",
        type=Path,
        help="stage 3 predictor (best.pt): sample durations + pitch/energy from it",
    )
    p.add_argument(
        "--pp-temperature",
        default="1.0",
        help="predictor noise temperature: one value, or 'DUR,PROSODY' to set "
        "duration and pitch/voicing/energy separately",
    )
    p.add_argument("--pp-steps", type=int, default=16)
    p.add_argument("--pp-solver", default="midpoint")
    p.add_argument(
        "--pp-style",
        help="style label for a style-conditioned predictor; 'label' (alias 'auto') "
        "= each eval row's recorded 'category', for scoring against the recording "
        "(unknown labels are an error). Omit for the unspecified style.",
    )
    p.add_argument("--pp-guidance", default="1.0",
                   help="classifier-free guidance on the style (>1 exaggerates): one "
                   "value, or 'DUR,PROSODY' to exaggerate timing and pitch/energy separately")
    p.add_argument("--pp-durations", choices=["predictor", "sdp"], default="predictor",
                   help="sdp: keep the voice's own duration predictor, use the stage 3 "
                   "predictor only for pitch/voicing/energy")
    p.add_argument("--pp-voiceless-fix", action="store_true",
                   help="keep voiceless consonants (f, s, t, ...) unvoiced in the predicted prosody")
    p.add_argument("--pp-voicing-fix", action="store_true",
                   help="voicing by phoneme type: voiceless consonants unvoiced, vowels and "
                   "sonorants voiced (pitch interpolated); use with --pp-durations sdp")
    p.add_argument("--dp-style-ckpt", type=Path,
                   help="style-conditioned duration predictor (train_dp_style best.pt); "
                   "the style comes from --pp-style (none = unspecified)")
    p.add_argument("--emphasis", type=Path,
                   help="emphasis.pt (per-token word emphasis by 'utt'); rows with a "
                   "'marked' field (*word* markup) are converted on the fly instead")
    p.add_argument("--emphasis-dur-scale", type=float, default=1.0,
                   help="scale the emphasis signal given to the style duration predictor "
                   "(>1 lengthens emphasised words more; ~3 brings ex02 close to the recordings)")
    p.add_argument("--emphasis-guidance", type=float, default=1.0,
                   help="classifier-free guidance on the emphasis input of the pitch/energy "
                   "predictor (>1 exaggerates the learned emphasis contour)")
    p.add_argument("--emphasis-template", type=Path,
                   help="emphasis_template.json: apply this pitch/duration shape to marked words "
                   "(on top of the emphasis-free prediction; needs --pp-durations sdp)")
    p.add_argument("--emphasis-template-scale", type=float, default=1.0)
    p.add_argument("--prominence-student", type=Path,
                   help="stage 4b student (train_prominence best.pt): predict word prominence "
                   "from the text and apply --emphasis-template to it (no marks needed)")
    p.add_argument("--prominence-threshold", type=float, default=0.5,
                   help="probability below which a word gets no emphasis; above it the "
                   "template strength ramps linearly up to --emphasis-template-scale")
    p.add_argument("--no-emphasis", action="store_true",
                   help="feed zero emphasis even when a row has marks")
    p.add_argument("--pp-max-dur", type=float,
                   help="cap (frames) on non-pause tokens, e.g. the data's 99.5th pct")
    p.add_argument("--pp-final-stretch", type=float, default=1.0,
                   help="duration multiplier for the last word (phrase-final lengthening)")
    p.add_argument("--config", type=Path, help="voice config.json for the phoneme map; "
                   "default: <eval-set dir>/config.json")
    args = p.parse_args()

    model = load_model(args.checkpoint, args.device, args.use_cfm, cfm_n_steps=args.cfm_n_steps)
    sample_rate = model.hparams.sample_rate
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.eval_set, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    pros = torch.load(args.prosody, weights_only=False) if args.prosody else None
    if pros is not None:
        from ..prosody.features import feature_matrix

    predictor = None
    if args.prosody_predictor:
        from ..prosody.predictor import ProsodyPredictor, text_hidden

        predictor = ProsodyPredictor.load(args.prosody_predictor, args.device)
        t = [float(v) for v in args.pp_temperature.split(",")]
        t_dur, t_pros = (t[0], t[0]) if len(t) == 1 else t
        # per-channel noise scale: [duration, lf0, vfrac, energy]
        pp_temp = torch.tensor([t_dur, t_pros, t_pros, t_pros], device=args.device).view(1, 4, 1)
        g = [float(v) for v in args.pp_guidance.split(",")]
        pp_guid = g[0] if len(g) == 1 else [g[0], g[1], g[1], g[1]]
        id_to_symbol = None
        if args.pp_max_dur or args.pp_final_stretch != 1.0 or args.pp_voiceless_fix or args.pp_voicing_fix:
            from ..prosody.predictor import fix_voiceless, fix_voicing, shape_durations

            cfg = json.load(open(args.config or args.eval_set.parent / "config.json", encoding="utf-8"))
            id_to_symbol = {v[0]: k for k, v in cfg["phoneme_id_map"].items()}

    emph_db = torch.load(args.emphasis, weights_only=False)["emphasis"] if args.emphasis else None
    _phz = None

    def row_emphasis(row, n_tok, n_ch):
        """[1, n_ch, T] per-token emphasis features for this row, or None."""
        nonlocal _phz
        if n_ch == 0:
            return None
        if args.no_emphasis:
            return torch.zeros(1, n_ch, n_tok, device=args.device)
        if emph_db is not None and row.get("utt") in emph_db:
            e = emph_db[row["utt"]]
        elif row.get("marked"):
            from ..eval.phonemize import PiperPhonemizer
            from ..prosody.emphasis import emphasis_features, emphasis_vector

            if _phz is None:
                _phz = PiperPhonemizer(args.config or args.eval_set.parent / "config.json")
            e_ids, e = (emphasis_features if n_ch > 1 else emphasis_vector)(row["marked"], _phz)
            assert e_ids == row["phoneme_ids"], row["id"]
        else:
            return None
        e = torch.from_numpy(e).reshape(1, -1, n_tok).to(args.device)
        assert e.size(1) == n_ch, f"emphasis has {e.size(1)} channels, model expects {n_ch}"
        return e

    template = json.load(open(args.emphasis_template)) if args.emphasis_template else None
    student = None
    if args.prominence_student:
        from ..prosody.prominence import ProminenceHead, pool_words

        assert template is not None and "anchor" in template, "--prominence-student needs a stress-anchored --emphasis-template"
        _ck = torch.load(args.prominence_student, map_location=args.device, weights_only=False)
        student = ProminenceHead(**_ck["kw"]).to(args.device).eval()
        student.load_state_dict(_ck["state_dict"])
    _phz_t = [None]
    dp_emb = None
    if args.dp_style_ckpt:
        from ..prosody.train_dp_style import load_style_dp, style_cond

        dp_emb, dp_styles = load_style_dp(args.dp_style_ckpt, model.model_g, args.device)

    torch.manual_seed(args.seed)
    manifest, total_audio, total_wall = [], 0.0, 0.0
    scales = [args.noise_scale, args.length_scale, args.noise_scale_w]
    for row in rows:
        ids = torch.LongTensor(row["phoneme_ids"]).unsqueeze(0).to(args.device)
        lengths = torch.LongTensor([ids.size(1)]).to(args.device)
        kw = {}
        if pros is not None:
            utt = pros["utts"][row["utt"]]
            assert len(utt["dur"]) == ids.size(1), row["id"]
            if model.hparams.get("prosody_channels", 0) > 0 and not args.no_prosody_input:
                feats = feature_matrix(utt, pros["stats"])
                kw["prosody"] = torch.from_numpy(feats).t().unsqueeze(0).to(args.device)
            if args.gt_durations:
                kw["durations"] = torch.from_numpy(utt["dur"].astype("float32")).unsqueeze(0).to(args.device)
        t0 = time.perf_counter()
        n_ch = max(predictor.token_feats if predictor is not None else 0,
                   getattr(dp_emb, "n_emph", 0) if dp_emb is not None else 0)
        emph = row_emphasis(row, ids.size(1), n_ch)
        if dp_emb is not None:
            name = None
            if args.pp_style:
                name = row.get("category") if args.pp_style in ("auto", "label") else args.pp_style
            with torch.no_grad():
                kw["dp_cond"] = style_cond(
                    dp_emb, dp_styles, name, args.device,
                    None if (emph is None or template is not None) else emph * args.emphasis_dur_scale,
                )
        if predictor is not None:
            x, x_mask = text_hidden(model.model_g, ids, lengths)
            style = None
            if args.pp_style:
                name = row.get("category") if args.pp_style in ("auto", "label") else args.pp_style
                style = torch.LongTensor([predictor.style_index(name)]).to(args.device)
            y = predictor.sample(
                x, x_mask, n_steps=args.pp_steps, temperature=pp_temp,
                solver=args.pp_solver, style=style, guidance=pp_guid,
                feats=emph if (predictor.token_feats and template is None) else None,
                feats_guidance=args.emphasis_guidance,
            )
            kw["durations"], kw["prosody"] = predictor.decode(y, x_mask)
            if args.pp_voicing_fix:
                kw["prosody"] = fix_voicing(kw["prosody"], ids.tolist(), id_to_symbol)
            elif args.pp_voiceless_fix:
                kw["prosody"] = fix_voiceless(kw["prosody"], ids.tolist(), id_to_symbol)
            if template is not None and (emph is not None or student is not None):
                from ..prosody.emphasis import apply_template, word_runs

                if "anchor" in template:  # stress-anchored parametric template
                    from ..eval.phonemize import PiperPhonemizer
                    from ..prosody.emphasis import apply_param_template, parse_marked, token_word_index, tonic_offsets

                    if _phz_t[0] is None:
                        _phz_t[0] = PiperPhonemizer(args.config or args.eval_set.parent / "config.json")
                    inv_ = {v: k for k, v in _phz_t[0].id_map.items()}
                    sname = row.get("category") if args.pp_style in ("auto", "label") else args.pp_style
                    if student is not None:  # stage 4b: prominence predicted from the text
                        plain, _ = parse_marked(row.get("marked") or row["text"])
                        t_ids, tok_w = token_word_index(plain, _phz_t[0])
                        assert t_ids == row["phoneme_ids"], row["id"]
                        with torch.no_grad():
                            tl = student(x, x_mask)[0]
                            n_w = max(tok_w) + 1
                            prob = torch.sigmoid(pool_words(tl, torch.LongTensor(tok_w).to(tl.device), n_w)).cpu().numpy()
                        thr = args.prominence_threshold
                        strength = np.clip((prob - thr) / max(1e-6, 1 - thr), 0.0, 1.0)
                        flags = (strength > 0).tolist()
                        row["_prominence"] = [round(float(v), 3) for v in prob]
                    else:
                        assert row.get("marked"), "the stress-anchored template needs a 'marked' text field"
                        plain, flags = parse_marked(row["marked"])
                        t_ids, tok_w = token_word_index(plain, _phz_t[0])
                        strength = np.ones(len(flags))
                    offs = tonic_offsets(t_ids, tok_w, inv_, flags)
                    p_st, d_mul = apply_param_template(
                        offs, template, args.emphasis_template_scale, sname,
                        symbols=[inv_.get(int(t), "?") for t in t_ids],
                    )
                    tok_s = np.array([strength[w] if w >= 0 else 0.0 for w in tok_w], np.float32)
                    p_st = p_st * tok_s
                    d_mul = np.power(d_mul, tok_s).astype(np.float32)
                else:
                    e_np = emph[0].cpu().numpy()
                    rel = word_runs(e_np[0], e_np[1] if e_np.shape[0] > 1 else None)
                    p_st, d_mul = apply_template(rel, template, args.emphasis_template_scale)
                st_ = predictor.stats
                lf0, vf = kw["prosody"][:, 0], kw["prosody"][:, 1]
                shift = torch.from_numpy(p_st).to(lf0) * (math.log(2) / 12) / st_["lf0_std"]
                kw["prosody"][:, 0] = torch.where(vf > 0.5, lf0 + shift, lf0)
                with torch.no_grad():
                    logw = model.model_g.dp(x, x_mask, reverse=True, noise_scale=args.noise_scale_w,
                                            h_add=kw.pop("dp_cond", None))
                # infer() does round(durations * length_scale); the normal SDP path
                # does ceil(w * length_scale).  Pre-apply ceil so both agree
                # (otherwise every token loses ~half a frame on average).
                ls = args.length_scale
                w = (torch.exp(logw) * x_mask)[:, 0] * torch.from_numpy(d_mul).to(logw)
                kw["durations"] = torch.ceil(w * ls) / ls * x_mask[:, 0]
            elif args.pp_durations == "sdp":
                del kw["durations"]
            elif args.pp_max_dur or args.pp_final_stretch != 1.0:
                kw["durations"] = shape_durations(
                    kw["durations"], ids.tolist(), id_to_symbol,
                    args.pp_max_dur, args.pp_final_stretch,
                )
        with torch.no_grad():
            audio = model(ids, lengths, scales, **kw).squeeze().float().cpu().numpy()
        wall = time.perf_counter() - t0
        wav = args.out_dir / f"{row['id']}.wav"
        sf.write(wav, audio, sample_rate)
        dur = len(audio) / sample_rate
        total_audio += dur
        total_wall += wall
        manifest.append({"id": row["id"], "text": row["text"], "wav": wav.name, "duration": dur})

    with open(args.out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} utterances, RTF={total_wall / max(total_audio, 1e-9):.3f}")


if __name__ == "__main__":
    main()
