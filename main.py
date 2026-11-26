"""Evaluate, translate and analyse errors for the Bikol (Virac / Pandan) LoRA model.

Subcommands:
    python main.py eval                       # zero-shot baseline vs fine-tuned on the test split
    python main.py latency                    # per-length latency (batch 1, beam 4)
    python main.py errors                     # 50 worst fine-tuned test outputs -> CSV for manual labelling
    python main.py translate "Hello" --src eng_Latn --tgt bcl_North
    python main.py stats                      # corpus statistics (no GPU/model needed)
"""
import argparse
import collections
import difflib
import statistics
import time

import pandas as pd

import common as C


def load_model(adapter, device):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSeq2SeqLM.from_pretrained(C.BASE_MODEL)
    model = PeftModel.from_pretrained(base, adapter).to(device).eval()
    return tok, model


def translate(tok, model, device, texts, s, t, tagged=True, beams=4, bs=32):
    import torch
    with torch.no_grad():
        return _translate(tok, model, device, texts, s, t, tagged, beams, bs)


def _translate(tok, model, device, texts, s, t, tagged, beams, bs):
    tok.src_lang = C.NLLB[s]
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([C.prep_src(x, t, tagged) for x in texts[i:i + bs]], return_tensors="pt",
                  padding=True, truncation=True, max_length=C.MAX_LEN).to(device)
        g = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(C.NLLB[t]),
                           num_beams=beams, max_new_tokens=64)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def cmd_stats(_):
    df = C.load_corpus()
    rows = []
    for c, name in (("en", "English"), ("vi", "Virac (South)"), ("pa", "Pandan (North)")):
        L = [len(C.tok_re(s)) for s in df[c]]
        T = [t for s in df[c] for t in C.tok_re(s)]
        ct = collections.Counter(T)
        rows.append(dict(language=name, tokens=len(T), types=len(ct), TTR=round(len(ct) / len(T), 3),
                         hapax=sum(v == 1 for v in ct.values()), mean_len=round(statistics.mean(L), 2),
                         median_len=statistics.median(L), max_len=max(L)))
    out = pd.DataFrame(rows)
    out.to_csv(f"{C.RESULTS_DIR}/corpus_stats.csv", index=False)
    print(out.to_string(index=False))
    vv = {t for s in df.vi for t in C.tok_re(s)}
    vp = {t for s in df.pa for t in C.tok_re(s)}
    print(f"Virac-Pandan vocabulary: {len(vv & vp)} shared types, Jaccard {len(vv & vp) / len(vv | vp):.3f}")
    print(f"Identical Virac/Pandan pairs: {df.vi_equals_pa.sum()} ({df.vi_equals_pa.mean():.1%})")


def cmd_eval(a):
    import torch
    from sacrebleu.metrics import BLEU, CHRF, TER
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(a.adapter, dev)
    df = C.load_corpus()
    test = df[df.split == "test"].reset_index(drop=True)
    bleu, chrf, ter = BLEU(), CHRF(word_order=2), TER()  # chrF++ = chrF with word_order=2
    res, preds = [], []
    for name in ("baseline", "finetuned"):
        for s, t in C.DIRECTIONS:
            src, ref = test[C.COL[s]].tolist(), test[C.COL[t]].tolist()
            if name == "baseline":  # same weights, adapter off, no dialect tags -> Bikol comes out as Tagalog
                with model.disable_adapter():
                    hyp = translate(tok, model, dev, src, s, t, tagged=False)
            else:
                hyp = translate(tok, model, dev, src, s, t, tagged=True)
            res.append(dict(model=name, direction=f"{s}->{t}", BLEU=bleu.corpus_score(hyp, [ref]).score,
                            chrFpp=chrf.corpus_score(hyp, [ref]).score, TER=ter.corpus_score(hyp, [ref]).score))
            preds += [dict(model=name, id=i, direction=f"{s}->{t}", source=x, reference=y, hypothesis=z)
                      for i, x, y, z in zip(test.id, src, ref, hyp)]
    R = pd.DataFrame(res).round(2)
    pd.DataFrame(preds).to_csv(f"{C.RESULTS_DIR}/predictions.csv", index=False, encoding="utf-8-sig")
    matrix = R.pivot(index="direction", columns="model", values=["BLEU", "chrFpp", "TER"])
    matrix.loc["MACRO-AVG"] = matrix.mean()
    matrix = matrix.round(2)
    matrix.to_csv(f"{C.RESULTS_DIR}/evaluation_matrix.csv")
    print(matrix)


def cmd_latency(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(a.adapter, dev)
    test = C.load_corpus().query("split == 'test'")
    buckets = {"1-3": (1, 3), "4-6": (4, 6), "7-9": (7, 9), "10+": (10, 999)}
    lat = {k: [] for k in buckets}
    translate(tok, model, dev, ["warm up"], "eng_Latn", "bcl_North")
    for x in test.pa.tolist()[:200]:
        n = len(x.split())
        for k, (lo, hi) in buckets.items():
            if lo <= n <= hi:
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                translate(tok, model, dev, [x], "bcl_North", "eng_Latn")
                if dev == "cuda":
                    torch.cuda.synchronize()
                lat[k].append((time.perf_counter() - t0) * 1000)
    L = pd.DataFrame([dict(words=k, n=len(v), mean_ms=round(statistics.mean(v)),
                           p95_ms=round(sorted(v)[max(0, int(.95 * len(v)) - 1)]))
                      for k, v in lat.items() if v])
    L.to_csv(f"{C.RESULTS_DIR}/latency.csv", index=False)
    print(L.to_string(index=False), f"\ndevice: {dev} (target < 300 ms)")


def suggest(src, ref, hyp):
    """Heuristic error category. A starting point only: review and fill manual_category by hand."""
    s, r, h = C.tok_re(src), C.tok_re(ref), C.tok_re(hyp)
    if not h:
        return "Untranslated (OOV)"
    if h != r and sorted(h) == sorted(r):
        return "Word order / Syntax"
    if h != r and len(set(h) & set(r)) / max(1, len(set(h) | set(r))) >= .6 \
            and [t for t in h if t in r] != [t for t in r if t in h]:
        return "Word order / Syntax"
    wrong = [t for t in h if t not in r]
    if wrong and sum(t in s for t in wrong) / len(wrong) >= .5:
        return "Untranslated (OOV)"
    for w in wrong:
        for m in [t for t in r if t not in h]:
            if difflib.SequenceMatcher(None, w, m).ratio() >= .6:
                return "Morphological / Affix"
    return "Other lexical"


def cmd_errors(_):
    P = pd.read_csv(f"{C.RESULTS_DIR}/predictions.csv", encoding="utf-8-sig", keep_default_na=False)
    F = P[P.model == "finetuned"].copy()
    F["sim"] = [difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() for x, y in zip(F.reference, F.hypothesis)]
    worst = F[F.sim < 1].sort_values("sim").head(50).copy()
    worst["suggested_category"] = [suggest(x, y, z) for x, y, z in zip(worst.source, worst.reference, worst.hypothesis)]
    worst["manual_category"] = ""
    worst["notes"] = ""
    worst.drop(columns="model").to_csv(f"{C.RESULTS_DIR}/error_analysis.csv", index=False, encoding="utf-8-sig")
    print(worst.suggested_category.value_counts())


def cmd_translate(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(a.adapter, dev)
    if a.src == a.tgt:
        raise SystemExit("--src and --tgt must differ")
    t0 = time.perf_counter()
    y = translate(tok, model, dev, [a.text], a.src, a.tgt)[0]
    print(f"{y}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("stats", cmd_stats), ("eval", cmd_eval), ("latency", cmd_latency), ("errors", cmd_errors)):
        p = sub.add_parser(name)
        p.add_argument("--adapter", default=C.ADAPTER_DIR)
        p.set_defaults(fn=fn)
    p = sub.add_parser("translate")
    p.add_argument("text")
    p.add_argument("--src", default="eng_Latn", choices=list(C.COL))
    p.add_argument("--tgt", default="bcl_North", choices=list(C.COL))
    p.add_argument("--adapter", default=C.ADAPTER_DIR)
    p.set_defaults(fn=cmd_translate)
    args = ap.parse_args()
    args.fn(args)
