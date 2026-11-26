"""Shared configuration, data preparation and translation helpers.

Used by train.py and main.py. Mirrors the logic of the original Colab notebook.
"""
import collections
import json
import random
import re
import unicodedata

import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SEED = 42
EPOCHS = 20
LORA_R, LORA_ALPHA, LORA_DROPOUT, LR = 16, 32, 0.05, 3e-4
BATCH, MAX_LEN = 32, 64

XLSX = "data/Dengue_Villafuerte_Benavidez_Samudio_Vargas.xlsx"
DATA_DIR = "data"
RESULTS_DIR = "results"
ADAPTER_DIR = "adapter"

# Language keys -> spreadsheet column short names
COL = {"eng_Latn": "en", "bcl_South": "vi", "bcl_North": "pa"}
# NLLB has no Bikol code, so both dialects reuse the Tagalog code (tgl_Latn)
NLLB = {"eng_Latn": "eng_Latn", "bcl_South": "tgl_Latn", "bcl_North": "tgl_Latn"}
# Plain-text dialect tags prepended to the source so one decoder language code serves both dialects
TAG = {"bcl_South": "<2vir>", "bcl_North": "<2pan>"}
DIRECTIONS = [(s, t) for s in COL for t in COL if s != t]
LABELS = {"English": "eng_Latn", "Pandan (North)": "bcl_North", "Virac (South)": "bcl_South"}


def prep_src(text, tgt, tagged=True):
    return f"{TAG[tgt]} {text}" if tagged and tgt in TAG else text


def tok_re(s):
    return re.findall(r"[\w']+", s.lower())


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"<[^>]+>|&\w+;", "", s)  # HTML remnants
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.?!,:;])", r"\1", s)  # no space before punctuation
    s = re.sub(r"(?<=[a-z])([.?!])(?=[A-Z])", r"\1 ", s)  # "Yes.I" -> "Yes. I"
    s = re.sub(r"([,:;])(?=\S)(?!\d)", r"\1 ", s)
    return s


def prepare_corpus(xlsx=XLSX, out_dir=DATA_DIR, seed=SEED):
    """Clean the spreadsheet, drop exact duplicates, and make a leakage-safe 70/15/15 split.

    Rows that share the same sentence in ANY of the three languages are merged into one group and
    the whole group goes to a single split, so no sentence appears in two splits.
    Writes bikol_clean_corpus.csv/.json plus train/validation/test CSVs to out_dir.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_excel(xlsx).iloc[:, :3]
    raw.columns = ["en", "vi", "pa"]
    raw = raw.astype(str)

    df = raw.map(norm)
    df["src_row"] = df.index + 2  # row number in the spreadsheet (header is row 1)
    n0 = len(df)
    df = df[~df[["en", "vi", "pa"]].apply(lambda c: c.str.lower()).duplicated()].reset_index(drop=True)
    print(f"removed {n0 - len(df)} exact duplicate rows -> {len(df)} pairs")

    # union-find over rows that share a sentence in any language
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in ("en", "vi", "pa"):
        first = {}
        for i, v in enumerate(df[c].str.lower()):
            if v in first:
                parent[find(i)] = find(first[v])
            else:
                first[v] = i
    groups = collections.defaultdict(list)
    for i in range(len(df)):
        groups[find(i)].append(i)
    gl = list(groups.values())
    random.Random(seed).shuffle(gl)

    N = len(df)
    cnt = collections.Counter()
    split = {}
    for g in gl:
        s = "train" if cnt["train"] < 0.70 * N else "validation" if cnt["validation"] < 0.15 * N else "test"
        for i in g:
            split[i] = s
        cnt[s] += len(g)
    df["split"] = [split[i] for i in range(N)]
    df.insert(0, "id", [f"bik-{i + 1:04d}" for i in range(N)])
    df["vi_equals_pa"] = (df.vi.str.lower() == df.pa.str.lower()).astype(int)

    df.to_csv(f"{out_dir}/bikol_clean_corpus.csv", index=False, encoding="utf-8-sig")
    with open(f"{out_dir}/bikol_clean_corpus.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=1)
    for s in ("train", "validation", "test"):
        df[df.split == s].to_csv(f"{out_dir}/{s}.csv", index=False, encoding="utf-8-sig")

    S = {s: df[df.split == s] for s in ("train", "validation", "test")}
    leaks = sum(
        len(set(S[a][c].str.lower()) & set(S[b][c].str.lower()))
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        for c in ("en", "vi", "pa")
    )
    print(dict(cnt), "| sentences shared across splits:", leaks)
    assert leaks == 0, "train/validation/test leakage detected"
    return df


def load_corpus(data_dir=DATA_DIR):
    return pd.read_csv(f"{data_dir}/bikol_clean_corpus.csv", encoding="utf-8-sig", keep_default_na=False)
