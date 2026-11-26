# Bikol (Virac / Pandan) ⇄ English Machine Translation with LoRA

Group 1: Kissy Ann Villafuerte, Rhealyn Benavidez, Jamie Erica Samudio, Michaella Vargas

LoRA fine-tuning of `facebook/nllb-200-distilled-600M` on a 3-way parallel corpus about dengue
(English, Virac / South Bikol, Pandan / North Bikol). One adapter handles all six directions.
NLLB has no Bikol language code, so both dialects use `tgl_Latn` on the decoder side and are told apart
by a plain-text tag prepended to the source (`<2vir>` or `<2pan>`).

## Repository layout

| Path | What it is |
|---|---|
| `common.py` | Config, text cleaning, leakage-safe split |
| `train.py` | LoRA fine-tuning; saves `adapter/` and `results/loss_curve.*` |
| `main.py` | `stats`, `eval`, `latency`, `errors`, `translate` subcommands |
| `data/` | Source `.xlsx`, `bikol_clean_corpus.csv` / `.json`, and `train` / `validation` / `test` CSVs |
| `results/` | Evaluation tables, latency, corpus stats, loss curve, error analysis |

## Reproduce

```bash
pip install -r requirements.txt
python -c "import common; common.prepare_corpus()"   # cleaning + split (CPU, seconds)
python train.py                                       # needs a GPU (about T4 level)
python main.py eval                                   # baseline vs fine-tuned on the test split
python main.py latency
python main.py errors                                 # then fill manual_category in results/error_analysis.csv
python main.py translate "Are you okay?" --src eng_Latn --tgt bcl_North
```

Seed is 42 throughout. Hyperparameters: LoRA r=16, alpha=32, dropout 0.05 on `q_proj`/`v_proj`
(2.36M trainable of 617M parameters, 0.38%), LR 3e-4 with linear decay and 10% warm-up, batch 32,
max length 64, up to 20 epochs with early stopping (patience 5) on validation loss.

## Data

- Source: 1,540 rows; 3 exact duplicates removed, leaving **1,537 triples**.
- Cleaning: Unicode NFC, HTML remnants removed, quote and punctuation spacing normalised.
- Split 70 / 15 / 15 = **1,076 / 231 / 230** triples. Rows sharing a sentence in *any* of the three
  languages are grouped and kept in the same split, so no sentence is shared across splits (asserted in code).
- Each triple yields 6 directed examples: 6,456 train and 1,386 validation examples.
- 30.3% of triples have identical Virac and Pandan text, which matters when reading the dialect-to-dialect scores.

## Results (test split, 230 triples per direction)

| Direction | BLEU base → tuned | chrF++ base → tuned | TER base → tuned |
|---|---|---|---|
| Pandan → Virac | 9.63 → 57.96 | 35.45 → 77.51 | 87.95 → 26.74 |
| Pandan → English | 17.43 → 38.94 | 38.88 → 56.44 | 86.47 → 50.74 |
| Virac → Pandan | 9.69 → 61.97 | 34.15 → 80.19 | 90.09 → 23.06 |
| Virac → English | 18.12 → 39.85 | 39.52 → 57.01 | 84.57 → 49.47 |
| English → Pandan | 7.35 → 25.31 | 29.47 → 47.32 | 94.31 → 62.51 |
| English → Virac | 7.83 → 22.20 | 31.21 → 46.75 | 92.46 → 67.87 |
| **Macro average** | **11.68 → 41.04** | **34.78 → 60.87** | **89.31 → 46.73** |

Baseline = same NLLB weights with the adapter disabled and no dialect tags (Bikol output is Tagalog).
Full table: `results/evaluation_matrix.csv`.

**Latency** (T4, batch 1, beam 4, Pandan → English): mean 262 ms (1–3 words), 281 ms (4–6), 325 ms (7–9);
p95 404–425 ms. The 300 ms target is met on average for short inputs only. See `results/latency.csv`.

## Deliverables

- Cleaned corpus: `data/bikol_clean_corpus.csv` and `.json`
- Evaluation tables: `results/`
- LoRA adapter weights (9.5 MB): **TODO add Hugging Face link here after uploading the `adapter/` folder from the Colab run**

## Limitations

- Small, single-domain (dengue) corpus of short sentences (mean about 4.5 words); results will not generalise to other domains or long text.
- Dialect-to-dialect scores are inflated by the 30% identical pairs.
- Error categories in `results/error_analysis.csv` are heuristic suggestions until labelled by hand.
