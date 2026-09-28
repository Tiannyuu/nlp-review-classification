# NLP Review Classification: Recommendation & Star Rating Prediction

Two text-classification tasks on women's clothing e-commerce reviews: predict whether a
reviewer **recommends** the product (binary) and predict the exact **star rating 1–5**
(multi-class) — from raw review text with a TF-IDF + Logistic Regression pipeline.

**Key results:** Task 1 — accuracy **0.9038**, weighted F1 **0.9077** · Task 2 — accuracy
**0.6605**. Both above the challenge's full-score thresholds.

Individual project for INF2179 (University of Toronto) — Tianyu Wang.

## Business problem

E-commerce platforms sit on huge volumes of unstructured review text. Automating
recommendation detection helps monitor customer satisfaction at scale, and star-rating
prediction fills in structured ratings where they're missing.

## Dataset

Women's clothing e-commerce reviews: review title, review text, star rating, and a
recommendation indicator — split into train/test sets.

The CSVs are not committed to this repo — see [`data/README.md`](data/README.md).

## Methods

1. **Text preprocessing** — lowercase, strip non-letter characters, normalize whitespace;
   title + review text combined into a single feature.
2. **TF-IDF vectorization** of the combined text (captures sentiment-bearing words and
   short phrases).
3. **Logistic Regression** for both tasks — strong, reproducible baseline for sparse text
   features; fixed random state.
4. **Evaluation** — Task 1: weighted F1 (primary), precision/recall/F1/accuracy;
   Task 2: accuracy (primary) with per-class breakdown.

## Key findings

| Task | Metric | Result |
|---|---|---|
| 1 — Recommend / not (binary) | Accuracy / weighted F1 | **0.9038 / 0.9077** |
| 2 — Star rating 1–5 (multi-class) | Accuracy | **0.6605** |

- Task 1 shines on the majority "recommended" class (precision 0.97, recall 0.91);
  negative reviews are harder (precision 0.69) — positive language is more consistent.
- Task 2: 5-star reviews are easiest (clear positive patterns); 1–2 star reviews suffer
  low recall (rare + varied wording); 3–4 stars are muddled by mixed sentiment.
- Honest limitation noted in the report: text-only features miss product category and
  user behavior; TF-IDF can't capture sarcasm or subtle tone — future work points at
  deeper language models.

Full write-up: [`docs/challenge_report.pdf`](docs/challenge_report.pdf)

## Repo structure

```
├── notebooks/
│   └── nlp_review_classification.ipynb   # full analysis (rendered with outputs on GitHub)
├── data/
│   └── README.md                         # where to get the datasets
├── docs/
│   └── challenge_report.pdf              # challenge report
```

## How to run

```
# 1. download the datasets into data/ (see data/README.md)
#    -> data/train_data.csv
#    -> data/test_data.csv

# 2. install dependencies
pip install pandas scikit-learn jupyter

# 3. open the notebook
jupyter notebook notebooks/nlp_review_classification.ipynb
```

Run cells top to bottom. The notebook loads data from `../data/` relative to the
notebook location.

## Tech stack

Python · pandas · scikit-learn (TfidfVectorizer, LogisticRegression) · Jupyter
