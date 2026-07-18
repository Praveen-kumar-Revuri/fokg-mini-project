# KG-Based Fact Checking Engine

**Course:** Foundations of Knowledge Graphs — Summer Term 2026

**University:** Paderborn University, Data Science Group

**Authors:** Praveen Kumar Revuri (Matriculation Number : 4063312), Anuhya Velagaturi (Matriculation Number : 4054875)

**Local CV ROC AUC (SW 2022 Train, 5-fold):** **70.40%** — see [Results](#results) for why this, not a GERBIL number, is reported.

---

## For evaluators

To verify this project runs locally:

1. Clone/unzip the project. No large files are required — this engine has no Git-LFS dependencies.
2. Create a venv and install: `pip install -r requirements.txt`
3. Run: `python fact_checker.py`
4. Run: `python validate_result.py` — expect **PASS**, **1342** lines.

GERBIL is not required for a local check; step 3 regenerates `result.ttl` and also prints 5-fold cross-validation ROC AUC on the labelled training data.

---

## What this project does

This is a **knowledge-graph fact checker**. Given RDF statements about DBpedia entities, it predicts how likely each statement is to be **true**, returning a score between **0** (false) and **1** (true).

The pipeline:

1. Builds an **undirected statement graph** directly from the provided train + test files (subjects/objects as nodes, statements as edges) — no external background KG or embeddings are used.
2. Computes classic **topological link-prediction features** for each `(subject, predicate, object)` triple: common neighbours, Jaccard coefficient, Adamic/Adar index, Resource Allocation index, preferential attachment, node degree, and a predicate-restricted common-neighbours count.
3. Trains a **Gradient Boosted Decision Tree** classifier on the 1234 labelled SW-2022 training statements.
4. Writes predictions to `result.ttl` for evaluation on [GERBIL](http://gerbil-kbc.aksw.org/gerbil/config).

---

## Quick start

### 1. Set up the environment

```bash
python -m venv fokg
source fokg/bin/activate   # or fokg\Scripts\activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Generate predictions

```bash
python fact_checker.py
python validate_result.py
```

This builds the statement graph from `data/KG-2022-train.nt.txt` and `data/KG-2022-test.nt.txt`, computes structural features, trains the classifier on the labelled training facts, predicts all **1342** test facts, and writes `result.ttl`.

### 3. Evaluate on GERBIL

1. Open [GERBIL Fact Checking](http://gerbil-kbc.aksw.org/gerbil/config)
2. Experiment type: **Fact Checking** (not "Fact Checking SLC 2019")
3. Enter your real name and university email
4. Upload `result.ttl`
5. Select dataset **SW 2022 Test** (this is the score that counts)
6. Submit

> **Note:** `result.ttl` only contains **test** predictions. If you also run **SW 2022 Train** on GERBIL, the score will look broken — that benchmark expects training-fact predictions we intentionally do not output.

---

## Project structure

```
fokg-structural-factchecker/
├── fact_checker.py              # Main pipeline — run this
├── validate_result.py           # Checks result.ttl format and completeness
├── requirements.txt
├── result.ttl                   # Output predictions (1342 test facts)
│
└── data/
    ├── KG-2022-train.nt.txt     # 1234 labeled training statements
    └── KG-2022-test.nt.txt      # 1342 test statements (no labels)
```

---

## How it works (short)

| Step | What happens |
|------|----------------|
| Background graph | Undirected co-occurrence graph built from `KG-2022-train.nt.txt` + `KG-2022-test.nt.txt` structure only (labels excluded — no leakage) |
| Features | Common neighbours, Jaccard, Adamic/Adar, Resource Allocation, preferential attachment, node degree, predicate-restricted common neighbours (8 features total) |
| Classifier | Single Gradient Boosted Decision Tree (`n_estimators=100, max_depth=2, learning_rate=0.03`) |
| Output | One `hasTruthValue` triple per test statement, score as `xsd:double` in `[0, 1]` |

This is deliberately a **different algorithm family** from a TransE-embedding + classifier-ensemble approach: no knowledge-graph embeddings are trained at all, and the signal comes purely from graph topology instead of vector geometry.

---

## Input and output format

**Training/test facts** are RDF reified statements:

```turtle
<http://swc2017.aksw.org/task2/dataset/3892429>
    rdf:type rdf:Statement ;
    rdf:subject <http://dbpedia.org/resource/Venus_Williams> ;
    rdf:predicate <http://dbpedia.org/ontology/birthPlace> ;
    rdf:object <http://dbpedia.org/resource/Lynwood,_California> .
```

**Output** (`result.ttl`) — one line per test fact:

```turtle
<http://swc2017.aksw.org/task2/dataset/3892429> <http://swc2017.aksw.org/hasTruthValue> "0.8901"^^<http://www.w3.org/2001/XMLSchema#double> .
```

Every statement in `KG-2022-test.nt.txt` appears exactly once in `result.ttl`.

---

## Command-line options

```bash
# Default — builds graph from data/, writes result.ttl, prints CV AUC
python fact_checker.py

# Skip the cross-validation report (faster)
python fact_checker.py --skip-cv

# Custom file locations
python fact_checker.py --train-file data/KG-2022-train.nt.txt --test-file data/KG-2022-test.nt.txt --output-file result.ttl

# Validate output before uploading to GERBIL
python validate_result.py
```

---

## Requirements

- Python 3.10+
- No GPU, no large downloads — everything runs on the two provided data files

**Libraries:** rdflib, scikit-learn, numpy

---

## Results

This environment had no access to the real DBpedia background KG shipped in the original repo (`data/dbpedia-reference-kg.nt` and `trans-e-embeddings/trained_model.pkl` were Git-LFS pointer files, not real data) and no network access to `dbpedia.org` or `gerbil-kbc.aksw.org` to rebuild the KG or submit to GERBIL directly. So instead of a GERBIL screenshot, here is 5-fold stratified cross-validation on the 1234 labelled training statements:

```
fold 1: ROC AUC = 0.6725
fold 2: ROC AUC = 0.6771
fold 3: ROC AUC = 0.7320
fold 4: ROC AUC = 0.7238
fold 5: ROC AUC = 0.7145
mean ROC AUC = 0.7040  (std 0.0245)
```

| Metric | Value |
|--------|-------|
| Local 5-fold CV ROC AUC (SW 2022 Train) | **70.40%** |
| Test facts predicted | 1342 / 1342 |
| GERBIL ROC AUC (SW 2022 Test) | *not verified here — upload `result.ttl` to GERBIL to get the official number* |

