# Hybrid News Recommender

A research-oriented news recommendation project comparing popularity, collaborative filtering, semantic content-based filtering, weighted hybrids, adaptive gating, and a neural NRMS implementation. The emphasis is on **comparable evaluation, cold-start behavior, and recognizing when added model complexity does not help**.

Developed by Lorenzo Cozzolino.

## Scope and contributions

This university coursework project implements and evaluates the following approaches on the **Microsoft News Dataset (MIND-small)**:

| Component | Implementation | Notes |
| --- | --- | --- |
| Popularity baseline | `baseline_model/baseline.ipynb` | Historical click popularity; no personalization. |
| Item-based collaborative filtering | `cf_itemknn/` | Co-click ItemKNN, recency weighting, optional popularity fallback. |
| Latent collaborative filtering | `cf_latent_item/` | Implicit-feedback ALS and category fallback. |
| Semantic content-based filtering (CBF) | `content_based_text/` | Sentence-Transformer embeddings of article title/abstract. |
| Content-profile variant | `content_based_clicked/` | Semantic, category and recency signals. |
| Weighted hybrid | `hybrid/hybrid_recommender.py` | Candidate-set score normalization and tunable popularity/CF/CBF weights. |
| Adaptive hybrid | `hybrid/experiments/adaptive_hybrid/` | Rule-based weighting and an experimental neural gating model. |
| NRMS experiment | `hybrid/experiments/nrms/` | Neural news/user encoders with attention, trained from scratch. |

The project-specific contributions include implementing these components, evaluating score combinations, investigating cold-start definitions and candidate-catalog coverage, auditing potential evaluation artifacts, and testing whether adaptive or neural alternatives improve upon the strongest CBF comparator. The underlying algorithms and model families originate from prior research; this repository does not claim to have invented ItemKNN, ALS, Sentence-BERT or NRMS.

## Repository layout

```text
baseline_model/              Popularity baseline notebook
cf_itemknn/                  ItemKNN scripts and exploration
cf_latent_item/              ALS scripts and exploration
content_based_text/          Sentence-Transformer CBF notebook
content_based_clicked/       Combined-content profile notebook
hybrid/
  hybrid_recommender.py      Weighted hybrid evaluator
  experiments/              Methodology audits, gating and NRMS
data_analysis.ipynb          MIND exploratory analysis (outputs cleared)
demo/                        Synthetic-only ItemKNN smoke demo
EXPERIMENTAL_PIPELINE.md     Research phase index and provenance
requirements.txt             Original project dependencies
```

**Not included:** MIND data, precomputed embeddings and interaction matrices, model checkpoints, historic prediction files, notebook outputs, and the academic report PDF. Those exclusions are intentional; the original private research repository remains unchanged. Some reports refer to derived results or files absent from this distribution.

## Install

Python 3.10+ is used by the original codebase. In an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The dependency list reflects the original project rather than a locked and independently certified environment. Some experiments require additional packages, notably **PyTorch for NRMS**; the full pipeline is not guaranteed to run from `requirements.txt` alone. Do not execute notebooks or download pretrained models before reviewing their resource requirements.

## Offline synthetic smoke demo

A small deterministic dataset is **generated locally and contains no Microsoft News records, real users, URLs, or article text**:

```bash
python3 demo/generate_synthetic_behaviors.py --output demo/generated
python3 cf_itemknn/itemknn_recency_fallback.py \
  --train-behaviors demo/generated/train_behaviors.tsv \
  --dev-behaviors demo/generated/dev_behaviors.tsv \
  --limit-dev 20
```

For an **end-to-end smoke test of the weighted hybrid** (popularity, CF and CBF, using deterministic **random numeric vectors instead of SBERT**):

```bash
python3 demo/run_hybrid_synthetic.py --limit 20
```

This runner makes temporary MIND-shaped synthetic news/behavior files and embeddings and removes them on exit. It does not download any external model or dataset. The generator uses only the Python standard library; the hybrid smoke test requires NumPy, Polars, SciPy, scikit-learn and tqdm from the original dependency list. Both demos run without network calls. The ItemKNN evaluator reads the synthetic fixtures and performs a basic functional check. **Its metrics are NOT research results and must never be compared with the MIND results below.** The synthetic hybrid smoke test checks its own local execution path, but neither demo establishes research-data reproducibility, semantic embedding quality, adaptive gating or NRMS performance.

## Authorized MIND research experiments

MIND-small is distributed by Microsoft for research under its own conditions. **The dataset is not redistributed in this portfolio repository**, including `behaviors.tsv`, `news.tsv`, entity/relation embeddings and user/impression histories. Read the current [MIND project page and license](https://msnews.github.io/) and [dataset description](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md). Obtain data yourself through the official download procedure and only after personally reviewing and accepting any applicable terms. This repository does not accept terms on your behalf.

For the original directory conventions, place separately acquired authorized files under:

```text
dataset/MINDsmall_train/{behaviors.tsv,news.tsv,entity_embedding.vec,relation_embedding.vec}
dataset/MINDsmall_dev/{behaviors.tsv,news.tsv,entity_embedding.vec,relation_embedding.vec}
```

The full hybrid needs precomputed embeddings corresponding to the *same article ordering* as `MINDsmall_dev/news.tsv`:

```text
content_based_text/embeddings/news_embeddings.npy
```

The CBF notebook documents creation of Sentence-Transformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`; the model is offered under [Apache-2.0 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Download pretrained resources yourself under their individual licenses; no weights are bundled here. Do not assume that regenerating one asset automatically restores every experiment: the latent-CF pipeline also uses derived matrices/ID order, and adaptive/NRMS comparison stages use intermediate result artifacts that are not distributed.

With authorized MIND data **and matching regenerated embedding files**, the original weighted hybrid's command-line entry point is:

```bash
python3 hybrid/hybrid_recommender.py --limit 2000
```

For a different combination:

```bash
python3 hybrid/hybrid_recommender.py --limit 2000 \
  --baseline-weight 0.0 --cf-weight 0.1 --cbf-weight 0.9
```

These are original code entry points, **not** verified complete reproduction instructions in this sanitized distribution. They are expected to fail if required excluded inputs are absent. Additional experimental scripts and chronological notes are described in [EXPERIMENTAL_PIPELINE.md](EXPERIMENTAL_PIPELINE.md).

## Research results and evaluation

The figures below are **historical results reported in the original project documentation**, not results of the synthetic demo and not independently re-computed during portfolio preparation. Distinct rows and experiments can use different impression subsets; do not interpret the tables as a single equal-protocol leaderboard.

| Initial MIND-small model comparison (development impressions) | Reported AUC |
| --- | ---: |
| ItemKNN, pure collaborative | 0.5010 |
| ItemKNN with popularity fallback | 0.5318 |
| ALS latent-item CF | 0.5229 |
| Pure text-based SBERT CBF | **0.6418** |
| CBF with metadata and recency | 0.6307 |

For an explicitly shared subset of candidate impressions, the separate [NRMS report](hybrid/experiments/nrms/NRMS_REPORT.md) records:

| Common evaluation set | NRMS AUC | CBF AUC | Evaluated impressions |
| --- | ---: | ---: | ---: |
| Temporal validation | 0.5979 | **0.6180** | 26,078 |
| Internal test | 0.5901 | **0.6325** | 27,002 |

The common subset removes candidates lacking a matching precomputed CBF embedding: substantial selection effects remain, especially in validation. NRMS uses train-catalog title coverage while the CBF uses a dev-catalog embedding index. This asymmetry required filtering to a common evaluation subset. The adaptive experiments likewise do not provide persuasive evidence that gating materially improves over pure CBF in the reported setup.

The original research pipeline investigates five distinct cold-start definitions, score ties, popularity information sources, temporal ordering, and candidate catalog coverage. See [methodology audit](hybrid/experiments/VALIDAZIONE_METODOLOGICA.md), [adaptive report](hybrid/experiments/adaptive_hybrid/ADAPTIVE_HYBRID_REPORT.md), and [NRMS report](hybrid/experiments/nrms/NRMS_REPORT.md).

**Evaluation caveats:** MIND-small's development split was reused for tuning and diagnosis in earlier work. The later FIT/VALIDATION/INTERNAL_TEST temporal protocol uses internal partitions of MIND-small TRAIN, but INTERNAL_TEST had been observed during prior analyses; it is not a wholly untouched final test set. Model comparisons must control for available article catalogs, candidate groups and candidate-set filtering. Reported differences do not demonstrate that a more complex architecture is better.

## Privacy, data and licensing

Microsoft's anonymized MIND identifiers and click histories are dataset records, not account credentials. They are nevertheless governed by the dataset's terms and are deliberately absent here. Do not commit personal data, tokens, exported user profiles, hidden notebook outputs or generated MIND-based artifacts. `.gitignore` excludes expected research data and output locations, but **ignoring a file never removes it from an existing Git history**; this portfolio snapshot contains no original Git history.

The code is an academic research implementation, **not a production service**. It has no deployed API, authentication layer, monitoring or operational guarantees. An offline demo does not validate every model.

See [rights and dataset policy](RIGHTS_AND_DATA.md) for the terms of the MIND dataset, externally developed models, and original source code.

The implementation in this portfolio is authored by Lorenzo Cozzolino. No repository-wide open-source license is granted: the project remains all-rights-reserved unless a license is added explicitly. **Do not infer the project's license from dependencies.** The original research papers, MIND dataset and pretrained model each have their own authorship and distribution conditions.

## Research references

- Fangzhao Wu et al. (2020), [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/).
- Chuhan Wu et al. (2019), [Neural News Recommendation with Multi-Head Self-Attention (NRMS)](https://aclanthology.org/D19-1671/).
- Nils Reimers and Iryna Gurevych (2019), [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).
