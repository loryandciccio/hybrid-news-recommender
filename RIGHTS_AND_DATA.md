# Authorship, redistribution and research-data policy

This repository contains the news recommendation implementation developed by **Lorenzo Cozzolino**, presented as a standalone portfolio project. It has an independent Git history: university presentation decks and reports from the earlier coursework repository are not included.

## Original code and license

The project-specific source code, experiments, notebooks, and analytic figures in this portfolio are presented as the author's original implementation of established recommender-system methods. Underlying techniques, datasets, libraries and research papers are credited to their respective authors; creating an implementation does not confer ownership of those external works.

**No repository-wide open-source license is granted.** A public GitHub repository is viewable but is not automatically licensed for reuse, modification, redistribution or commercial deployment. A license can be added by the author separately if desired.

## Microsoft MIND dataset

MIND-small is provided for research under Microsoft's conditions and is **not freely redistributable** simply because it is downloadable. This portfolio deliberately excludes MIND \`behaviors.tsv\`, \`news.tsv\`, entity/relation vectors, derived user/article lists, click matrices, article-content embeddings, and research checkpoints.

To reproduce research experiments, obtain and use the data yourself through the [official MIND project page](https://msnews.github.io/), after reading the current [documentation](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md) and applicable terms. Never commit licensed MIND data, content-derived artifacts, real user records or authentication information.

## Pretrained language models

The original content-based implementation refers to \`sentence-transformers/all-MiniLM-L6-v2\`, whose [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) lists Apache-2.0. This repository contains **no third-party pretrained weights**. Download any permitted models directly from their authorized publishers under their own terms.

## Results and scope

Numeric results documented from the university research run are **historical experimental records**, not outputs of the synthetic demonstrations. The local demos generate invented users, news identifiers, and numeric vectors. They neither measure performance on real news nor verify full MIND replication. Research comparisons have documented evaluation and catalog-coverage limitations.

The project is a research prototype, not a production-ready news service. Credit must remain with the original researchers for the concepts and algorithms cited in the [README](README.md).

## Checklist before changing visibility to public

- Keep licensed MIND datasets, checkpoint files and regenerated MIND-derived caches out of both the working tree and published history.
- Check that code, figures and reports being distributed are independently authored or appropriately licensed; the portfolio asserts original implementation authorship without asserting ownership of third-party algorithms.
- Avoid personal identifiers, API keys, notebook outputs, local account information and private university documents.
- Retain dataset/model provenance, limitations and citations; do not claim that a synthetic test reproduces the historical benchmarks.

This policy does not accept external licenses or grant rights in third-party works.
