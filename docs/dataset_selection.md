# Dataset Selection

## First-Paper Rule

Use:

- Existing benchmark dataset
- Existing AI APIs
- Small evaluation framework
- Hallucination and citation analysis

Avoid:

- Training a new model
- Scraping random websites
- Using copyrighted internal documents
- Creating a huge custom dataset
- Overcomplicating the first paper

## Recommended Dataset Path

### Primary Dataset: HotpotQA

Use HotpotQA first.

Why:

- It includes questions and gold answers.
- It includes sentence-level supporting facts.
- It is designed for multi-hop question answering.
- Supporting facts make it useful for citation and grounding evaluation.

How we use it:

- Convert context sentences into source sentence IDs.
- Use the gold answer as `expected_answer`.
- Use supporting facts as `expected_citation`.
- Compare `no_rag`, `basic_rag`, and `strict_citation_rag`.

### Secondary Extension: FEVER or Climate-FEVER

Use FEVER-style datasets after the HotpotQA pilot if we want stronger hallucination/refusal analysis.

Why:

- They classify claims as supported, refuted, or not enough information.
- They are naturally connected to hallucination and evidence support.
- They are useful for testing whether the model makes unsupported claims.

### Retrieval Extension: BEIR

Use BEIR when the paper needs a stronger retrieval section.

Why:

- It is a heterogeneous retrieval benchmark.
- It includes multiple retrieval tasks and domains.
- It can help measure whether bad answers come from bad retrieval or bad generation.

### Optional Later Dataset: Natural Questions

Natural Questions is useful for open-domain QA, but it should not be the first dataset.

Reason:

- It is larger and more complex.
- Licensing and redistribution details should be checked carefully before publishing subsets.

### Optional Later Dataset: LibreEval

LibreEval may be useful as a RAG hallucination benchmark, but it should be checked after the HotpotQA pilot.

Reason:

- It is closer to hallucination evaluation.
- We should review access, license, format, and whether it fits our research question before adopting it.

## First Experiment Design

Start with:

- Dataset: HotpotQA
- Sample size: 50 examples next, then 100
- Models: existing AI APIs
- Retrieval settings: gold evidence, keyword retrieval, optional embedding retrieval
- Prompt settings: no-RAG, basic-RAG, strict-citation-RAG

Main metrics:

- Answer accuracy
- Partial-credit answer score
- Citation accuracy
- Partial-credit citation score
- Hallucination rate
- Faithfulness / grounding

Secondary metric:

- Refusal accuracy, added later with FEVER-style or unanswerable data

## Why This Is Research

The contribution is not a new model.

The contribution is:

> A reproducible evaluation framework and analysis of hallucination and citation accuracy in RAG-based AI assistants using existing benchmark datasets and existing AI APIs.
