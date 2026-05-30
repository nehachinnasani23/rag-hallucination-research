# Research Proposal

## Title

Evaluating Hallucination and Citation Accuracy in RAG-Based Question Answering Using Gold and Retrieved Evidence

## Abstract

Retrieval-augmented generation (RAG) is widely used to reduce hallucinations in large language model assistants by grounding answers in external documents. However, RAG systems can still produce answers that are incorrect, unsupported by retrieved evidence, or accompanied by citations that do not actually justify the generated claims. This study proposes a small evaluation framework for measuring hallucination and citation accuracy using existing benchmark datasets such as HotpotQA, with optional later extension to BEIR-style retrieval benchmarks. Existing AI APIs will be evaluated under no-RAG, basic-RAG, and strict-citation-RAG conditions. The expected contribution is a reproducible benchmark-style evaluation workflow and error taxonomy for measuring whether RAG assistants provide correct, source-supported, and properly cited answers.

## Problem Statement

AI assistants are increasingly used to answer questions from company knowledge bases, policy documents, support articles, and technical manuals. RAG is intended to make these systems safer by retrieving relevant source passages before generating an answer. Yet retrieval alone does not guarantee reliability. A system may retrieve the right document but generate unsupported details, cite a source that does not support the claim, overgeneralize from partial evidence, or answer confidently when the documents do not contain enough information.

This creates a practical reliability problem for enterprise AI: users may trust responses because they include citations, even when the cited sources do not verify the answer. The proposed study addresses this gap by evaluating not only whether RAG answers are correct, but also whether their citations truly support the generated claims.

## Research Question

How does hallucination and citation accuracy change when RAG systems use gold supporting evidence versus automatically retrieved evidence?

## Research Objectives

1. Build a small benchmark subset from an existing dataset such as HotpotQA.
2. Evaluate RAG assistant outputs for answer accuracy, citation accuracy, faithfulness, hallucination rate, and refusal accuracy.
3. Compare hallucination patterns across multiple RAG configurations or models.
4. Develop an error taxonomy for unsupported claims and citation failures.
5. Release a reproducible evaluation workflow suitable for future extension.

## Background and Motivation

The original RAG formulation showed that combining parametric language models with retrieved non-parametric memory can improve factuality and provenance for knowledge-intensive tasks. Later work has shown that RAG evaluation remains difficult because systems must be judged across retrieval quality, answer quality, and grounding quality. RAGTruth demonstrates that RAG systems can still produce unsupported or contradictory claims, while citation-verifiability research shows that generated answers may appear trustworthy even when citations do not fully support the text. This study focuses on a practical enterprise setting: question answering over controlled knowledge-base documents.

## Methodology

### Dataset Creation

The first version will use **HotpotQA** because it already provides questions, gold answers, context passages, and sentence-level supporting facts. This avoids scraping random websites, using private documents, or creating a large new dataset. A later extension may use BEIR for retrieval-focused experiments.

The initial dataset subset will include 25 to 100 questions. Each example will contain:

- Benchmark example ID
- Question
- Gold answer
- Context sentences converted into source sentence IDs
- Expected supporting sentence IDs from benchmark supporting facts
- Answerability label

### Question Types

The first benchmark will include:

- Multi-hop questions requiring evidence from more than one supporting sentence
- Comparison questions where the model must compare two entities
- Citation-sensitive questions where the answer is only justified by specific supporting facts
- Retrieval-stress cases where the retrieved context may omit one of the needed facts

### RAG Setup

The initial experiment will use a simple RAG pipeline:

- Benchmark conversion into sentence-level source documents
- Keyword, embedding, or gold-evidence retrieval
- Top-k context selection
- LLM answer generation with required citations
- Standardized prompt template across models

The study may compare several conditions:

- Different LLMs using the same retrieved context
- Gold supporting facts versus retrieved context
- Different retrieval top-k values
- Keyword retrieval versus embedding retrieval
- With and without explicit refusal instructions

The main comparison for the first paper will be:

| Condition | Purpose |
|---|---|
| Gold supporting facts | Best-case RAG condition |
| Retrieved top-k passages | Realistic RAG condition |
| Distractor passages | Tests robustness |
| Missing evidence | Tests refusal behavior |
| Conflicting evidence | Tests citation and reasoning reliability |

### Evaluation Metrics

The main metrics will be:

| Metric | Definition |
|---|---|
| Answer accuracy | Whether the final answer is factually correct according to the source document |
| Citation accuracy | Whether the cited source passage supports the answer |
| Faithfulness | Whether the answer contains only claims grounded in the retrieved document |
| Hallucination rate | Percentage of answers containing unsupported, invented, or contradictory claims |
| Refusal accuracy | Whether the system correctly refuses when the source does not contain the answer; secondary for HotpotQA because it is mostly answerable |

### Human Evaluation

Each output will be reviewed using a structured rubric:

- Correct: fully supported and accurately cited
- Partially correct: mostly correct but incomplete or weakly supported
- Unsupported: contains claims not present in the source
- Incorrect: contradicts the source
- Citation failure: answer may be correct, but citation does not support it
- Missing refusal: model answers despite insufficient source evidence

For reliability, a subset of examples should be double-annotated by a second reviewer if available. Disagreements can be resolved through adjudication.

## Error Taxonomy

The paper will classify hallucination and citation failures into the following categories:

| Error Type | Meaning |
|---|---|
| Unsupported claim | Answer includes information not found in the provided source |
| Wrong citation | Citation points to a passage that does not support the claim |
| Overgeneralization | Answer stretches the source beyond what it states |
| Contradiction | Answer conflicts with the source document |
| Missing refusal | Model answers when it should say there is not enough information |
| Incomplete grounding | Answer is partly supported but includes extra unsupported detail |

## Expected Results

The study is expected to show that RAG reduces but does not eliminate hallucination. The most important expected finding is that citation presence does not guarantee citation accuracy. Some answers may look reliable because they include source references, while the cited passage only partially supports, weakly supports, or fails to support the generated claim.

Actual numerical results will be reported only after experiments are completed.

## Expected Contributions

This study contributes:

1. A practical evaluation framework for measuring hallucination and citation accuracy using existing QA benchmarks.
2. A human-evaluation rubric for separating answer correctness from citation support.
3. An error taxonomy for common RAG hallucination and citation failures.
4. A reproducible experimental setup that can be shared through GitHub.

Core contribution statement:

This study contributes a practical evaluation framework for measuring hallucination and citation accuracy in RAG-based AI assistants using existing benchmark datasets with gold answers and supporting evidence.

## Limitations

The first version of the study will have several limitations:

- Dataset size may be limited to 25 to 100 benchmark examples in the first version.
- HotpotQA is not an enterprise dataset, so enterprise generalization must be discussed carefully.
- HotpotQA is mostly answerable, so refusal accuracy may require a later extension with unanswerable examples.
- Evaluation may depend partly on human judgment.
- Results may vary across prompts, retrieval settings, chunking strategies, and model versions.
- The benchmark will focus on text-based documents rather than multimodal knowledge bases.

## Ethics and Data Use

The study will use established research benchmarks and their documented licenses/terms rather than scraping random websites or using private internal documents. The released code should reproduce benchmark conversion steps and avoid redistributing data in ways that conflict with benchmark terms.

## Publication Plan

The recommended publication path is:

1. Build the benchmark and code repository.
2. Run initial experiments and write the full paper.
3. Release a preprint on arXiv.
4. Submit to an AI, NLP, information retrieval, or responsible AI workshop.
5. Expand the dataset and experiments for a conference or journal submission.

Possible later venues include ACL, EMNLP, NAACL, ICLR, NeurIPS, AAAI, and responsible AI or NLP workshops. Journal options after strengthening the work may include IEEE Access, Engineering Applications of Artificial Intelligence, Expert Systems with Applications, or related AI reliability venues.

## Initial References

- Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Yang et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." EMNLP 2018. https://arxiv.org/abs/1809.09600
- Thakur et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021. https://arxiv.org/abs/2104.08663
- Liu, Zhang, and Liang. "Evaluating Verifiability in Generative Search Engines." Findings of EMNLP 2023. https://arxiv.org/abs/2304.09848
- Es et al. "Ragas: Automated Evaluation of Retrieval Augmented Generation." arXiv 2023, revised 2025. https://arxiv.org/abs/2309.15217
- Saad-Falcon et al. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." NAACL 2024. https://arxiv.org/abs/2311.09476
- Niu et al. "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models." arXiv 2024. https://arxiv.org/abs/2401.00396

## Next Milestones

1. Prepare a 50-example HotpotQA subset.
2. Run gold-evidence RAG and retrieved-evidence RAG conditions.
3. Label answer correctness, citation correctness, and hallucination.
4. Compare gold-evidence retrieval against keyword or embedding retrieval.
5. Refine the annotation rubric.
6. Scale to 100 examples after the pilot.
