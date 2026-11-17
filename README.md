# Calibrating XAI in RAG Systems: An Empirical Study on User Trust

> **Official Codebase** | Osnabrück University & University of Bremen (virtUOS / Digital Media Lab)

This repository hosts the complete data pipelines, backend architectures, and statistical evaluations for an empirical study on Large Language Model (LLM) conversational agents. The research investigates how underlying retrieval mechanisms—specifically traditional **Semantic Retrieval-Augmented Generation (RAG)** versus **Knowledge Graph-based RAG (GraphRAG)**—impact the explainability and trustworthiness of these systems within a domain-specific academic context.

## 🔬 Research Scope & Methodology

To comprehensively evaluate the chatbots, this project employs a hybrid evaluation methodology:

* **Human-in-the-Loop Evaluation:** A controlled user study conducted via a custom Streamlit web application (*UniCompanion*) to capture subjective ratings on system explainability and user trust.

* **Automated LLM-as-a-Judge:** Leveraged frameworks like DeepEval alongside lexical metrics (BERTScore) to establish baseline RAG performance across the standard benchmark RAGBench (HotpotQA split) and a custom academic dataset.

* **Statistical analyses:** Rigorous hypothesis testing, including Repeated Measures ANOVA, Aligned Rank Transform, and Spearman correlations, to assess evaluator bias and the interplay between transparency and trust.

## 📂 Codebase Architecture

The repository is modularized chronologically to reflect the pipeline of the experiment. Below is an overview of the main folders.

```text
.
├── data
│   ├── pilot_study_data
│   ├── rag_corpora
│   └── user_study_exports
├── rag_pipelines
│   ├── semantic_benchmark_data
│   ├── semantic_academic_data
│   ├── graph_benchmark_data
│   └── graph_academic_data
├── streamlit_app
├── statistical_analysis
│   ├── rq1_automated_evaluations
│   ├── rq2_human_vs_llm_judge
│   └── rq3_trust_analysis
│       └── demographics_analysis
└── results
    ├── automated_evaluation
    └── human_evaluation
```

### 1. `data/` : Core Datasets & Ground Truth

Stores the foundational data used throughout the study.

* `pilot_study_data/`: Exports and link documentation from initial pilot testing.
* `rag_corpora/`: Contains evaluation data split into two subdirectories:
  * `benchmark_eval_data/`: Includes sampled benchmark dataset in structured formats for automated testing.
  * `academic_eval_data/`: Contains structural samples demonstrating the proprietary Osnabrück University dataset schema, human-crafted ground-truth Q&A benchmarks for DeepEval, and data restriction documentation.

* `user_study_exports/`: Contains the encrypted `ExperimentData.zip` securing the raw human-subject study data.

### 2. `rag_pipelines/` : Backend Implementations

Contains the Python scripts for data ingestion, chunking, retrieval, and evaluation across both tested architectures.

* `semantic_benchmark_data/` & `semantic_academic_data/`: Traditional vector-based RAG pipelines.

* `graph_benchmark_data/` & `graph_academic_data/`: Graph database architectures utilizing Neo4j and Cypher query integrations.

### 3. `streamlit_app/` : Frontend Evaluation

Contains the codebase for the **UniCompanion** interactive web application used to deploy rendered responses from four distinct agents and facilitate the human user study.

### 4. `statistical_analysis/` : Analytical Scripts

A collection of R and Python scripts for data cleaning, processing, and visualization.

* `rq1_automated_evaluations/`: Executes automated metric computations to address the first research question.

* `rq2_human_vs_llm_judge/`: Analyzes evaluator bias alignments between human scores and LLM judges to address the second research question.

* `rq3_trust_analysis/`: Executes hypothesis testing and covariate analysis for the trust and explainability dimensions to address the third research question.
    * `demographics_analysis/`: Computes participant age and gender distributions.

### 5. `results/` : Computed Outputs
Stores all final statistical summaries and visual artifacts (including tables and plots) produced by the analysis scripts, organized into `automated_evaluation/` and `human_evaluation/`subdirectories.

## 🚀 Reproducibility & Setup Guide

### 1. Data Decryption & Environment Preparation

To comply with GDPR and human-subject research protocols, raw demographic and evaluation datasets are AES-encrypted within `data/user_study_exports/ExperimentData.zip`.

* **Reviewer Access**: Decrypt the archive using the key provided via secure email. Extract the .csv payload directly into its host directory to ensure all relative file paths resolve correctly during downstream script execution.

### 2. Backend Execution

Navigate to the `rag_pipelines/` directory to review or execute the ingestion and retrieval code for both the semantic and graph-based approaches. Ensure your local environment supports the necessary Graph Database instances (Neo4j).

### 3. Running the Analysis

Navigate to `statistical_analysis/` to review the data cleaning and evaluation logic. Scripts should be executed sequentially (RQ1, RQ2, and concluding with RQ3, where demographics are also analyzed) to successfully generate the figures found in `results/`.

## 📬 Contact

If you have questions regarding this research, the methodology, or the repository structure, please feel free to open an issue or reach out directly at mamini@uni-osnabrueck.de.

Thank you for your interest in this research!