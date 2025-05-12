[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README.zh-CN.md)

## Overview

`APIUtils` is a **Python utility package** for API recommendation research and development, designed to provide efficient and easy-to-use API tools that support quick integration and usage of various API services.
Please refer to module documentation for details.

---

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [API Entity Parsing and Normalization](#api-entity-parsing-and-normalization)
  - [Sentence Encoding and Semantic Matching](#sentence-encoding-and-semantic-matching)
  - [LLM Service Calling](#llm-service-calling)
  - [Dataset Operations](#dataset-operations)
  - [Evaluation Metrics Calculation](#evaluation-metrics-calculation)
  - [Drawing Evaluation Charts](#drawing-evaluation-charts)
- [Module Description](#module-description)
  - [Calculator](#calculator)
  - [dataset](#dataset)
  - [API](#api)
  - [LLMService](#llmservice)
  - [SentenceEncoder](#sentenceencoder)
  - [chart](#chart)
- [Configuration and Dependencies](#configuration-and-dependencies)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

---

## Installation

```bash
pip install apiutils
```

We recommend using [uv](https://www.datacamp.com/tutorial/python-uv) tool for installation:

```bash
uv add apiutils
```

To install the latest version:

```bash
uv add git+https://github.com/WhiteRain2/APIUtils.git
```

---

## Quick Start

### API Entity Parsing and Normalization

```python
from apiutils import API

# Parse API from string
api = API("java.util.List.add(Object)")
print(api.fullname)  # java.util.List.add
print(api.method)    # add
print(api.prefix)    # java.util.List
print(api.args)      # ['Object']

# Parse multiple APIs from text
apis = API.from_string("Use java.util.List.add to add elements, then sort with java.util.Collections.sort")
for api in apis:
    print(f"Found API: {api}")

# Check if it's a standard API
if api.is_standard:
    print("This is a standard API")
else:
    # Get possible standard APIs
    possible_apis = api.get_possible_standard_apis()
    print(f"Possible standard APIs: {possible_apis}")
```

### Sentence Encoding and Semantic Matching

```python
from apiutils import SentenceEncoder

# Initialize encoder
encoder = SentenceEncoder('all-MiniLM-L6-v2')

# Encode queries
queries_dict = {
    1: "How to add elements to a list?",
    2: "How to sort a collection?",
    3: "Converting string to integer in Java"
}
encoder.encode_queries(queries_dict)

# Find semantically similar queries
results = encoder.find_similar_queries(
    ["How can I put items into a list?"],
    top_k=2
)
print(results)  # [[(query_id, similarity_score), ...]]

# Save and load embeddings
encoder.save_embeddings("embeddings.pkl")
encoder.load_embeddings("embeddings.pkl")
```

### LLM Service Calling

```python
import asyncio
from apiutils import LLMService

# Set global client configuration
LLMService.set_llm_client_config(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# Create service instance
service = LLMService(
    model="gpt-4o-mini",
    system_prompt="You are a Java API expert",
    configs={"temperature": 0.7}
)

# Asynchronous dialogue
async def chat_example():
    async for chunk in service.chat("How to implement thread-safe collections in Java?"):
        print(chunk, end="")

# Process multiple queries in batch
async def batch_example():
    questions = [
        "How to implement a singleton pattern in Java?",
        "How to use Java Stream API?",
        "How to handle NullPointerException?"
    ]
    responses = await service.queries(questions, batch_size=2)
    for res in responses:
        print(f"Question: {res.query}\nAnswer: {res.answer}\nTokens: {res.tokens}\n")

# Run asynchronous examples
asyncio.run(chat_example())
asyncio.run(batch_example())
```

### Dataset Operations

```python
from apiutils.dataset import Dataset, DatasetName

# Load predefined datasets
biker_dataset = Dataset(DatasetName.BIKER, 'test', 'filtered', nrows=100)
apibench_dataset = Dataset(DatasetName.APIBENCH_Q, 'train', nrows=50)

# Access data
for idx, row in enumerate(biker_dataset):
    print(f"Question {idx}: {row.title}")
    print(f"API answer: {row.answer}")
    if idx >= 2:
        break

# Create custom dataset from DataFrame
import pandas as pd
df = pd.DataFrame({
    'title': ["How to convert string to integer?", "How to sort a list?"],
    'answer': ["Integer.parseInt()", "Collections.sort()"]
})
custom_dataset = Dataset.from_dataframe("Custom Dataset", df)
```

### Evaluation Metrics Calculation

```python
from apiutils import Calculator

# Prepare candidate sequences and reference answers
candidate_lists = [
    ["java.util.List.add", "java.util.ArrayList.size", "java.util.Collections.sort"],
    ["java.lang.Integer.parseInt", "java.lang.Long.parseLong"]
]
reference_lists = [
    ["java.util.List.add", "java.util.Collections.sort"],
    ["java.lang.Integer.parseInt"]
]

# Calculate evaluation metrics
calculator = Calculator(candidate_lists, reference_lists)
print(f"MRR: {calculator.mrr:.4f}")
print(f"BLEU: {calculator.bleu:.4f}")
print(f"MAP: {calculator.map:.4f}")

# Calculate metrics for multiple k values
k_values = [1, 3, 5]
metrics = calculator.calculate_metrics_for_multiple_k(k_values)
print(f"Success@{k_values}: {metrics.successrate_at_ks}")
print(f"Precision@{k_values}: {metrics.precision_at_ks}")
print(f"Recall@{k_values}: {metrics.recall_at_ks}")
print(f"NDCG@{k_values}: {metrics.ndcg_at_ks}")
```

### Drawing Evaluation Charts

```python
from apiutils.chart import draw_liner
import numpy as np
import pathlib

# Prepare data
x_values = [1, 3, 5, 10]
y_dict = {
    "Model A": [0.75, 0.82, 0.88, 0.92],
    "Model B": [0.68, 0.78, 0.85, 0.90]
}

# Draw line chart
draw_liner(
    x_values=x_values,
    label_ys_dict=y_dict,
    x_label="K values",
    y_label="Precision@K",
    save_path="precision_comparison.png",
    title="Model Performance Comparison"
)
```

---

## Module Description

### Calculator

The Calculator module provides comprehensive evaluation metrics for information retrieval and generative models:

- MRR (Mean Reciprocal Rank): Evaluates the quality of the first correct answer's ranking
- BLEU: Bilingual Evaluation Understudy, evaluates similarity between generated sequences and reference sequences
- MAP: Mean Average Precision, comprehensive evaluation of retrieval result quality
- Success@k: Proportion of having at least one correct answer in the top k items
- Precision@k: Proportion of correct answers in the top k items
- Recall@k: Proportion of correct answers successfully retrieved in the top k items
- NDCG@k: Normalized Discounted Cumulative Gain, evaluation metric considering ranking position

It enables comprehensive evaluation of API recommendation model outputs against standard answers, suitable for API recommendation, code completion, and similar tasks.

### dataset

The dataset module provides access to commonly used datasets in the API recommendation domain:

- BIKER: A large question-answer dataset for API recommendation, including training and test sets
- APIBENCH-Q: API benchmark evaluation dataset, suitable for evaluating models on API query tasks

It supports accessing data by index or slice, converting to API entity objects, and creating datasets from custom DataFrames.

### API

The API module provides parsing and normalization functions for Java API strings:

- Parse API full name, method name, prefix, and parameters from strings
- Check if an API is a standard API
- Get a list of possible standard API matches
- Support for custom standard API libraries

It is suitable for API identification, standardization, and matching tasks.

### LLMService

The LLMService module is an advanced wrapper for the OpenAI API, providing:

- Asynchronous streaming dialogue (with multi-turn conversation management)
- Batch concurrent query processing (with rate limiting, timeout, and retry mechanisms)
- Session history management
- Token usage statistics

Optimized for production environments, it provides robust access to large language models.

### SentenceEncoder

The SentenceEncoder module uses the SentenceTransformer library to implement sentence semantic encoding and matching:

- Text vectorization encoding and decoding
- Batch query encoding and saving
- Similarity calculation and semantic matching
- Support for CPU and GPU(CUDA) acceleration (requires installing compatible CUDA and PyTorch)

It is suitable for semantic retrieval, question matching, document similarity analysis, and related tasks.

### chart

The chart module provides easy-to-use chart drawing functions:

- Multi-series line chart drawing
- Support for custom labels, fonts, and styles
- Convenient file saving interface

Ideal for quickly visualizing evaluation results and experimental data.

---

## Configuration and Dependencies

- Python ≥ 3.8;
- `openai`, `tqdm`, `sentence-transformers`, `scikit-learn`, `nltk`, etc.—see the `dependencies` list in `pyproject.toml`.
- We recommend using the `uv` tool for a consistent experience.

---

## Contribution Guidelines

**Issues** and **Pull Requests** are welcome.

1. Fork this repository and create a new branch;
2. Write code following [PEP8](https://peps.python.org/pep-0008/) and [Google Style](https://google.github.io/styleguide/pyguide.html);
3. Participate in test coverage and documentation supplements;
4. Submit PR and describe the background of changes.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. Feel free to use and contribute.
