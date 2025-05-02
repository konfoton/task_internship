# Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

<br>
<br>

# Table of contents
* [Overview](#Overview)
* [Setup](#Setup)
* [Example usage](#Example-Usage)
* [Performance evaluation](#Performance-evaluation)
* [Latancy-quality-token trade-off](#Latancy-quality-token-trade-off)

<br>
<br>

# Overview
In principle my model is able to search files from any git hub repository based on query. It is done by converting each file into embedding vector, then encoding query and finally choosing the closest vector in appropriate metric such as (Manhattan metric, cosine similarity and Lp norms). Sometimes files are to big and then we haev toconvert them into chunks. All vectors are stored within optimized vector database which uses clustering algorithms for fast search. To enhance efficiency I added metadata to each chunk, reranker and query expasion.
The system is structured into several major components:


### Working Baseline

This module is based on an abstract class called "base_model," which allows for creating working indexes using OpenAI embedding models. It also includes a chunking method that divides files using a sliding window or truncation when exceeding token limits. The design can be extended to accommodate different (including local) models by inheriting from the abstract class.

---

### Reranker

Two types of rerankers are implemented:
1. An LLM-based reranker that utilizes OpenAI GPT models.  
2. A cross-encoder-based reranker using a local model (cross-encoder/ms-marco-MiniLM-L-12-v2).

---

### Query Expansion

Two approaches are provided for query expansion:
1. LLM-based expansion (OpenAI GPT) to generate additional keywords for improved searching.  
2. Dictionary-based expansion to find and insert synonyms for existing keywords.

---

### Annotations

Optionally, you can append various metadata to each chunk, including file extensions, absolute paths, relative paths for hierarchical insights, as well as function and class definitions (So far it works only for .js and .vue but it can be easily extended).

---

### Metric

A customizable recall metric is included for evaluating the indexing quality. You can also integrate additional metrics as needed.

---

### Additional Features

• Track overall token consumption and processing time.  
• View a summary of retrieved files for convenient review.  
• Use the user-friendly CLI for an intuitive model interaction.

---
### Motivation

This OOP-oriented design facilitates scalability, enabling the addition of new classes or models that can seamlessly integrate with the existing system.

<br>
<br>

# Setup
### Prerequisites

* Python 3.13.0
* conda 24.9.2

---
### Creating enviroment
```
git clone https://github.com/konfoton/task_internship
cd task_internship
conda create -n rag python=3.13
conda activate rag
pip install -r requirements.txt
export OMP_NUM_THREADS=1
export OPENAI_API_KEY=<YOUR-API-KEY>
```

---

### Using user_interface
In order to use my model in user-friendly format run:
```
python user_interface.py
```

---

### Evalution
In order to run my best performing model (also in terms of quality-latency-token trade-off) run:
```
python evalutation.py
```

<br>
<br>


# Example usage
Example demonstarte how to create simple index and then ask query to a system  
Note: Remember to first run user_intrface

```
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Welcome to the Example RAG System! (Type HELP for a list of commands)                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

Enter command: help

Available Commands:
  cancel - cancel any operation and go back to the main menu
  price - approximate price of the building index
  index - Index a GitHub repo
  query - Start querying the data
  evaluate  - show evaluation of the model
  exit  - Quit the system
  clear - Clear the screen
  help  - Show available commands

Enter command: index
Enter GitHub repo URL: https://github.com/viarotel-org/escrcpy
Enter model number
 1. text-embedding-3-small 
 2. text-embedding-ada-002 
 3. text-embedding-3-large
 insert number here: 3
Enter chunk size: 2000
Enter overlapp size: 500
Do you want to use metadata? (yes/no): yes
Creating...
Indexing complete!

Enter command: query
Enter your query: What parameters are available for configuring video preferences in the codebase?
Do you want to use a reranker? (yes/no): yes
Enter reranker type (number)
 1. LLM 
 2. CrossEncoder 
 insert number here: 2
Enter number to rerank (number how many files rereanker will take into account (reccomended 15)): 10
Do you want to use query expansion? (yes/no): no
Enter number of top k results: 5
Do you want to display summary? (yes/no): yes
Do you want to display token usage? (yes/no): yes
Do you want to diplay time? (yes/no): yes
Searching...
Searchring finished...

top 5 results
docs/en/guide/preferences.md
src/store/preference/model/video/index.js
docs/en/reference/scrcpy/video.md
src/components/PreferenceForm/components/SelectVideoCodec/index.vue
src/locales/languages/zh-TW.json

Summary:
The "Preferences" document outlines configurable settings for a software application, divided into 
categories such as General, Video Control, Device Control, Window Control, Audio Control, Audio/Video 
Recording, Input Control, and Camera Control. Each category includes specific options like theme 
selection, language settings, video bitrate, display orientation, and audio codec. Users can adjust 
settings for device connectivity, video and audio forwarding, and control mechanisms like mouse and 
keyboard modes. The document emphasizes continuous improvement and support for common configurations 
to enhance user experience.

Token Usage: 31865

Time: 13.377908945083618
```

<br>
<br>

# Performance Evaluation

My evaluation is based on the [commit data](escrcpy-commits-generated.json) and the [escrcpy GitHub repository](https://github.com/viarotel-org/escrcpy).  
Please note that the repository may have changed since the time of evaluation, so results could differ if re-run.

All experimental results are documented in [file](experiments.txt), and the corresponding execution scripts are provided in [script](experiments.py).

---

### Best Achieved Performance

The highest performance my model achieved was:

**RECALL@10: 0.8422**

With the following configuration:
- **Embedding Model:** `text-embedding-3-large`
- **Chunk Size:** 2000
- **Overlap Size:** 500
- **Query Expansion:** `LLMQueryExpander with gpt-4o-mini` 
- **Reranker:** `LLMReranker with gpt-4o-mini`
- **Metadata Provider:** True

---

### Baseline Experimentation

To motivate this architecture, I first conducted experiments focusing solely on firrent openai models, metadata usage, as well as different chunking strategies and sliding window sizes.

The best result from this initial setup was:

**RECALL@10: 0.8348**

Configuration:
- **Embedding Model:** `text-embedding-3-large`
- **Chunk Size:** 2000
- **Overlap Size:** 500
- **Metadata Provider:** True

This configuration also performed particularly well on a broader retrieval metric:

**RECALL@20: 0.8806**

This indicated that adding a reranker had strong potential to significantly improve top-10 retrieval performance, which was confirmed in later experiments.

# Latancy-quality-token trade-off

It is important to note that execution time for both single and multiple queries can vary—not only due to OpenAI API rate limits, but also due to the latency introduced by loading the vector store. In practice, this latency can reach approximately **1 second**, even before any model computation begins.

---

### Best Model — High Recall, High Latency

While my best-performing model achieved excellent retrieval metrics (RECALL@10: 0.8422), it comes with significant computational and practical overhead during parallel query processing:

- **Average Execution Time:** ~170.72 seconds
- **Average Token Usage (for multiple parallel queries):** ~22,786.79 tokens

This level of resource usage makes the model **impractical under typical API rate limits**, which frequently cause errors when handling many concurrent queries. However, with access to a higher budget or elevated rate limits, this model could achieve an **approximate average execution time of 11 seconds**, including both retrieval and summary generation.

---

### Efficient Model — Lower Recall, High Practicality

On the other hand, a simplified model that only uses metadata (with no query expansion or reranker) demonstrated far better practical efficiency while still performing well:

- **RECALL@10:** 0.8348
- **Average Latency (single query):** ~1.04 second  
- **Average Execution Time with Summary:** ~4 seconds
- **Average Token Usage:** ~(mostly dependent on the query length so really small + 200 with summary)

This model achieved an impressive average query performance of **0.0443 seconds** when running multiple queries in parallel (excluding summary generation), which makes it highly suitable for real-time or resource-constrained applications.











