# Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

<br>
<br>

# Table of contents
* Overview
* Setup
* Example usage
* Performance
* Latancy quality trade-off

<br>
<br>

# Overview

The system is structured into several major components:

### Working Baseline

This module is based on an abstract class called "base_model," which allows for creating working indexes using OpenAI embedding models. It also includes a chunking method that divides files using a sliding window or truncation when exceeding token limits. The design can be extended to accommodate different (including local) models by inheriting from the abstract class.

### Reranker

Two types of rerankers are implemented:
1. An LLM-based reranker that utilizes OpenAI GPT models.  
2. A cross-encoder-based reranker using a local model (cross-encoder/ms-marco-MiniLM-L-12-v2).

### Query Expansion

Two approaches are provided for query expansion:
1. LLM-based expansion (OpenAI GPT) to generate additional keywords for improved searching.  
2. Dictionary-based expansion to find and insert synonyms for existing keywords.

### Annotations

Optionally, you can append various metadata to each chunk, including file extensions, absolute paths, relative paths for hierarchical insights, as well as function and class definitions (So far it works only for .js and .vue but it can be easily extended).

### Metric

A customizable recall metric is included for evaluating the indexing quality. You can also integrate additional metrics as needed.

### Additional Features

• Track overall token consumption and processing time.  
• View a summary of retrieved files for convenient review.  
• Use the user-friendly CLI for an intuitive model interaction.

### Motivation

This OOP-oriented design facilitates scalability, enabling the addition of new classes or models that can seamlessly integrate with the existing system.

<br>
<br>

# Setup
### Preriquisitives

* Python 3.13.0
* conda 24.9.2

### Creating enviroment
```
git clone https://github.com/konfoton/task_internship
cd task_internship
conda create -n rag python=3.13
conda activate rag
pip install -r requirements.txt
export OMP_NUM_THREADS=1
export OPENAI_API_KEY='YOUR-API-KEY'
```
### Using user_interface
In order to use my model in user-friendly format run:
```
python user_interface.py
```
### Evalution
In order to run my best performing model (also in terms of quality-latency trade-off) run:
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
