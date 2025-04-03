# Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.
## table of contents
* overview structure
* set up
* example usage
* performance
* latancy quality trade off
# Overview

The system is structured into several major components:

## Working Baseline

This module is based on an abstract class called "base_model," which allows for creating working indexes using OpenAI embedding models. It also includes a chunking method that divides files using a sliding window or truncation when exceeding token limits. The design can be extended to accommodate different (including local) models by inheriting from the abstract class.

## Reranker

Two types of rerankers are implemented:
1. An LLM-based reranker that utilizes OpenAI GPT models.  
2. A cross-encoder-based reranker using a local model (cross-encoder/ms-marco-MiniLM-L-12-v2).

## Query Expansion

Two approaches are provided for query expansion:
1. LLM-based expansion (OpenAI GPT) to generate additional keywords for improved searching.  
2. Dictionary-based expansion to find and insert synonyms for existing keywords.

## Annotations

Optionally, you can append various metadata to each chunk, including file extensions, absolute paths, relative paths for hierarchical insights, as well as function and class definitions.

## Metric

A customizable recall metric is included for evaluating the indexing quality. You can also integrate additional metrics as needed.

## Additional Features

• Track overall token consumption and processing time.  
• View a summary of retrieved files for convenient review.  
• Use the user-friendly CLI for an intuitive model interaction.

## Motivation

This OOP-oriented design facilitates scalability, enabling the addition of new classes or models that can seamlessly integrate with the existing system.
  
