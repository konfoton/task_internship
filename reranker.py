from pydantic import BaseModel
from openai import OpenAI
import tiktoken
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import numpy as np
from typing import List, Tuple, Dict
import time

class RerankerTemplate(BaseModel):
    ranked: list[int]

class LLMReranker:
    def __init__(self, model_name="gpt-4o", limit=8191, rate_limit=29000, shortening_token=500):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI()
        self.model_name = model_name
        self.limit = limit
        self.rate_limit = rate_limit
        self.shortening_token = shortening_token

    def rerank(self, query, list_of_candidates_paths):
        """
        LLM reranker based on list_of_candidates_paths relvent to the query.
        Returns [ranked_indices, total_tokens]
        """

        consecutive_sum = []
        messages = []
        # Read files and create message content
        for index, path in enumerate(list_of_candidates_paths):
            try:
                with open(f"escrcpy/{path}", "r", encoding="utf-8") as f:
                    content = f.read()
                    d = self.enc.encode(content)
                    length = len(d)
                    if length > self.limit:
                        d = d[:self.limit]
                        consecutive_sum.append(self.limit)
                        content = self.enc.decode(d)
                    else:
                        consecutive_sum.append(length)
                    messages.append(f"Index: {index}\nContent:\n[{content}]\n")
            except FileNotFoundError:
                print(f"Warning: File {path} not found.")
            except Exception as e:
                print(f"Error reading {path}: {e}")

        while sum(consecutive_sum) > self.rate_limit:
            temp = consecutive_sum.index(max(consecutive_sum))
            consecutive_sum[temp] -= self.shortening_token
            d = self.enc.encode(messages[temp])
            d = d[:len(d) - self.shortening_token]
            messages[temp] = self.enc.decode(d)

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": 
                 "given a query and a list of code snippets, rerank the code snippets based on "
                 "the query and return the reranked list indices from the most relevant to "
                 "the least relevant"},
                {"role": "user", "content": f"query: {query} {" ".join(messages)}"},
            ],
            response_format=RerankerTemplate
        )

        parsed_response = completion.choices[0].message.parsed

        return [parsed_response, completion.usage.total_tokens]
    
class CrossEncoderReranker:
    """
    Cross-encoder reranker based on list_of_candidates_paths relvent to the query.
    returuns [ranked_indices, total_tokens]
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", 
                 chunk_size=512, 
                 overlap=100):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Size of each document chunk we remove 40 tokens for qurey in order not to remove much text
        self.chunk_size = chunk_size - 40 
        # Overlap between consecutive chunks
        self.overlap = overlap 
        
    def chunk_document(self, content: str) -> List[str]:
        tokens = self.enc.encode(content)
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            if i + self.chunk_size <= len(tokens):
                chunk_tokens = tokens[i:i + self.chunk_size]
                if len(chunk_tokens) < 50:  # Skip very small chunks
                    continue
            else:
                if len(chunk_tokens) < 50:  # Skip very small chunks
                    continue
                chunk_tokens = tokens[i:]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
         
    def rerank(self, query, list_of_candidates_paths):
        # stroring chunk (index, content) wehre same index denote same file
        all_chunks = []
        valid_indices = []
        total_tokens = 0
        
        # Process each file
        for index, path in enumerate(list_of_candidates_paths):
            try:
                with open(f"escrcpy/{path}", "r", encoding="utf-8") as f:
                    content = f.read()
                    tokens = self.enc.encode(content)
                    token_count = len(tokens)
                if token_count == 0:
                    continue
            except Exception as e:
                print(f"Error reading {path}: {e}")

            
            # For very large files, break into chunks
            if token_count > self.chunk_size:
                chunks = self.chunk_document(content)
                for chunk in chunks:
                    all_chunks.append((index, chunk))
                    total_tokens += len(self.enc.encode(chunk))
            else:
                all_chunks.append((index, content))
                total_tokens += token_count
                
            valid_indices.append(index)
            
        # Calculate scores for all chunks
        chunk_scores = {}  # {file_idx: [scores]}
        
        # Process chunks in batches
        for i in range(0, len(all_chunks)):
            
            # Prepare inputs for cross-encoder
            features = self.tokenizer(query, all_chunks[i][1], 
                                     padding=True, truncation=True,
                                     max_length=self.chunk_size, return_tensors="pt")
            
            # Get relevance scores
            with torch.no_grad():
                batch_score = self.model(**features).logits.flatten().cpu().numpy()
            
            # Store scores by file index
            index_of_chunk = all_chunks[i][0]
            if index_of_chunk not in chunk_scores:
                chunk_scores[index_of_chunk] = []
            chunk_scores[index_of_chunk].append(batch_score)
    
        # Aggregate chunk scores for each file (using max score as the file's relevance)
        file_scores = {}
        for idx, scores in chunk_scores.items():
            # Use the max score as the file's score (could also use mean, or weighted by chunk position)
            file_scores[idx] = max(scores)  # Alternative: np.mean(scores)
        
        # Sort files by their scores
        ranked_indices = [idx for idx, _ in sorted(file_scores.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)]
        
        
        parsed_response = RerankerTemplate(
            ranked=ranked_indices,
        )
        return [parsed_response, total_tokens]