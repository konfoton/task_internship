from base_model import BaseModel
from metric import Recall
import os
import pickle
import torch
from torch import tensor
import faiss
import git
import numpy as np
import time as timee
import ast
from meta_data_provider import MetaDataProvider
import summary as summarizer
class OpenAIModel(BaseModel):
    def __init__(self, link_repo, model, encoding, limit, length_treshold, client):
        super().__init__(link_repo, model)
        self.encoding = encoding
        self.limit = limit
        self.length_treshold = length_treshold
        self.client = client

    def get_embedding(self, text):
        response = self.client.embeddings.create(input = text, model=self.model)
        embeddings = [tensor(item.embedding) for item in response.data]
        return torch.stack(embeddings, dim=0)

    def clone_repo(self):
        cwd = os.getcwd()
        repo_name = self.link_repo.split('/')[-1].replace('.git', '')
        self.repo_path = repo_name
        clone_path = os.path.join(cwd, repo_name)
        if os.path.exists(clone_path):
            print(f"Repository already cloned at {clone_path}")
            return
        git.Repo.clone_from(self.link_repo, clone_path)

    @staticmethod
    def split_into_chunks(text, chunk_size, overlap_size):
            chunks = []
            for i in range(0, len(text), chunk_size - overlap_size):
                if i + chunk_size <= len(text):
                    chunks.append(text[i:i + chunk_size])
                else:
                    chunks.append(text[i:])
            return chunks
    
    def build_cost(self, mutiple_chunks_per_file = False, chunk_size = 1200, overlap_size = 200, metadata = None):
        """
        Approximately calculates the cost of the embedding for the files in the repo
        """
        if self.model == "text-embedding-3-small":
            cost = 0.02 / 1000000
        elif self.model == "text-embedding-3-large":
            cost = 0.13 / 1000000
        elif self.model == "text-embedding-ada-002":
            cost = 0.10 / 1000000

        final_cost = 0
    
        if not mutiple_chunks_per_file:
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            c = f.read()
                            d = self.encoding.encode(c)
                            if (len(d) > self.limit):
                                d = d[:self.limit]
                            if len(c) <= self.length_treshold:
                                continue
                            final_cost += cost * len(d)
                    except UnicodeDecodeError:
                        continue
        else:
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            c = f.read()
                            d = self.encoding.encode(c)
                            if metadata is not None:
                                metadata_text = metadata.get_metadata(file_path)
                                meta_data_tokens = self.encoding.encode(metadata_text)
                            if len(d) <= self.length_treshold:
                                continue
                            chunks = OpenAIModel.split_into_chunks(d, chunk_size, overlap_size)
                            for chunk in chunks:
                                final_cost += cost * len(chunk)
                                if metadata is not None:
                                    final_cost += cost * len(meta_data_tokens)
                    except UnicodeDecodeError:
                        continue

        return final_cost

    def create_index(self, index_name, file_store_name, mutiple_chunks_per_file = False, chunk_size = 1200, overlap_size = 200, metadata = None):
        """
        Quick description: creates an index for the files in the repo 
        """
        paths = []
        file_contents = []
        if not mutiple_chunks_per_file:
            #trunacting the file to the limit
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            c = f.read()
                            d = self.encoding.encode(c)
                            if (len(d) > self.limit):
                                d = d[:self.limit]
                                c = self.encoding.decode(d)
                            if len(c) <= self.length_treshold:
                                continue
                            paths.append(relative_path)
                            file_contents.append(c)
                    except UnicodeDecodeError:
                        continue
        else:
            # handling multiple chunks per file
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            c = f.read()
                            d = self.encoding.encode(c)
                            if metadata is not None:
                                metadata_text = metadata.get_metadata(file_path)
                            if len(d) <= self.length_treshold:
                                continue
                            chunks = OpenAIModel.split_into_chunks(d, chunk_size, overlap_size)
                            for chunk in chunks:
                                paths.append(relative_path)
                                if metadata is not None:
                                    file_contents.append(metadata_text + self.encoding.decode(chunk))
                                else:
                                    file_contents.append(self.encoding.decode(chunk))
                    except UnicodeDecodeError:
                        continue

        embeddings = self.get_embedding(file_contents)
        embeddings = embeddings.cpu().numpy() 

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        faiss.write_index(index, index_name)


        with open(file_store_name, 'wb') as f:
            pickle.dump(paths, f)

        self.index_name = index_name
        self.file_store_name = file_store_name
            
    def search(self, queries, top_k=1, upper_bound_search = 50, rerank=None, number_to_rerank = None, query_expansion=None, summary = False, summary_model_name = "gpt-4o", token_usage = False, time = False): 
        """
        Quick description: searches for the top_k most relevant files for each query in the index

        Args: 
        queries: list of queries
        top_k: number of files to return
        upper_bound_search: number of files to search in the index
        number_to_rerank: number of files to rerank
        rerank: reranker object
        query_expansion: query expansion object
        summary: if True, return the summary of the files
        token_usage: if True, return the token usage of the queries
        time: if True, return the time taken to search

        Returns:

        [list of list of retrieved files per query, list of summaries, list token usages, time]

        if some are not specified they are asigned to False

        """


        if time:
            time_elapsed = timee.time()
        else:
            time_elapsed = False

        with open(self.file_store_name, 'rb') as f:
            file_paths =  pickle.load(f)
        index = faiss.read_index(self.index_name)

        top_paths_all_queries = []
        values_top_all_queries = []

        if summary:
            summary_list = []
        else:
            summary_list = False

        if token_usage:
            token_usage_list = [len(self.encoding.encode(query)) for query in queries]
        else:
            token_usage_list = False

        
        current_scope = top_k

        # adding how many documents to rerank
        if number_to_rerank is not None:
            current_scope = number_to_rerank


        query_embeddings = self.get_embedding(queries)
  
        # building the index
        D, I = index.search(query_embeddings, upper_bound_search)
        for i in range(len(I)):

            distinct_results = []
            distinct_values = []
            seen_ids = set()
            for idx1, idx in enumerate(I[i]):
                if file_paths[idx] not in seen_ids:
                    seen_ids.add(file_paths[idx])
                    distinct_results.append(file_paths[idx])
                    distinct_values.append(D[i][idx1])
                if len(distinct_results) == current_scope:
                    break
            top_paths_all_queries.append(distinct_results)
            values_top_all_queries.append(distinct_values)


        # query expansion
        # claculating first number of current scope documents and its vaues
        # then i qury expansion and then search again
        # finding doucments which smaller dinstanes and are not included in the first search
        if query_expansion is not None:
            queries_expansion = []
            for idx, query in enumerate(queries):
                results = query_expansion.expand_query(query)
                queries_expansion.append(results[0])
                if token_usage:
                    token_usage_list[idx] += results[1]
            for index_query, query in enumerate(queries_expansion):
                for alternative_query in query:
                    query_embedding_add = self.get_embedding([alternative_query])
                    values, search = index.search(query_embedding_add, upper_bound_search)
                    for i in range(len(search[0])):
                        if file_paths[search[0][i]] not in top_paths_all_queries[index_query]:
                            for j in range(current_scope):
                                if values_top_all_queries[index_query][j] > values[0][i]:
                                    top_paths_all_queries[index_query].insert(j, file_paths[search[0][i]])
                                    top_paths_all_queries[index_query].pop()
                                    values_top_all_queries[index_query].insert(j, values[0][i])
                                    values_top_all_queries[index_query].pop()
                                    break


        # reranking documents
        if rerank is not None:
            for index_query, query in enumerate(queries):
                result = rerank.rerank(query, top_paths_all_queries[index_query])
                if(len(result[0].ranked) < current_scope):
                    top_paths_all_queries[index_query] =  top_paths_all_queries[index_query][:top_k]
                else:
                     top_paths_all_queries[index_query] = [top_paths_all_queries[index_query][result[0].ranked[i]] for i in range(top_k)]  
                if token_usage:
                    token_usage_list[index_query] += result[1]
                if hasattr(rerank, 'rate_limit'):
                    if index_query + 1 != len(queries):
                        timee.sleep(90)

        if summary:
            for index_query in range(len(top_paths_all_queries)):
                result = summarizer.summarize_file(os.path.join(self.repo_path, top_paths_all_queries[index_query][0]), model_name=summary_model_name)
                summary_list.append(result[0])
                if token_usage:
                    token_usage_list[index_query] += (result[1]) 

        if time:
            end_time = timee.time()
            time_elapsed = end_time - time_elapsed



        return [top_paths_all_queries, summary_list, token_usage_list, time_elapsed]

    def evaluate(self, top_k=10, upper_bound_search = 50, rerank=None, query_expansion=None, number_to_rerank = None, metric=None, token_usage = False, time = False):
        predicted_files_batch = self.search(metric.queries, top_k=top_k, upper_bound_search = 50, rerank=rerank, number_to_rerank=number_to_rerank, query_expansion=query_expansion, token_usage = token_usage, time = time)
        metric_result = metric.evaluate(predicted_files_batch[0])
        return [predicted_files_batch, metric_result]