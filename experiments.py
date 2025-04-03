from openai_model import OpenAIModel
from metric import Recall
from openai import OpenAI
import tiktoken
import os
from query_expansion import LLMQueryExpander
from query_expansion import DictionaryQueryExpander
from reranker import LLMReranker
from reranker import CrossEncoderReranker
from meta_data_provider import MetaDataProvider

# Constants
client = OpenAI()
link_repo = "https://github.com/viarotel-org/escrcpy"
limit = 8191
rate_limit = 29000
length_threshold = 5
enc = tiktoken.get_encoding("cl100k_base")
models = ["text-embedding-3-small", "text-embedding-ada-002", "text-embedding-3-large"]

# Initialize objects
MetaDataProvider_instance = MetaDataProvider()
LLMQueryExpander_instance = LLMQueryExpander(model="gpt-4o")
DictionaryQueryExpander_instance = DictionaryQueryExpander(1)
LLMReranker_instance = LLMReranker(model_name="gpt-4o", limit=limit, rate_limit=rate_limit, shortening_token=500)
CrossEncoderReranker_instance = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
recall10 = Recall(10, data_path='escrcpy-commits-generated.json')
recall15 = Recall(20, data_path='escrcpy-commits-generated.json')
recall30 = Recall(30, data_path='escrcpy-commits-generated.json')
recall40 = Recall(40, data_path='escrcpy-commits-generated.json')
recall50 = Recall(50, data_path='escrcpy-commits-generated.json')

# Initialize model objects
object1 = OpenAIModel(link_repo, models[0], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object1.repo_path = "escrcpy"
object2 = OpenAIModel(link_repo, models[1], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object2.repo_path = "escrcpy"
object3 = OpenAIModel(link_repo, models[2], encoding=enc, limit=limit, length_treshold=length_threshold, client=client)
object3.repo_path = "escrcpy"

def single_experiment(f, model_instance, chunk_size, overlap_size,  experiment_number, upper_bound_search = 50, metadata_provider=None, query_expansion=None, reranker=None, name_reranker = None, name_query_expansion = None):
    model_instance.create_index(
        index_name=f"index_experiment_{experiment_number}",
        file_store_name=f"file_experiment_{experiment_number}",
        mutiple_chunks_per_file=True,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        metadata=metadata_provider
    )
    
    f.write(f"Model: {model_instance.model}\n")
    f.write(f"Chunk size: {chunk_size}\n")
    f.write(f"Overlap size: {overlap_size}\n")
    if name_query_expansion is not None:
        f.write(f"Query expansion: {name_query_expansion}\n")
    else :
        f.write(f"Query expansion: False\n")
    if name_reranker:
        f.write(f"Reranker: {name_reranker}\n")
    else:
        f.write(f"Reranker: False\n")
    f.write(f"Metadata provider: {metadata_provider is not None}\n")

    if reranker is not None:
        result_10 = model_instance.evaluate(top_k=10, metric=recall10, query_expansion=query_expansion,  number_to_rerank=15, rerank=reranker, token_usage=True, time=True)
        f.write(f"\nRECALL@10: {result_10[1]}\n")
        f.write(f"Time: {result_10[0][3]}\n")
        f.write(f"Average Time: {result_10[0][3] / len(result_10[0][2])}\n")
        f.write(f"Token Usage: {sum(result_10[0][2])}\n")
        f.write(f"Average Token Usage: {sum(result_10[0][2]) / len(result_10[0][2])}\n")
    else:
        result_10 = model_instance.evaluate(top_k=10, metric=recall10, query_expansion=query_expansion, rerank=reranker, token_usage=True, time=True)
        f.write(f"\nRECALL@10: {result_10[1]}\n")
        f.write(f"Time: {result_10[0][3]}\n")
        f.write(f"Average Time: {result_10[0][3] / len(result_10[0][2])}\n")
        f.write(f"Token Usage: {sum(result_10[0][2])}\n")
        f.write(f"Average Token Usage: {sum(result_10[0][2]) / len(result_10[0][2])}\n")

    result_15 = model_instance.evaluate(top_k=15, metric=recall15, query_expansion=query_expansion, upper_bound_search = upper_bound_search, rerank=reranker, token_usage=True, time=True)
    f.write(f"\nRECALL@20: {result_15[1]}\n")
    f.write(f"Time: {result_15[0][3]}\n")
    f.write(f"Average Time: {result_15[0][3] / len(result_15[0][2])}\n")
    f.write(f"Token Usage: {sum(result_15[0][2])}\n")
    f.write(f"Average Token Usage: {sum(result_15[0][2]) / len(result_15[0][2])}\n\n")

    #Additional metrics can be added here, such as recall30, recall40, recall50

# Run experiments
with open("experiments.txt", "a") as f:
    f.write("Experiment Results\n")
    f.write("====================================\n")
    f.write("BASELINE\n")
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=1200, overlap_size=200, experiment_number=1, metadata_provider=None, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=2000, overlap_size=500, experiment_number=2, metadata_provider=None, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=3000, overlap_size=700, experiment_number=3, metadata_provider=None, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=6000, overlap_size=1500, experiment_number=4, metadata_provider=None, query_expansion=None, reranker=None)
    f.write("====================================\n")
    f.write("More Complex tests adding reranker, query expansion and metadata\n")
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=1200, overlap_size=200, experiment_number=5, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=2000, overlap_size=500, experiment_number=6, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=3000, overlap_size=700, experiment_number=7, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object1, chunk_size=6000, overlap_size=1500, experiment_number=8, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object2, chunk_size=1200, overlap_size=200, experiment_number=9, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object2, chunk_size=2000, overlap_size=500, experiment_number=10, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object3, chunk_size=1200, overlap_size=200, experiment_number=11, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object3, chunk_size=2000, overlap_size=500, experiment_number=12, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=None)
    f.write("====================================\n")
    single_experiment(f, object2, chunk_size=2000, overlap_size=500, experiment_number=13, upper_bound_search=100, metadata_provider=MetaDataProvider_instance, query_expansion=DictionaryQueryExpander_instance, reranker=None, name_query_expansion="DictionaryQueryExpander")
    f.write("====================================\n")
    single_experiment(f, object3, chunk_size=2000, overlap_size=500, experiment_number=14, metadata_provider=MetaDataProvider_instance, query_expansion=LLMQueryExpander_instance, reranker=None, name_query_expansion="LLMQueryExpander")
    f.write("====================================\n")
    single_experiment(f, object2, chunk_size=2000, overlap_size=500, experiment_number=15, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=LLMReranker_instance, name_reranker="LLMReranker")
    f.write("====================================\n")
    single_experiment(f, object2, chunk_size=2000, overlap_size=500, experiment_number=16, metadata_provider=MetaDataProvider_instance, query_expansion=None, reranker=CrossEncoderReranker_instance, name_reranker="CrossEncoderReranker")
    f.write("====================================\n")