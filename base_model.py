from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, link_repo, model):
        self.model = model
        self.link_repo = link_repo
        self.index_name = None
        self.file_store_name = None
        self.repo_path = None

    @abstractmethod
    def clone_repo(self):
        pass

    @abstractmethod
    def build_cost(self, price_per_token, mutiple_chunks_per_file = False, chunk_size = 1200, overlap_size = 200):
        pass

    @abstractmethod
    def get_embedding(self, text):
        pass

    @abstractmethod
    def create_index(self, index_name, file_store_name, mutiple_chunks_per_file = False, chunk_size = 1200, overlap_size = 200 ):
        pass

    @abstractmethod
    def search(self, queries, top_k=1, upper_bound_search = 100, rerank=None, number_to_rerank = None, query_expansion=None, summary = False, token_usage = False, time = False):
        pass

    @abstractmethod
    def evaluate(self, top_k=1, rerank=None, query_expansion=None, number_to_rerank = None, metric=None):
        pass