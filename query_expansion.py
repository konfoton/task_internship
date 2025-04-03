import nltk
from nltk.corpus import wordnet as wn
from openai import OpenAI
from typing import List, Tuple, Dict
from pydantic import BaseModel
class ExpandedQuery(BaseModel):
     expanded_query: str

class DictionaryQueryExpander:
    def __init__(self, top_n):
        self.top_n = top_n
    
    def get_top_synonyms(self, word):
        synonyms = {}
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms[lemma.name().replace('_', ' ')] = lemma.count()

        sorted_synonyms = sorted(synonyms.items(), key=lambda item: item[1], reverse=True)
        top_synonyms = [synonym for synonym, count in sorted_synonyms[:self.top_n]]
        return top_synonyms

    def expand_query(self, query):
        alternative_queries = [[] for _ in range(self.top_n)]
        for word in query.split():
            synonyms = self.get_top_synonyms(word)
            for index, synonym in enumerate(synonyms):
                alternative_queries[index].append(synonym)
        
        return [[' '.join(query) for query in alternative_queries], 0]
    

class LLMQueryExpander: 
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()

    def expand_query(self, query: str) -> tuple:
        """
        Expands a user-provided query using an LLM for semantic richness.
        Also returns total token usage from the API call.
        """
        completion = self.client.beta.chat.completions.parse(
            model = self.model,
            messages=[
                {
                    "role": "system",
                    "content": "expands queries for improved search results."
                },
                {
                    "role": "user",
                    "content": f"Expand the following query for better semantic coverage. \n Display only expanded query \n only few sentences: '{query}' "
                }
            ],
            response_format=ExpandedQuery,
            temperature=0.7
        )
        expanded_query = completion.choices[0].message.parsed.expanded_query  
        total_tokens = completion.usage.total_tokens
        return [[expanded_query], total_tokens]