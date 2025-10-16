from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model

    def get_embedding(self, text):
        """Generates embedding for given text"""
        return np.array(self.model.encode(text, convert_to_numpy=True))
