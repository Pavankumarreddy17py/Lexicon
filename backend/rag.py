import faiss
import numpy as np
import re
import torch
from backend.embeddings import EmbeddingModel
from sentence_transformers import util
from transformers import pipeline

# Load the lightweight model-Flan-T5-Base
device = 0 if torch.cuda.is_available() else -1

# Load the lightweight model-Flan-T5-Base
try:
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  
        device=device # Use GPU (0) if available, otherwise CPU (-1)
    )
    print("Flan-T5-Base Loaded Successfully")
except Exception as e:
    print(f"Error loading LLM: {e}")
    llm_pipeline = None

class RAGModel:
    def __init__(self):
        self.documents = {}
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.sentences = []

    def add_document(self, doc_name, text):
        """Processes a document and stores its text embeddings."""
        self.documents[doc_name] = text
        self.sentences = self.split_text(text)
        self._update_index()

    def split_text(self, text):
        """Splits text into small, meaningful sections for better retrieval."""
        return re.split(r'(?<=\.)\s+', text)  # Splits sentences properly

    def _update_index(self):
        """Creates a FAISS index from text embeddings using batch encoding."""
        if not self.sentences:
            self.index = None
            return
        
        # *** OPTIMIZATION: Use the model's batch encoding function directly ***
        # This is significantly faster than calling get_embedding(sentence) in a loop
        embeddings = self.embedding_model.model.encode(
            self.sentences,
            convert_to_numpy=True,
            show_progress_bar=False,  # Keep console clean during upload
            normalize_embeddings=True # Normalizing is good practice for vector search
        )
        
        if embeddings.ndim == 1:
            embeddings = np.array([embeddings])

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def rerank_results(self, question, retrieved_sentences):
        """Uses cosine similarity on existing embeddings (for speed) to pick the most relevant sentence."""
        # This function relies on embedding generation for each check, which can still be a bottleneck.
        # However, since it only runs on 10 or fewer sentences, the impact is less severe than the full index build.
        question_embedding = self.embedding_model.get_embedding(question)
        
        # Batch encode the retrieved sentences for faster similarity calculation
        retrieved_embeddings = self.embedding_model.model.encode(
            retrieved_sentences,
            convert_to_numpy=True,
            show_progress_bar=False 
        )
        
        # Calculate similarities in a single step
        similarities = util.pytorch_cos_sim(question_embedding, retrieved_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        
        return retrieved_sentences[best_match_idx]
    def clean_answer(self, text):
        """Removes unnecessary formatting from the final answer."""
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'(\d{4}-\d{2}-\d{2})', '', text)  # Remove any date patterns like '2024-25'
        text = re.sub(r'(^\d+/\w+)', '', text, flags=re.MULTILINE)  # Remove things like "2/KALEIDOSCOPE"
        return text.strip()

    def generate_llm_answer(self, question, context):
        """Generates a well-formed answer using Flan-T5-Base."""
        if llm_pipeline is None:
            return "Error: No local LLM is available."

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        try:
            response = llm_pipeline(prompt, max_length=250, truncation=True, do_sample=True, temperature=0.7)
            return response[0]["generated_text"].strip()
        except Exception as e:
            return f"Error generating LLM answer: {str(e)}"

    def answer_question(self, question):
        """Retrieves the best answer for the given question and provides better context."""
        if not self.index or not self.sentences:
            return "No documents uploaded yet."

        # 1. Get the vector embedding for the question
        question_vec = self.embedding_model.get_embedding(question).reshape(1, -1)
        
        # 2. Retrieve a larger pool of candidates (top 10)
        # We search a larger pool to give the reranker better options
        k = min(10, len(self.sentences)) # Ensure k is not larger than the number of sentences
        _, I = self.index.search(question_vec, k=k)
        
        candidate_indices = I[0]
        candidate_sentences = [self.sentences[idx] for idx in candidate_indices]

        # 3. Use the defined re-ranker to pick the single most relevant sentence from the candidates
        best_sentence = self.rerank_results(question, candidate_sentences)

        # 4. Find the index of the best sentence in the full sentence list to get its neighbors
        try:
            best_idx = self.sentences.index(best_sentence)
        except ValueError:
            # Fallback if the sentence isn't found (shouldn't happen)
            return self.generate_llm_answer(question, best_sentence)

        # 5. Collect the best sentence and its immediate neighbors for a focused context
        start_idx = max(0, best_idx - 1)
        end_idx = min(len(self.sentences), best_idx + 2) # +2 to include the best and the next sentence
        
        # Create a list of 1-3 highly focused sentences
        focused_context = [self.sentences[i] for i in range(start_idx, end_idx)]
        
        full_context = " ".join(focused_context) # Combine the best sentences

        # Pass the focused context to the LLM
        refined_answer = self.generate_llm_answer(question, full_context)

        return self.clean_answer(refined_answer)
