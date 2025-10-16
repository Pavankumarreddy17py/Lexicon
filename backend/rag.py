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
        self.all_sentences = [] # Stores all sentences for all documents

    def add_document(self, doc_name, text):
        """Processes a document and incrementally updates the index."""
        self.documents[doc_name] = text
        new_sentences = self.split_text(text)
        
        # 1. Update the master list of all sentences
        self.all_sentences.extend(new_sentences)
        
        # 2. Add embeddings for ONLY the new sentences (Incremental FAISS add)
        self._add_embeddings(new_sentences)

    def split_text(self, text):
        """Splits text into small, meaningful sections for better retrieval."""
        return re.split(r'(?<=\.)\s+', text)  # Splits sentences properly

    def _add_embeddings(self, sentences_to_embed):
        """Helper to generate embeddings and add them incrementally to FAISS."""
        if not sentences_to_embed:
            return

        # 1. Generate embeddings for the new sentences using batch encoding
        new_embeddings = self.embedding_model.model.encode(
            sentences_to_embed,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        if new_embeddings.ndim == 1:
            new_embeddings = np.array([new_embeddings])

        # 2. Initialize index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatL2(new_embeddings.shape[1])
            
        # 3. Add embeddings incrementally (FASTER than rebuilding)
        self.index.add(new_embeddings)

    # NOTE: The old _update_index method is removed as it's no longer necessary.

    def rerank_results(self, question, retrieved_sentences):
        """Uses cosine similarity on existing embeddings (for speed) to pick the most relevant sentence."""
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
        # FIXED: Check against self.all_sentences
        if not self.index or not self.all_sentences:
            return "No documents uploaded yet."

        # 1. Get the vector embedding for the question
        question_vec = self.embedding_model.get_embedding(question).reshape(1, -1)
        
        # 2. Retrieve a larger pool of candidates (top 10)
        # FIXED: Use self.all_sentences for length check
        k = min(10, len(self.all_sentences)) # Ensure k is not larger than the number of sentences
        _, I = self.index.search(question_vec, k=k)
        
        candidate_indices = I[0]
        # FIXED: Use self.all_sentences for retrieving candidates
        candidate_sentences = [self.all_sentences[idx] for idx in candidate_indices]

        # 3. Use the defined re-ranker to pick the single most relevant sentence from the candidates
        best_sentence = self.rerank_results(question, candidate_sentences)

        # 4. Find the index of the best sentence in the full sentence list to get its neighbors
        try:
            # FIXED: Use self.all_sentences to find the index
            best_idx = self.all_sentences.index(best_sentence)
        except ValueError:
            # Fallback if the sentence isn't found (shouldn't happen)
            return self.generate_llm_answer(question, best_sentence)

        # 5. Collect the best sentence and its immediate neighbors for a focused context
        start_idx = max(0, best_idx - 1)
        # FIXED: Use self.all_sentences for length check
        end_idx = min(len(self.all_sentences), best_idx + 2) # +2 to include the best and the next sentence
        
        # Create a list of 1-3 highly focused sentences
        # FIXED: Use self.all_sentences to construct the focused context
        focused_context = [self.all_sentences[i] for i in range(start_idx, end_idx)]
        
        full_context = " ".join(focused_context) # Combine the best sentences

        # Pass the focused context to the LLM
        refined_answer = self.generate_llm_answer(question, full_context)

        return self.clean_answer(refined_answer)