import re
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from jellyfish import metaphone, levenshtein_distance
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import torch
from transformers import BertTokenizer, BertModel

class AdvancedSpellingCorrector:
    def __init__(self, known_misspellings_file=None, context_aware=True):
        self.word_counts = Counter()
        self.known_corrections = {}
        self.min_word_freq = 2
        self.context_aware = context_aware
        self.ngram_model = None
        self.phonetic_dict = {}
        self.word_embeddings = {}
        
        # Initialize BERT for context-aware corrections
        if context_aware:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()

        # Download required NLTK data
        try:
            nltk.data.find('models/words')
        except LookupError:
            nltk.download('words')
            
        if known_misspellings_file:
            self.load_misspellings(known_misspellings_file)

    def load_misspellings(self, file_path):
        """Load known misspellings and build phonetic dictionary."""
        with open(file_path, 'r') as file:
            misspellings_dict = json.load(file)
            for correct, wrong_list in misspellings_dict.items():
                # Store direct mappings
                for wrong in wrong_list:
                    self.known_corrections[wrong.lower()] = correct.lower()
                
                # Build phonetic dictionary
                correct_phonetic = metaphone(correct)
                self.phonetic_dict[correct_phonetic] = self.phonetic_dict.get(correct_phonetic, [])
                self.phonetic_dict[correct_phonetic].append(correct)

    def build_ngram_model(self, text_data, n=3):
        """Build an n-gram language model for context-aware corrections."""
        # Prepare training data
        train_data, vocab = padded_everygram_pipeline(n, [self.tokenize(text) for text in text_data])
        
        # Train the model
        self.ngram_model = MLE(n)
        self.ngram_model.fit(train_data, vocab)

    def get_bert_embeddings(self, text, word_index):
        """Get BERT embeddings for context-aware word correction."""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.bert_model(**inputs)
            # Get the hidden states for the target word
            word_embedding = outputs.last_hidden_state[0, word_index + 1].numpy()
            return word_embedding

    def phonetic_candidates(self, word):
        """Generate candidates based on phonetic similarity."""
        word_phonetic = metaphone(word)
        candidates = set()
        
        # Add words with the same phonetic encoding
        if word_phonetic in self.phonetic_dict:
            candidates.update(self.phonetic_dict[word_phonetic])
        
        # Add words with similar phonetic encodings
        for phonetic in self.phonetic_dict:
            if levenshtein_distance(word_phonetic, phonetic) <= 1:
                candidates.update(self.phonetic_dict[phonetic])
        
        return candidates

    def context_score(self, candidate, context_before, context_after):
        """Score a candidate word based on its context using the n-gram model."""
        if not self.ngram_model:
            return 0
            
        context = context_before + [candidate] + context_after
        return self.ngram_model.score(candidate, context_before)

    def semantic_similarity(self, word_embedding, candidate):
        """Calculate semantic similarity using word embeddings."""
        if candidate in self.word_embeddings:
            candidate_embedding = self.word_embeddings[candidate]
            return np.dot(word_embedding, candidate_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(candidate_embedding))
        return 0

    def train(self, file_paths, min_word_freq=2):
        """Train the corrector with advanced NLP features."""
        text_data = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().lower()
                    words = self.tokenize(text)
                    text_data.append(words)
                    self.word_counts.update(words)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        # Build n-gram model
        self.build_ngram_model(text_data)
        
        # Build word embeddings if context-aware is enabled
        if self.context_aware:
            self.build_word_embeddings(text_data)

    def build_word_embeddings(self, text_data):
        """Build word embeddings using BERT."""
        print("Building word embeddings...")
        with torch.no_grad():
            for text in text_data:
                inputs = self.tokenizer(' '.join(text), return_tensors="pt", padding=True, truncation=True)
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[0]
                
                # Store embeddings for each word
                for i, word in enumerate(text):
                    if word not in self.word_embeddings:
                        self.word_embeddings[word] = embeddings[i].numpy()

    def get_candidates(self, word, context_before=None, context_after=None):
        """Generate correction candidates using multiple methods."""
        candidates = set()
        
        # Known corrections
        if word in self.known_corrections:
            candidates.add(self.known_corrections[word])
        
        # Edit distance candidates
        candidates.update(self.known(self.edits1(word)))
        if not candidates:
            candidates.update(self.known(self.edits2(word)))
        
        # Phonetic candidates
        candidates.update(self.phonetic_candidates(word))
        
        if not candidates:
            return {word}
            
        # Score candidates using context if available
        if context_before and context_after and self.context_aware:
            word_embedding = self.get_bert_embeddings(
                ' '.join(context_before + [word] + context_after),
                len(context_before)
            )
            
            # Score candidates using multiple factors
            scored_candidates = []
            for candidate in candidates:
                score = 0
                # Frequency score
                score += np.log(self.word_counts[candidate] + 1)
                # Context score
                score += self.context_score(candidate, context_before, context_after)
                # Semantic similarity score
                score += self.semantic_similarity(word_embedding, candidate)
                
                scored_candidates.append((candidate, score))
            
            # Return top candidates
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return {c[0] for c in scored_candidates[:3]}
        
        return candidates

    def correct(self, word, context_before=None, context_after=None):
        """Return the most probable spelling correction considering context."""
        word = word.lower()
        
        if len(word) <= 2 or word in self.word_counts:
            return word
            
        candidates = self.get_candidates(word, context_before, context_after)
        return max(candidates, key=lambda w: self.word_counts[w] or 1)

    def correct_text(self, text):
        """Correct all words in a text using context."""
        words = self.tokenize(text)
        corrected_words = []
        
        for i, word in enumerate(words):
            # Get context (up to 3 words before and after)
            context_before = words[max(0, i-3):i]
            context_after = words[i+1:min(len(words), i+4)]
            
            corrected = self.correct(word, context_before, context_after)
            
            # Preserve original capitalization
            if word[0].isupper():
                corrected = corrected.capitalize()
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)

    # Helper methods remain the same...
    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def edits1(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def known(self, words):
        return {w for w in words if w in self.word_counts}

# Example usage
if __name__ == "__main__":
    # Initialize with advanced features
    corrector = AdvancedSpellingCorrector(
        known_misspellings_file='sample_misspellings.json',
        context_aware=True
    )
    
    # Train the corrector
    corrector.train(['training_data/sample.txt'])
    
    # Test individual words with context
    text = "My abilaty to spell is not very good"
    print(f"Original text: {text}")
    print(f"Corrected text: {corrector.correct_text(text)}")