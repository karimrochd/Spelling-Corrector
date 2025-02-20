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

    def train_from_directory(self, directory_path, file_extensions=['.txt'], min_word_freq=2):
        """Train the corrector on all text files in a directory."""
        directory = Path(directory_path)
        text_files = []
        for ext in file_extensions:
            text_files.extend(directory.glob(f"*{ext}"))
        
        print(f"Found {len(text_files)} files to process")
        self.train(text_files, min_word_freq)

    def train(self, file_paths, min_word_freq=2):
        """Train the corrector with advanced NLP features."""
        text_data = []
        raw_texts = []
        total_files = len(file_paths)
        
        print("Starting training process...")
        print("Phase 1: Reading and processing text files")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"\rProcessing file {i}/{total_files}: {file_path}", end='')
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().lower()
                    words = self.tokenize(text)
                    
                    # Only keep reasonable-length chunks for BERT
                    if len(words) > 512:
                        chunks = [words[i:i + 512] for i in range(0, len(words), 512)]
                        text_data.extend(chunks)
                    else:
                        text_data.append(words)
                    
                    raw_texts.append(text)
                    self.word_counts.update(words)
            except Exception as e:
                print(f"\nError processing file {file_path}: {str(e)}")

        print("\n\nPhase 2: Building vocabulary")
        self.word_counts = Counter({word: count 
                                for word, count in self.word_counts.items() 
                                if count >= min_word_freq})

        print("\nPhase 3: Building n-gram model")
        self.build_ngram_model(raw_texts)
        
        if self.context_aware:
            print("\nPhase 4: Building word embeddings")
            self.build_word_embeddings(text_data)

    def build_ngram_model(self, texts, n=3):
        """Build an n-gram language model for context-aware corrections."""
        # Process each text into sentences of tokens
        tokenized_texts = [self.tokenize(text) for text in texts]
        
        # Create the n-gram model
        train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)
        self.ngram_model = MLE(n)
        self.ngram_model.fit(train_data, vocab)


    def build_word_embeddings(self, text_data):
        """Build word embeddings using BERT."""
        print("Building word embeddings...")
        MAX_LENGTH = 510  # BERT maximum length minus special tokens
        
        with torch.no_grad():
            for text in text_data:
                # Process text in chunks
                for i in range(0, len(text), MAX_LENGTH):
                    chunk = text[i:i + MAX_LENGTH]
                    chunk_text = ' '.join(chunk)
                    
                    try:
                        inputs = self.tokenizer(chunk_text, 
                                            return_tensors="pt", 
                                            padding=True, 
                                            truncation=True,
                                            max_length=512)
                        outputs = self.bert_model(**inputs)
                        embeddings = outputs.last_hidden_state[0, 1:-1]  # Remove [CLS] and [SEP] tokens
                        
                        # Match embeddings to words
                        for j, word in enumerate(chunk):
                            if j < len(embeddings) and word not in self.word_embeddings:
                                self.word_embeddings[word] = embeddings[j].numpy()
                    except Exception as e:
                        print(f"Error processing chunk: {str(e)}")
                        continue

    def get_bert_embeddings(self, text, word_index):
        """Get BERT embeddings for context-aware word correction."""
        with torch.no_grad():
            try:
                inputs = self.tokenizer(text, 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=True,
                                    max_length=512)
                outputs = self.bert_model(**inputs)
                
                # Ensure word_index is within bounds
                if word_index + 1 >= outputs.last_hidden_state.size(1) - 1:
                    return None
                    
                word_embedding = outputs.last_hidden_state[0, word_index + 1].numpy()
                return word_embedding
            except Exception as e:
                print(f"Error getting embeddings: {str(e)}")
                return None

    
    def phonetic_candidates(self, word):
        """Generate candidates based on phonetic similarity."""
        word_phonetic = metaphone(word)
        candidates = set()
        
        if word_phonetic in self.phonetic_dict:
            candidates.update(self.phonetic_dict[word_phonetic])
        
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

    def edits1(self, word):
        """Generate all strings that are one edit away from the input word."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """Generate all strings that are two edits away from the input word."""
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def known(self, words):
        """Return the subset of words that appear in the dictionary."""
        return {w for w in words if w in self.word_counts}

    def tokenize(self, text):
        """Convert text into a list of words."""
        return re.findall(r'\w+', text.lower())

    def correct(self, word, context_before=None, context_after=None):
        """Return the most probable spelling correction considering context."""
        word = word.lower()
        
        # Don't correct short words or known words
        if len(word) <= 2 or word in self.word_counts:
            return word
            
        # First check known misspellings dictionary
        if word in self.known_corrections:
            return self.known_corrections[word]
        
        # Get candidates
        candidates = self.get_candidates(word, context_before, context_after)
        
        # If no candidates found, return original word
        if not candidates or candidates == {word}:
            return word
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = 0
            
            # Higher weight for edit distance
            if candidate in self.known(self.edits1(word)):
                score += 1.0
            elif candidate in self.known(self.edits2(word)):
                score += 0.5
                
            # Weight for word frequency
            score += np.log(self.word_counts.get(candidate, 0) + 1) * 0.5
            
            # Weight for phonetic similarity
            if metaphone(candidate) == metaphone(word):
                score += 12.0
                
            # Context score if available
            if context_before and context_after:
                context_score = self.context_score(candidate, context_before, context_after)
                score += context_score * 8.0
            
            scored_candidates.append((candidate, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Only return a correction if we're confident enough
        best_candidate, best_score = scored_candidates[0]
        if best_score < 1.0:  # Threshold for correction
            return word
            
        return best_candidate

    def get_candidates(self, word, context_before=None, context_after=None):
        """Generate correction candidates using multiple methods."""
        candidates = set()
        
        # Known corrections first
        if word in self.known_corrections:
            return {self.known_corrections[word]}
        
        # Edit distance 1 candidates
        edit1_candidates = self.known(self.edits1(word))
        if edit1_candidates:
            candidates.update(edit1_candidates)
        
        # Phonetic candidates
        phonetic_candidates = self.phonetic_candidates(word)
        if phonetic_candidates:
            candidates.update(phonetic_candidates)
        
        # Only try edit distance 2 if we don't have enough candidates
        if len(candidates) < 3:
            edit2_candidates = self.known(self.edits2(word))
            candidates.update(edit2_candidates)
        
        # If still no candidates, return original word
        if not candidates:
            return {word}
        
        return candidates

    def correct_text(self, text):
        """Correct all words in a text using context."""
        words = self.tokenize(text)
        corrected_words = []
        
        for i, word in enumerate(words):
            # Get context (up to 2 words before and after)
            context_before = words[max(0, i-2):i]
            context_after = words[i+1:min(len(words), i+3)]
            
            # Only try to correct words that might be misspelled
            if not word.isalpha() or word in self.word_counts or len(word) <= 2:
                corrected_words.append(word)
                continue
                
            corrected = self.correct(word, context_before, context_after)
            
            # Preserve original capitalization
            if word[0].isupper():
                corrected = corrected.capitalize()
                
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)

    def get_stats(self):
        """Return statistics about the training data and model."""
        stats = {
            'vocabulary_size': len(self.word_counts),
            'total_words': sum(self.word_counts.values()),
            'known_misspellings': len(self.known_corrections),
            'most_common_words': self.word_counts.most_common(20),
            'features_enabled': {
                'context_aware': self.context_aware,
                'ngram_model': self.ngram_model is not None,
                'phonetic_matching': bool(self.phonetic_dict),
                'word_embeddings': bool(self.word_embeddings)
            }
        }
        return stats

if __name__ == "__main__":
    # Initialize with advanced features
    corrector = AdvancedSpellingCorrector(
        known_misspellings_file='spelling_dictionary.json',
        context_aware=True
    )
    
    # Train on directory
    corrector.train_from_directory('')
    
    # Print statistics
    stats = corrector.get_stats()
    print("\nTraining Statistics:")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Total words processed: {stats['total_words']}")
    print(f"Known misspellings: {stats['known_misspellings']}")
    print("\nEnabled features:")
    for feature, enabled in stats['features_enabled'].items():
        print(f"- {feature}: {'✓' if enabled else '✗'}")
    print("\nMost common words:")
    for word, count in stats['most_common_words'][:10]:
        print(f"{word}: {count}")
    
    # Test the corrector
    test_texts = [
        "My abilaty to spell is not very good",
        "He went to the store yestarday",
        "The weather is beautifull today"
    ]
    
    print("\nTesting corrections:")
    for text in test_texts:
        corrected = corrector.correct_text(text)
        print(f"\nOriginal: {text}")
        print(f"Corrected: {corrected}")
        
        
