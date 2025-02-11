import jellyfish  # For phonetic matching
from collections import Counter
import json
import numpy as np

class SpellChecker:
    def __init__(self, dictionary_path, corpus_path=None):
        """
        Initialize the spell checker with a dictionary and optional corpus for word frequencies
        
        Args:
            dictionary_path (str): Path to dictionary file
            corpus_path (str): Path to text corpus for word frequencies
        """
        self.dictionary = self.load_dictionary(dictionary_path)
        self.word_frequencies = self.load_word_frequencies(corpus_path) if corpus_path else {}
        self.keyboard_distances = self.create_keyboard_distance_matrix()
        self.common_mistakes = self.load_common_mistakes()

    def load_dictionary(self, file_path):
        """Load dictionary with additional metadata"""
        words = {}
        with open(file_path, 'r') as file:
            for line in file:
                word = line.strip().lower()
                words[word] = {
                    'metaphone': jellyfish.metaphone(word),
                    'length': len(word)
                }
        return words

    def load_word_frequencies(self, corpus_path):
        """Calculate word frequencies from a corpus"""
        frequencies = Counter()
        with open(corpus_path, 'r') as file:
            for line in file:
                words = line.strip().lower().split()
                frequencies.update(words)
        # Convert to probabilities
        total = sum(frequencies.values())
        return {word: count/total for word, count in frequencies.items()}

    def create_keyboard_distance_matrix(self):
        """Create a matrix of QWERTY keyboard distances"""
        keyboard_layout = {
            'q': (0,0), 'w': (0,1), 'e': (0,2), 'r': (0,3), 't': (0,4),
            'y': (0,5), 'u': (0,6), 'i': (0,7), 'o': (0,8), 'p': (0,9),
            'a': (1,0), 's': (1,1), 'd': (1,2), 'f': (1,3), 'g': (1,4),
            'h': (1,5), 'j': (1,6), 'k': (1,7), 'l': (1,8),
            'z': (2,0), 'x': (2,1), 'c': (2,2), 'v': (2,3), 'b': (2,4),
            'n': (2,5), 'm': (2,6)
        }
        
        distances = {}
        for c1 in keyboard_layout:
            for c2 in keyboard_layout:
                pos1 = keyboard_layout[c1]
                pos2 = keyboard_layout[c2]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances[f"{c1}{c2}"] = distance
        return distances

    def load_common_mistakes(self):
        """Load dictionary of common spelling mistakes"""
        return {
            'ie': 'ei',
            'separate': 'seperate',
            'definitely': 'definately',
            'receive': 'recieve',
            # Add more common mistakes here
        }

    def get_keyboard_distance(self, char1, char2):
        """Get the physical distance between two keys on the keyboard"""
        key = f"{char1}{char2}"
        return self.keyboard_distances.get(key, float('inf'))

    def wagner_fischer(self, s1, s2):
        """Enhanced Wagner-Fischer with keyboard distance weighting"""
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 > len_s2:
            s1, s2 = s2, s1
            len_s1, len_s2 = len_s2, len_s1

        current_row = range(len_s1 + 1)
        for i in range(1, len_s2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len_s1
            for j in range(1, len_s1 + 1):
                add, delete = previous_row[j] + 1, current_row[j-1] + 1
                
                # Weight substitution cost by keyboard distance
                if s1[j-1] != s2[i-1]:
                    keyboard_weight = self.get_keyboard_distance(s1[j-1], s2[i-1]) * 0.5
                    change = previous_row[j-1] + 1 + keyboard_weight
                else:
                    change = previous_row[j-1]
                
                current_row[j] = min(add, delete, change)

        return current_row[len_s1]

    def get_phonetic_similarity(self, word1, word2):
        """Calculate phonetic similarity using Metaphone"""
        return 1 if self.dictionary[word1]['metaphone'] == self.dictionary[word2]['metaphone'] else 0

    def spell_check(self, word, context=None, max_suggestions=10):
        """
        Enhanced spell checking with multiple similarity metrics
        
        Args:
            word (str): Word to check
            context (list): Optional list of surrounding words
            max_suggestions (int): Maximum number of suggestions to return
        
        Returns:
            list: Sorted list of (word, score) tuples
        """
        word = word.lower()
        if word in self.dictionary:
            return [(word, 1.0)]

        suggestions = []
        word_metaphone = jellyfish.metaphone(word)

        for dict_word in self.dictionary:
            # Skip words with too large length difference
            if abs(len(dict_word) - len(word)) > 3:
                continue

            # Calculate different similarity metrics
            edit_distance = self.wagner_fischer(word, dict_word)
            phonetic_score = self.get_phonetic_similarity(dict_word, word)
            frequency_score = self.word_frequencies.get(dict_word, 0.0)

            # Context score based on co-occurrence (if context provided)
            context_score = 0.0
            if context:
                context = [w.lower() for w in context]
                if dict_word in self.word_frequencies:
                    # Simple context score based on presence in surrounding words
                    context_score = 1.0 if dict_word in context else 0.0

            # Combine scores with weights
            total_score = (
                (1.0 / (1 + edit_distance)) * 0.4 +
                phonetic_score * 0.3 +
                frequency_score * 0.2 +
                context_score * 0.1
            )

            suggestions.append((dict_word, total_score))

        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]

    def check_text(self, text):
        """
        Check and correct a full text, using context for better suggestions
        
        Args:
            text (str): Text to check
            
        Returns:
            tuple: (corrected text, list of corrections made)
        """
        words = text.split()
        corrections = []
        
        for i, word in enumerate(words):
            # Get context (surrounding words)
            start = max(0, i - 2)
            end = min(len(words), i + 3)
            context = words[start:i] + words[i+1:end]
            
            # Get suggestions with context
            suggestions = self.spell_check(word, context=context)
            
            if suggestions and suggestions[0][0] != word:
                corrections.append((word, suggestions[0][0]))
                words[i] = suggestions[0][0]
        
        return ' '.join(words), corrections

# Example usage:
if __name__ == "__main__":
    # Initialize spell checker with dictionary and corpus
    spell_checker = SpellChecker(
        dictionary_path="words.txt",
        corpus_path="corpus.txt"  # Optional: for word frequencies
    )
    
    # Example 1: Check single word with context
    word = "wrlod"
    context = ["hello", "beautiful"]
    suggestions = spell_checker.spell_check(word, context=context)
    print(f"\nSuggestions for '{word}' in context {context}:")
    for word, score in suggestions:
        print(f"{word}: {score:.3f}")
    
    # Example 2: Check full text
    text = "The quik brown foks jumps ovr the lazy dog"
    corrected_text, corrections = spell_checker.check_text(text)
    print(f"\nOriginal text: {text}")
    print(f"Corrected text: {corrected_text}")
    print("Corrections made:")
    for original, corrected in corrections:
        print(f"  {original} -> {corrected}")
