import jellyfish
from collections import Counter
import json
import numpy as np

# Global variables to maintain state
_dictionary = {}
_word_frequencies = {}
_keyboard_distances = None

def create_keyboard_distance_matrix():
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

def load_dictionary(file_path, corpus_path=None):
    """Load dictionary with additional metadata"""
    global _dictionary, _word_frequencies, _keyboard_distances
    
    # Load dictionary with metadata
    _dictionary = {}
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip().lower()
            _dictionary[word] = {
                'metaphone': jellyfish.metaphone(word),
                'length': len(word)
            }
    
    # Load word frequencies if corpus provided
    if corpus_path:
        frequencies = Counter()
        try:
            with open(corpus_path, 'r') as file:
                for line in file:
                    words = line.strip().lower().split()
                    frequencies.update(words)
            total = sum(frequencies.values())
            _word_frequencies = {word: count/total for word, count in frequencies.items()}
        except FileNotFoundError:
            _word_frequencies = {}
    
    # Initialize keyboard distances
    _keyboard_distances = create_keyboard_distance_matrix()
    
    return list(_dictionary.keys())  # Return list of words for compatibility

def get_keyboard_distance(char1, char2):
    """Get the physical distance between two keys on the keyboard"""
    key = f"{char1}{char2}"
    return _keyboard_distances.get(key, float('inf'))

def wagner_fischer(s1, s2):
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
                keyboard_weight = get_keyboard_distance(s1[j-1], s2[i-1]) * 0.5
                change = previous_row[j-1] + 1 + keyboard_weight
            else:
                change = previous_row[j-1]
            
            current_row[j] = min(add, delete, change)

    return current_row[len_s1]

def get_phonetic_similarity(word1, word2):
    """Calculate phonetic similarity using Metaphone"""
    if word1 not in _dictionary or word2 not in _dictionary:
        return 0
    return 1 if _dictionary[word1]['metaphone'] == _dictionary[word2]['metaphone'] else 0

def spell_check(word, dictionary=None, context=None, max_suggestions=10):
    """
    Enhanced spell checking with multiple similarity metrics
    
    Args:
        word (str): Word to check
        dictionary (list): Ignored for compatibility
        context (list): Optional list of surrounding words
        max_suggestions (int): Maximum number of suggestions to return
    
    Returns:
        list: List of (word, score) tuples
    """
    word = word.lower()
    if word in _dictionary:
        return [(word, 1.0)]

    suggestions = []
    word_metaphone = jellyfish.metaphone(word)

    for dict_word in _dictionary:
        # Skip words with too large length difference
        if abs(len(dict_word) - len(word)) > 3:
            continue

        # Calculate different similarity metrics
        edit_distance = wagner_fischer(word, dict_word)
        phonetic_score = get_phonetic_similarity(dict_word, word)
        frequency_score = _word_frequencies.get(dict_word, 0.0)

        # Context score based on co-occurrence (if context provided)
        context_score = 0.0
        if context:
            context = [w.lower() for w in context]
            if dict_word in _word_frequencies:
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
    return [(word, -score) for word, score in suggestions[:max_suggestions]]  # Convert score to distance for compatibility

# Example usage
if __name__ == "__main__":
    # Initialize with dictionary and optional corpus
    dictionary = load_dictionary("words.txt", corpus_path="corpus.txt")
    
    # Test the spell checker
    test_words = ["wrlod", "phonetik", "recieve", "seperate"]
    for word in test_words:
        suggestions = spell_check(word)
        print(f"\nSuggestions for '{word}':")
        for suggestion, distance in suggestions:
            print(f"  {suggestion} (distance: {-distance:.3f})")