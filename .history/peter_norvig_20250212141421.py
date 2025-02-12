import re
from collections import Counter
import json

class SpellingCorrector:
    def __init__(self, text_file_path=None, known_misspellings_file=None):
        self.word_counts = Counter()
        self.known_corrections = {}
        
        # Load training text if provided
        if text_file_path:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                self.word_counts.update(self.tokenize(text))
        
        # Load known misspellings if provided
        if known_misspellings_file:
            with open(known_misspellings_file, 'r') as file:
                misspellings_dict = json.load(file)
                for correct, wrong_list in misspellings_dict.items():
                    for wrong in wrong_list:
                        self.known_corrections[wrong.lower()] = correct.lower()

    def tokenize(self, text):
        """Convert text into a list of words."""
        return re.findall(r'\w+', text.lower())

    def edits1(self, word):
        """Generate all strings that are one edit away from the input word."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts    = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """Generate all strings that are two edits away from the input word."""
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def known(self, words):
        """Return the subset of words that appear in the dictionary."""
        return {w for w in words if w in self.word_counts or w in self.known_corrections}

    def get_candidates(self, word):
        """Generate possible spelling corrections for the word."""
        # First check known misspellings
        if word in self.known_corrections:
            return [self.known_corrections[word]]
        
        # Then try to find valid words with edit distance 1 or 2
        candidates = (
            {word} if word in self.word_counts else
            self.known(self.edits1(word)) or 
            self.known(self.edits2(word)) or 
            {word}  # If no better candidates found, return the original word
        )
        return candidates

    def correct(self, word):
        """Return the most probable spelling correction for the word."""
        word = word.lower()
        
        # First check known misspellings
        if word in self.known_corrections:
            return self.known_corrections[word]
        
        # Get all possible candidates
        candidates = self.get_candidates(word)
        
        # Return the candidate with the highest frequency in our training data
        return max(candidates, key=lambda w: self.word_counts[w] or 1)

    def correct_text(self, text):
        """Correct all words in a text."""
        return ' '.join(self.correct(word) for word in self.tokenize(text))

# Example usage:
if __name__ == "__main__":
    # Initialize the corrector with your training text and known misspellings
    corrector = SpellingCorrector(
        text_file_path="path_to_training_text.txt",  # Optional
        known_misspellings_file="path_to_misspellings.json"  # Optional
    )
    
    # Example corrections
    test_words = ["abilaty", "nevade", "steffen"]
    for word in test_words:
        correction = corrector.correct(word)
        print(f"Original: {word}, Corrected: {correction}")

    # Correct a whole text
    text = "My abilaty to spell is not very good"
    corrected_text = corrector.correct_text(text)
    print(f"\nOriginal text: {text}")
    print(f"Corrected text: {corrected_text}")