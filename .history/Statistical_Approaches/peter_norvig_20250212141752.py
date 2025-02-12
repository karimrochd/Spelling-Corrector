import re
from collections import Counter
import json

class SpellingCorrector:
    def __init__(self, text_file_path=None, known_misspellings_file=None):
        self.word_counts = Counter()
        self.known_corrections = {}
        
        # Initialize with a basic set of common words to prevent overcorrection
        common_words = """
        the be to of and a in that have i it for not on with he as you do at
        this but his by from they we say her she or an will my one all would there
        their what so up out if about who get which go me when make can like time
        no just him know take people into year your good some could them see other
        than then now look only come its over think also back after use two how
        our work first well way even new want because any these give day most us
        """.split()
        self.word_counts.update(common_words)
        
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
        # First check if the word is already known
        if word in self.word_counts:
            return {word}
            
        # Then check known misspellings
        if word in self.known_corrections:
            return {self.known_corrections[word]}
        
        # Then try edit distance 1
        edit1_candidates = self.known(self.edits1(word))
        if edit1_candidates:
            return edit1_candidates
            
        # Finally try edit distance 2
        edit2_candidates = self.known(self.edits2(word))
        if edit2_candidates:
            return edit2_candidates
            
        # If no candidates found, return the original word
        return {word}

    def correct(self, word):
        """Return the most probable spelling correction for the word."""
        word = word.lower()
        
        # Don't correct short words (likely to be correct)
        if len(word) <= 2:
            return word
            
        # Don't correct words that are already known
        if word in self.word_counts and self.word_counts[word] > 0:
            return word
        
        # Check known misspellings
        if word in self.known_corrections:
            return self.known_corrections[word]
        
        # Get candidates and select the most probable one
        candidates = self.get_candidates(word)
        best_candidate = max(candidates, key=lambda w: self.word_counts[w] or 1)
        
        # Only return the correction if we're confident enough
        if self.word_counts[best_candidate] > 1:
            return best_candidate
        return word

    def correct_text(self, text):
        """Correct all words in a text."""
        words = self.tokenize(text)
        corrected_words = []
        
        for i, word in enumerate(words):
            corrected = self.correct(word)
            # Preserve original capitalization
            if word[0].isupper():
                corrected = corrected.capitalize()
            corrected_words.append(corrected)
            
        return ' '.join(corrected_words)

# Example usage:
if __name__ == "__main__":
    # Sample misspellings dictionary
    sample_misspellings = {
        "ability": ["abilaty", "abillity", "ablity", "abilty", "abilitey", "abbility"],
        "nevada": ["nevade"],
        "stephen": ["steffen"]
    }
    
    # Save sample misspellings to a temporary file
    with open('sample_misspellings.json', 'w') as f:
        json.dump(sample_misspellings, f)
    
    # Initialize the corrector
    corrector = SpellingCorrector(known_misspellings_file='sample_misspellings.json')
    
    # Test individual words
    test_words = ["abilaty", "nevade", "steffen"]
    for word in test_words:
        correction = corrector.correct(word)
        print(f"Original: {word}, Corrected: {correction}")
    
    # Test text correction
    text = "My abilaty to spell is not very good"
    corrected_text = corrector.correct_text(text)
    print(f"\nOriginal text: {text}")
    print(f"Corrected text: {corrected_text}")