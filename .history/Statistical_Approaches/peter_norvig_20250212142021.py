import re
from collections import Counter
import json
from pathlib import Path

class SpellingCorrector:
    def __init__(self, known_misspellings_file=None):
        self.word_counts = Counter()
        self.known_corrections = {}
        self.min_word_freq = 2  # Minimum frequency threshold
        
        # Initialize with basic common words
        common_words = """
        the be to of and a in that have i it for not on with he as you do at
        this but his by from they we say her she or an will my one all would there
        their what so up out if about who get which go me when make can like time
        no just him know take people into year your good some could them see other
        than then now look only come its over think also back after use two how
        our work first well way even new want because any these give day most us
        """.split()
        self.word_counts.update(common_words)
        
        # Load known misspellings if provided
        if known_misspellings_file:
            self.load_misspellings(known_misspellings_file)

    def load_misspellings(self, file_path):
        """Load known misspellings from a JSON file."""
        with open(file_path, 'r') as file:
            misspellings_dict = json.load(file)
            for correct, wrong_list in misspellings_dict.items():
                for wrong in wrong_list:
                    self.known_corrections[wrong.lower()] = correct.lower()

    def train(self, file_paths, min_word_freq=2):
        """
        Train the corrector on multiple text files.
        
        Args:
            file_paths: List of paths to training text files
            min_word_freq: Minimum frequency threshold for considering a word valid
        """
        self.min_word_freq = min_word_freq
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().lower()
                    words = self.tokenize(text)
                    # Only count words with alphabetic characters
                    valid_words = [word for word in words if any(c.isalpha() for c in word)]
                    self.word_counts.update(valid_words)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        # Remove low-frequency words
        self.word_counts = Counter({word: count 
                                  for word, count in self.word_counts.items() 
                                  if count >= min_word_freq})
        
        print(f"Vocabulary size after training: {len(self.word_counts)}")
        print(f"Total words processed: {sum(self.word_counts.values())}")

    def train_from_directory(self, directory_path, file_extensions=['.txt'], min_word_freq=2):
        """
        Train the corrector on all text files in a directory.
        
        Args:
            directory_path: Path to directory containing training files
            file_extensions: List of file extensions to include
            min_word_freq: Minimum frequency threshold for considering a word valid
        """
        directory = Path(directory_path)
        text_files = []
        for ext in file_extensions:
            text_files.extend(directory.glob(f"*{ext}"))
        
        print(f"Found {len(text_files)} files to process")
        self.train(text_files, min_word_freq)

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
        return {w for w in words if w in self.word_counts}

    def get_candidates(self, word):
        """Generate possible spelling corrections for the word."""
        if word in self.word_counts and self.word_counts[word] >= self.min_word_freq:
            return {word}
            
        if word in self.known_corrections:
            return {self.known_corrections[word]}
        
        edit1_candidates = self.known(self.edits1(word))
        if edit1_candidates:
            return edit1_candidates
            
        edit2_candidates = self.known(self.edits2(word))
        if edit2_candidates:
            return edit2_candidates
            
        return {word}

    def correct(self, word):
        """Return the most probable spelling correction for the word."""
        word = word.lower()
        
        if len(word) <= 2:
            return word
            
        if word in self.word_counts and self.word_counts[word] >= self.min_word_freq:
            return word
        
        if word in self.known_corrections:
            return self.known_corrections[word]
        
        candidates = self.get_candidates(word)
        return max(candidates, key=lambda w: self.word_counts[w] or 1)

    def correct_text(self, text):
        """Correct all words in a text."""
        words = self.tokenize(text)
        corrected_words = []
        
        for word in words:
            corrected = self.correct(word)
            # Preserve original capitalization
            if word[0].isupper():
                corrected = corrected.capitalize()
            corrected_words.append(corrected)
            
        return ' '.join(corrected_words)

    def get_stats(self):
        """Return statistics about the training data."""
        return {
            'vocabulary_size': len(self.word_counts),
            'total_words': sum(self.word_counts.values()),
            'known_misspellings': len(self.known_corrections),
            'most_common_words': self.word_counts.most_common(10)
        }

# Example usage with training:
if __name__ == "__main__":
    # Initialize corrector with known misspellings
    corrector = SpellingCorrector('spelling_dictionary.json')
    
    # Train on a directory of text files
    corrector.train_from_directory(
        '',  # Directory containing training text files
        file_extensions=['.txt'],
        min_word_freq=2
    )
    
    # Print statistics
    stats = corrector.get_stats()
    print("\nTraining Statistics:")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Total words processed: {stats['total_words']}")
    print(f"Known misspellings: {stats['known_misspellings']}")
    print("\nMost common words:")
    for word, count in stats['most_common_words']:
        print(f"{word}: {count}")
    
    # Test the corrector
    test_words = ["abilaty", "nevade", "steffen"]
    print("\nTesting corrections:")
    for word in test_words:
        correction = corrector.correct(word)
        print(f"Original: {word}, Corrected: {correction}")
    
    text = "My abilaty to spell is not very good"
    corrected_text = corrector.correct_text(text)
    print(f"\nOriginal text: {text}")
    print(f"Corrected text: {corrected_text}")