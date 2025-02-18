import json
import random
from pathlib import Path
from peter_norvig import SpellingCorrector

def evaluate_spelling_corrector(dictionary_path, corpus_dir, test_ratio=0.4):
    """
    Evaluate the spelling corrector's accuracy using corpus data and a test dictionary.
    
    Args:
        dictionary_path: Path to the JSON dictionary file
        corpus_dir: Directory containing training text files
        test_ratio: Proportion of dictionary to use as test set (default 0.4)
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Load the full dictionary
    with open(dictionary_path, 'r') as f:
        full_dict = json.load(f)
    
    # Create lists of (misspelling, correct_word) pairs
    word_pairs = []
    for correct_word, misspellings in full_dict.items():
        for misspelling in misspellings:
            word_pairs.append((misspelling.lower(), correct_word.lower()))
    
    # Randomly shuffle and split into train/test sets
    random.seed(42)  # For reproducibility
    random.shuffle(word_pairs)
    split_idx = int(len(word_pairs) * (1 - test_ratio))
    
    train_pairs = word_pairs[:split_idx]
    test_pairs = word_pairs[split_idx:]
    
    # Create training dictionary
    train_dict = {}
    for misspelling, correct_word in train_pairs:
        if correct_word not in train_dict:
            train_dict[correct_word] = []
        train_dict[correct_word].append(misspelling)
    
    # Save training dictionary to temporary file
    train_dict_path = 'train_spelling_dict.json'
    with open(train_dict_path, 'w') as f:
        json.dump(train_dict, f)
    
    # Initialize corrector with known misspellings
    corrector = SpellingCorrector(train_dict_path)
    
    # Get all text files from the corpus directory
    corpus_dir = Path(corpus_dir)
    text_files = list(corpus_dir.glob('*.txt'))
    print(f"Found {len(text_files)} text files for training")
    
    # Train on the corpus
    print("Training corrector on corpus files...")
    corrector.train(text_files, min_word_freq=2)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    total = len(test_pairs)
    correct = 0
    incorrect_examples = []
    
    for misspelling, true_word in test_pairs:
        predicted_word = corrector.correct(misspelling)
        if predicted_word == true_word:
            correct += 1
        else:
            incorrect_examples.append({
                'misspelling': misspelling,
                'predicted': predicted_word,
                'true_word': true_word
            })
    
    accuracy = correct / total
    
    # Return detailed results
    return {
        'total_examples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'train_size': len(train_pairs),
        'test_size': len(test_pairs),
        'corpus_files': len(text_files),
        'vocab_size': len(corrector.word_counts),
        'incorrect_examples': incorrect_examples[:10]  # First 10 mistakes
    }

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_spelling_corrector(
        'spelling_dictionary.json',
        ''  # Directory containing .txt files
    )
    
    # Print results
    print("\nSpelling Corrector Evaluation Results")
    print("-" * 40)
    print(f"Training set size: {results['train_size']} pairs")
    print(f"Test set size: {results['test_size']} pairs")
    print(f"Corpus files used: {results['corpus_files']}")
    print(f"Vocabulary size: {results['vocab_size']} words")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Total test examples: {results['total_examples']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Print some incorrect examples
    if results['incorrect_examples']:
        print("\nSample of incorrect predictions:")
        print("-" * 40)
        for ex in results['incorrect_examples']:
            print(f"Misspelling: {ex['misspelling']}")
            print(f"Predicted:   {ex['predicted']}")
            print(f"True word:   {ex['true_word']}")
            print()



'''
Spelling Corrector Evaluation Results
----------------------------------------
Training set size: 24091 pairs
Test set size: 16062 pairs
Corpus files used: 5
Vocabulary size: 21569 words
Correct predictions: 4800
Total test examples: 16062
Accuracy: 29.88%

Sample of incorrect predictions:
----------------------------------------
Misspelling: eniceation
Predicted:   enucleation
True word:   initiation

Misspelling: seartenly
Predicted:   seartenly
True word:   certainly

Misspelling: scouce
Predicted:   source
True word:   saucer

Misspelling: garle
Predicted:   gale
True word:   gallery

Misspelling: lifd 1
Predicted:   lifd 1
True word:   lived

Misspelling: traing 1
Predicted:   traing 1
True word:   trying

Misspelling: magnifas
Predicted:   magnifas
True word:   magnificent

Misspelling: contributers
Predicted:   contributed
True word:   contributors

Misspelling: strawes
Predicted:   states
True word:   straws

Misspelling: afedavit
Predicted:   afedavit
True word:   affidavit

'''