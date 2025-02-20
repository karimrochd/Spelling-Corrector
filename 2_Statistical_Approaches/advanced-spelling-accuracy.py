import json
import random
from pathlib import Path
from advanced_statistical_method import AdvancedSpellingCorrector

def evaluate_advanced_spelling_corrector(dictionary_path, corpus_dir, test_ratio=0.4, context_aware=True):
    """
    Evaluate the advanced spelling corrector's accuracy using corpus data and a test dictionary.
    
    Args:
        dictionary_path: Path to the JSON dictionary file
        corpus_dir: Directory containing training text files
        test_ratio: Proportion of dictionary to use as test set (default 0.4)
        context_aware: Whether to enable context-aware corrections (default True)
    
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
    
    # Initialize corrector
    print("Initializing Advanced Spelling Corrector...")
    corrector = AdvancedSpellingCorrector(
        known_misspellings_file=train_dict_path,
        context_aware=context_aware
    )
    
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
    
    # Function to get context from a word's actual usage in corpus
    def get_sample_context(word):
        """Find a sample context for the word from the corpus."""
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().lower()
                    words = corrector.tokenize(text)
                    for i, w in enumerate(words):
                        if w == word:
                            start = max(0, i - 2)
                            end = min(len(words), i + 3)
                            return words[start:i], words[i+1:end]
            except Exception:
                continue
        return [], []  # Return empty context if word not found
    
    for misspelling, true_word in test_pairs:
        # Get real context from corpus if possible
        context_before, context_after = get_sample_context(true_word)
        
        # Test both with and without context
        predicted_word_no_context = corrector.correct(misspelling)
        predicted_word_with_context = corrector.correct(
            misspelling,
            context_before=context_before,
            context_after=context_after
        )
        
        # Count as correct if either method gets it right
        if predicted_word_no_context == true_word or predicted_word_with_context == true_word:
            correct += 1
        else:
            incorrect_examples.append({
                'misspelling': misspelling,
                'predicted_no_context': predicted_word_no_context,
                'predicted_with_context': predicted_word_with_context,
                'true_word': true_word,
                'context_used': ' '.join([*context_before, '[WORD]', *context_after])
            })
    
    accuracy = correct / total
    
    # Get model stats
    model_stats = corrector.get_stats()
    
    # Return detailed results
    return {
        'total_examples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'train_size': len(train_pairs),
        'test_size': len(test_pairs),
        'corpus_files': len(text_files),
        'model_stats': model_stats,
        'incorrect_examples': incorrect_examples[:10]  # First 10 mistakes
    }

if __name__ == "__main__":
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_advanced_spelling_corrector(
        'spelling_dictionary.json',
        '',  # Directory containing .txt files
        test_ratio=0.4,
        context_aware=True
    )
    
    # Print results
    print("\nAdvanced Spelling Corrector Evaluation Results")
    print("-" * 50)
    print(f"Training set size: {results['train_size']} pairs")
    print(f"Test set size: {results['test_size']} pairs")
    print(f"Corpus files used: {results['corpus_files']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Total test examples: {results['total_examples']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Print model stats
    print("\nModel Statistics:")
    print("-" * 50)
    print(f"Vocabulary size: {results['model_stats']['vocabulary_size']}")
    print(f"Known misspellings: {results['model_stats']['known_misspellings']}")
    print("\nEnabled features:")
    for feature, enabled in results['model_stats']['features_enabled'].items():
        print(f"- {feature}: {'✓' if enabled else '✗'}")
    
    # Print some incorrect examples
    if results['incorrect_examples']:
        print("\nSample of incorrect predictions:")
        print("-" * 50)
        for ex in results['incorrect_examples']:
            print(f"Misspelling: {ex['misspelling']}")
            print(f"Predicted (no context): {ex['predicted_no_context']}")
            print(f"Predicted (with context): {ex['predicted_with_context']}")
            print(f"True word: {ex['true_word']}")
            print(f"Context: {ex['context_used']}")
            print()



'''
Advanced Spelling Corrector Evaluation Results
--------------------------------------------------
Training set size: 24091 pairs
Test set size: 16062 pairs
Corpus files used: 5
Correct predictions: 6099
Total test examples: 16062
Accuracy: 37.97%

Model Statistics:
--------------------------------------------------
Vocabulary size: 22438
Known misspellings: 23061

Enabled features:
- context_aware: ✓
- ngram_model: ✓
- phonetic_matching: ✓
- word_embeddings: ✓

Sample of incorrect predictions:
--------------------------------------------------
Misspelling: eniceation
Predicted (no context): enucleation
Predicted (with context): enucleation
True word: initiation
Context: your further [WORD] a like

Misspelling: scouce
Predicted (no context): six
Predicted (with context): six
True word: saucer
Context: a small [WORD] of milk

Misspelling: garle
Predicted (no context): girl
Predicted (with context): girl
True word: gallery
Context: modern picture [WORD] and every

Misspelling: lifd 1
Predicted (no context): left
Predicted (with context): left
True word: lived
Context: diver who [WORD] in the

Misspelling: traing 1
Predicted (no context): during
Predicted (with context): during
True word: trying
Context: a little [WORD] to do

Misspelling: magnifas
Predicted (no context): magnifas
Predicted (with context): magnifas
True word: magnificent
Context: calm contented [WORD] proud he

Misspelling: contributers
Predicted (no context): contributed
Predicted (with context): contributed
True word: contributors
Context: [WORD]

Misspelling: strawes
Predicted (no context): stories
Predicted (with context): stories
True word: straws
Context: [WORD]

Misspelling: orcatstr
Predicted (no context): orcatstr
Predicted (with context): orcatstr
True word: orchestra
Context: my own [WORD] but shouldn

Misspelling: tund
Predicted (no context): tongue
Predicted (with context): tongue
True word: turned
Context: life he [WORD] out but
'''
