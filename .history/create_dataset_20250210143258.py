from collections import defaultdict
import json

def load_and_process_files():
   files = ['aspell.dat', 'holbrook-missp.dat', 'missp.dat', 'wikipedia.dat']
   spelling_dict = defaultdict(set)
   
   for filename in files:
       try:
           with open(filename, 'r', encoding='utf-8') as f:
               current_word = None
               for line in f:
                   line = line.strip()
                   if not line:
                       continue
                   if line.startswith('$'):
                       current_word = line[1:].lower().replace('_', ' ')
                   elif current_word:
                       misspelling = line.lower().replace('_', ' ')
                       spelling_dict[current_word].add(misspelling)
       except Exception as e:
           print(f"Error loading {filename}: {e}")

   # Convert sets to lists for JSON serialization
   json_dict = {word: list(misspellings) for word, misspellings in spelling_dict.items()}
   
   with open('spelling_dictionary.json', 'w', encoding='utf-8') as f:
       json.dump(json_dict, f, indent=2)

   print(f"Created dictionary with {len(json_dict)} words")

if __name__ == "__main__":
   load_and_process_files()