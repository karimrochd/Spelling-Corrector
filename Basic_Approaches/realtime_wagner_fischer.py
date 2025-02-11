import re
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import nltk
from nltk.corpus import words
nltk.download("words")

from wagner_fischer import spell_check, load_dictionary

class SpellingChecker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("600x500")
        self.root.title("Spelling Checker")

        self.text = ScrolledText(self.root, font=("Arial", 14), wrap=tk.WORD)
        self.text.bind("<KeyRelease>", self.check_spelling)
        self.text.bind("<Motion>", self.show_suggestion)  # Detect hover movement
        self.text.pack(expand=True, fill=tk.BOTH)

        self.old_space_count = 0
        self.dictionary = load_dictionary("words.txt")  # Load dictionary from file
        self.misspelled_words = {}  # Store misspelled words with their positions
        self.suggestions = {}  # Store precomputed suggestions

        self.tooltip = tk.Label(self.root, bg="yellow", font=("Arial", 12), relief=tk.SOLID, borderwidth=1)
        self.tooltip.place_forget()  # Hide initially

        self.root.mainloop()

    def check_spelling(self, event):
        """ Check spelling and store the best suggestions immediately. """
        content = self.text.get("1.0", tk.END).strip()
        space_count = content.count(" ")

        if space_count != self.old_space_count:
            self.old_space_count = space_count

            # Clear previous highlights
            for tag in self.text.tag_names():
                self.text.tag_delete(tag)

            self.misspelled_words = {}
            self.suggestions = {}  # Reset suggestions on every check

            # Check each word and store its range (start & end positions)
            words_list = content.split()
            current_pos = 0  # Track current character position

            for word in words_list:
                clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())  # Remove punctuation
                start_pos = content.find(word, current_pos)  # Find word position
                end_pos = start_pos + len(word)

                if clean_word and clean_word not in words.words():
                    self.text.tag_add("misspelled", f"1.{start_pos}", f"1.{end_pos}")
                    self.text.tag_config("misspelled", foreground="red")

                    # Store misspelled word's full range
                    self.misspelled_words[(start_pos, end_pos)] = word

                    # Precompute the best suggestion
                    suggestions = spell_check(clean_word, self.dictionary)
                    if suggestions:
                        self.suggestions[(start_pos, end_pos)] = suggestions[0][0]  # Store best suggestion
                
                current_pos = end_pos  # Update position tracker

    def show_suggestion(self, event):
        """ Show the precomputed spelling suggestion on hover. """
        index = self.text.index(f"@{event.x},{event.y}")
        _, cursor_pos = map(int, index.split("."))

        # Find if the cursor is inside any misspelled word range
        for (start, end), word in self.misspelled_words.items():
            if start <= cursor_pos < end:
                best_suggestion = self.suggestions.get((start, end))
                if best_suggestion:
                    # Display tooltip near the cursor
                    self.tooltip.config(text=f"Did you mean: {best_suggestion}?", fg="black")
                    self.tooltip.place(x=event.x_root - self.root.winfo_rootx(), y=event.y_root - self.root.winfo_rooty() + 20)
                    return

        # Hide tooltip if not hovering over a misspelled word
        self.tooltip.place_forget()


if __name__ == "__main__":
    SpellingChecker()
