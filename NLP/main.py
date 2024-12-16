import os
import re
import shutil
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

nltk.download('punkt')
nltk.download('stopwords')

class TextGrouperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Szöveg Csoportosító")
        self.file_paths = []
        self.groups = []

        self.center_window(800, 600)
        self.create_widgets()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text="Fájlok feltöltése", command=self.upload_files)
        self.upload_button.pack(pady=10)

        self.group_button = tk.Button(self.root, text="Fájlok csoportosítása", command=self.group_files)
        self.group_button.pack(pady=10)

        self.back_button = tk.Button(self.root, text="Vissza", command=self.show_groups)
        self.back_button.pack(pady=10)
        self.back_button.pack_forget()

        self.file_list = tk.Listbox(self.root)
        self.file_list.pack(pady=10, fill=tk.BOTH, expand=True)

    def upload_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Szöveg fájlok", "*.txt")])
        for file_path in file_paths:
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
                self.file_list.insert(tk.END, os.path.basename(file_path))

    def group_files(self):
        if not self.file_paths:
            messagebox.showwarning("Nincs fájl", "Nincs fájl a csoportosításhoz.")
            return

        output_folder = 'Output'
        threshold = 0.3

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        documents = [open(file_path, 'r', encoding='utf-8').read() for file_path in self.file_paths]
        cleaned_documents = [self.preprocess_text(doc) for doc in documents]
        similarity_matrix = self.get_cosine_similarity_matrix(cleaned_documents)

        initial_groups = self.group_documents_by_similarity(similarity_matrix, threshold)
        self.groups = self.expand_groups(initial_groups, similarity_matrix, threshold)

        for group in self.groups:
            group_text = ' '.join(cleaned_documents[idx] for idx in group)
            group_name = self.get_most_common_words(group_text)
            group_folder = os.path.join(output_folder, group_name)

            if not os.path.exists(group_folder):
                os.makedirs(group_folder)

            for idx in group:
                original_file_path = self.file_paths[idx]
                with open(original_file_path, 'r', encoding='utf-8') as file:
                    document_text = file.read()

                cleaned_text = self.preprocess_text(document_text)
                important_words = self.get_most_common_words(cleaned_text)
                new_file_name = f"{important_words}.txt"
                new_file_path = os.path.join(group_folder, new_file_name)

                shutil.copyfile(original_file_path, new_file_path)

        self.show_groups()

    def preprocess_text(self, text):
        """Szöveg előfeldolgozása"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        cleaned_text = [word for word in words if word not in stop_words]

        return ' '.join(cleaned_text)

    def get_cosine_similarity_matrix(self, documents):
        """Cosine similarity számítása"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return cosine_similarity(tfidf_matrix)

    def group_documents_by_similarity(self, similarity_matrix, threshold):
        """Csoportosítás a hasonlóság alapján"""
        groups = []
        for idx, similarities in enumerate(similarity_matrix):
            assigned = False
            for group in groups:
                if any(similarity_matrix[idx][doc_idx] > threshold for doc_idx in group):
                    group.append(idx)
                    assigned = True
                    break
            if not assigned:
                groups.append([idx])
        return groups

    def expand_groups(self, groups, similarity_matrix, threshold):
        """Csoportok kiterjesztése a hasonlóság alapján"""
        expanded_groups = []
        for group in groups:
            expanded_group = set(group)
            for idx in range(similarity_matrix.shape[0]):
                if idx not in expanded_group:
                    if sum(similarity_matrix[idx][doc_idx] > threshold for doc_idx in group) > len(group) // 2:
                        expanded_group.add(idx)
            expanded_groups.append(list(expanded_group))
        return expanded_groups

    def get_most_common_words(self, text, num_words=3):
        """Leggyakoribb szavak kinyerése"""
        words = word_tokenize(text)
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(num_words)
        return ' '.join(word for word, count in most_common_words)

    def show_groups(self):
        self.file_list.delete(0, tk.END)

        output_folder = 'Output'
        if os.path.exists(output_folder):
            group_folders = os.listdir(output_folder)
            for group_folder in group_folders:
                self.file_list.insert(tk.END, group_folder)

        self.back_button.pack_forget()
        self.upload_button.pack(pady=10)
        self.group_button.pack(pady=10)

        self.file_list.bind('<Double-1>', self.show_group_contents)

    def show_group_contents(self, event):
        selected_group = self.file_list.get(self.file_list.curselection())
        output_folder = 'Output'
        group_folder = os.path.join(output_folder, selected_group)

        if os.path.isdir(group_folder):
            self.file_list.delete(0, tk.END)
            for file_name in os.listdir(group_folder):
                self.file_list.insert(tk.END, file_name)

            self.upload_button.pack_forget()
            self.group_button.pack_forget()
            self.back_button.pack(pady=10)

            self.file_list.bind('<Double-1>', lambda e: self.open_file(
                os.path.join(group_folder, self.file_list.get(self.file_list.curselection()))))

    def open_file(self, file_path):
        if os.path.isfile(file_path):
            subprocess.run(['notepad', file_path])  # Notepad megnyitás

if __name__ == "__main__":
    root = tk.Tk()
    app = TextGrouperApp(root)
    root.mainloop()
