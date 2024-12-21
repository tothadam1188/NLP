import os
import re
import shutil
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import subprocess
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')


class TextGrouperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Szöveg Csoportosító")
        self.file_paths = []
        self.groups_cosine = []
        self.groups_kmeans = []
        self.groups_lda = []

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
        threshold = 0.2

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        documents = [open(file_path, 'r', encoding='utf-8').read() for file_path in self.file_paths]
        cleaned_documents = [self.preprocess_text(doc) for doc in documents]

        # Cosine Similarity-based Grouping
        similarity_matrix = self.get_cosine_similarity_matrix(cleaned_documents)
        initial_groups_cosine = self.group_documents_by_similarity(similarity_matrix, threshold)
        self.groups_cosine = self.expand_groups(initial_groups_cosine, similarity_matrix, threshold)

        # KMeans Clustering-based Grouping
        tfidf_matrix = self.get_tfidf_matrix(cleaned_documents)
        kmeans_groups = self.kmeans_clustering(tfidf_matrix)
        self.groups_kmeans = self.expand_groups_with_kmeans(kmeans_groups)

        # LDA-based Grouping
        lda_matrix = self.get_tfidf_matrix(cleaned_documents)  # LDA also uses the TF-IDF matrix
        lda_labels, topic_probs, topic_keywords = self.lda_topic_modeling(lda_matrix)

        self.groups_lda = lda_labels
        self.save_lda_topic_summary(topic_probs, topic_keywords, documents, lda_labels)

        # Create folders for Cosine Similarity, KMeans, and LDA
        self.save_groups_to_folders(output_folder, "CosineSimilarity", self.groups_cosine, cleaned_documents)
        self.save_groups_to_folders(output_folder, "KMeans", self.groups_kmeans, cleaned_documents)
        self.save_groups_to_folders(output_folder, "LDA", self.groups_lda, cleaned_documents)

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

    def get_tfidf_matrix(self, documents):
        """TF-IDF mátrix létrehozása"""
        self.vectorizer = TfidfVectorizer()
        return self.vectorizer.fit_transform(documents)

    def kmeans_clustering(self, tfidf_matrix, num_clusters=5):
        """KMeans csoportosítás"""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)
        return kmeans.labels_

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

    def expand_groups_with_kmeans(self, kmeans_labels):
        """Csoportok kiterjesztése a KMeans alapján"""
        expanded_groups = {}
        for idx, label in enumerate(kmeans_labels):
            if label not in expanded_groups:
                expanded_groups[label] = []
            expanded_groups[label].append(idx)

        return list(expanded_groups.values())

    def lda_topic_modeling(self, tfidf_matrix, num_topics=5):
        """LDA topic modeling"""
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf_matrix)

        # Get topic probabilities for each document
        topic_probs = lda.transform(tfidf_matrix)

        # Determine the dominant topic for each document (most likely topic)
        lda_labels = [np.argmax(doc_probs) for doc_probs in topic_probs]

        # Get feature names from the TF-IDF vectorizer (you should use the vectorizer used to create tfidf_matrix)
        feature_names = self.vectorizer.get_feature_names_out()

        # Extract the top keywords for each topic
        topic_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_keywords_idx = topic.argsort()[-10:][::-1]  # Get the indices of the top 10 keywords
            topic_keywords.append([feature_names[i] for i in top_keywords_idx])

        return lda_labels, topic_probs, topic_keywords

    def save_lda_topic_summary(self, topic_probs, topic_keywords, documents, lda_labels):
        """Save the LDA topic percentages and keywords to a text file"""
        output_folder = 'Output/LDA'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        lda_summary_path = os.path.join(output_folder, "lda_topic_summary.txt")
        with open(lda_summary_path, 'w', encoding='utf-8') as summary_file:
            for doc_idx, probs in enumerate(topic_probs):
                summary_file.write(f"Document {doc_idx + 1}:\n")
                for topic_idx, prob in enumerate(probs):
                    summary_file.write(f"  Topic {topic_idx + 1}: {prob * 100:.2f}%\n")
                summary_file.write("  Keywords: " + ", ".join(topic_keywords[lda_labels[doc_idx]]) + "\n")
                summary_file.write("\n")

        messagebox.showinfo("LDA Summary", f"LDA topic summary saved to: {lda_summary_path}")

    def save_groups_to_folders(self, output_folder, method, groups, cleaned_documents):
        """Mappákba mentés a csoportok alapján"""
        method_folder = os.path.join(output_folder, method)
        if not os.path.exists(method_folder):
            os.makedirs(method_folder)

        for group_idx, group in enumerate(groups):
            group_text = ' '.join(cleaned_documents[idx] for idx in group)
            group_name = self.get_most_common_words(group_text)
            group_folder = os.path.join(method_folder, f"Group_{group_idx + 1}_{group_name}")

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
            method_folders = os.listdir(output_folder)
            for method_folder in method_folders:
                self.file_list.insert(tk.END, method_folder)

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
