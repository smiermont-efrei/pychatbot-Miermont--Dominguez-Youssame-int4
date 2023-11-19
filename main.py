import os


def extract_president_names(file_names):
    """returns a list of the names of each president from the speech file"""
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        name = parts[1]
        if 48 <= ord(name[-1]) <= 57:
            name = ''
            for i in range(len(parts[1]) - 1):
                name += str(parts[1][i])
        president_names.add(name)

    return list(president_names)


directory = "./speeches-20231110"
files_names = os.listdir(directory)
president_names = extract_president_names(files_names)
print("President Names:", president_names)

import os
import string


def clean_text(file_path, output_directory):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content_lower = content.lower()

    translator = str.maketrans('', '', string.punctuation)
    content_no_punct = content_lower.translate(translator)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cleaned_file_path = os.path.join(output_directory, os.path.basename(file_path))

    with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
        cleaned_file.write(content_no_punct)


speeches_directory = "./speeches-20231110"
cleaned_directory = "./cleaned"

speech_files = [file for file in os.listdir(speeches_directory) if file.endswith('.txt')]

for speech_file in speech_files:
    speech_file_path = os.path.join(speeches_directory, speech_file)
    clean_text(speech_file_path, cleaned_directory)


def count_word_occurrences(text):
    words = text.split()

    word_counts = Counter(words)

    return word_counts


text_example = "This is an example text. Example text is here."
word_occurrences = count_word_occurrences(text_example)
print("Word Occurrences:", word_occurrences)

import os
import math


def calculate_idf(directory):
    """Get the list of file names in the directory"""
    file_names = [file for file in os.listdir(directory) if file.endswith('.txt')]

    num_documents = len(file_names)

    idf_scores = {}

    for file_name in file_names:
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        words = content.split()

        unique_words = set(words)

        for word in unique_words:
            idf_scores[word] = idf_scores.get(word, 0) + 1

    for word, count in idf_scores.items():
        idf_scores[word] = math.log(num_documents / count)

    return idf_scores


speeches_directory = "./cleaned"
idf_scores = calculate_idf(speeches_directory)
print("IDF Scores:", idf_scores)

import os
import math
from collections import Counter

def calculate_tf_idf_matrix(directory):
    file_names = [file for file in os.listdir(directory) if file.endswith('.txt')]

    idf_scores = calculate_idf(directory)

    tf_idf_matrix = []

    for file_name in file_names:
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        word_occurrences = count_word_occurrences(content)

        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        tf_idf_matrix.append(tf_idf_scores)

    return tf_idf_matrix

speeches_directory = "./cleaned"
tf_idf_matrix = calculate_tf_idf_matrix(speeches_directory)
print("TF-IDF Matrix:", tf_idf_matrix)

