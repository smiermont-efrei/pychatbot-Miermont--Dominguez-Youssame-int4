from os import *
from string import *
from math import *
from collections import *

#Fonction 1
def extract_president_names(file_names):
    """returns a list of the names of each president from the speech file"""
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        name1 = ''
        for i in range(len(parts[1])-4):
            name1 += parts[1][i]

        if 48 <= ord(name1[-1]) <= 57:
            name = ''
            for i in range(len(name1) - 1):
                name += str(name1[i])
        else:
            name = name1
        president_names.add(name)

    return list(president_names)

speeches_directory = "./speeches-20231110"
files_names = listdir(speeches_directory)
president_names = extract_president_names(files_names)
#print("President Names:", president_names)

'''
#fonction 2 ///////////////////////////////////////////////////////////
def clean_text(file_path, output_directory):
    """converts the speeches to lowercase and removes the punctuation"""
    with open(file_path, 'r') as file:
        content = file.read()

        content_lower = content.lower()

    translator = str.maketrans('', '', str.punctuation)
    content_no_punct = content_lower.translate(translator)

    if not path.exists(output_directory):
        makedirs(output_directory)

    cleaned_file_path = path.join(output_directory, path.basename(file_path))

    with open(cleaned_file_path, 'w') as cleaned_file:
        cleaned_file.write(content_no_punct)


cleaned_directory = "/cleaned"
'''
# We fill speech_files to make it a list of all speech files
speech_files = []
for file in listdir(speeches_directory):
    if file.endswith('.txt'):
        speech_files.append(file)
'''

# We make a path for each speech to use it as the input directory, and cleaned_directory is the path the 'cleaned' files will be put into
for speech_file in speech_files:
    speech_file_path = path.join(speeches_directory, speech_file)
    clean_text(speech_file_path, cleaned_directory)

# on a des problèmes ici //////////////////////////////////////////
'''

# Fonction 3
def count_word_occurrences(text):
    words = text.split()
    word_counts = Counter(words)
    return word_counts


#text_example = "this is an example text example text is here"
#word_occurrences = count_word_occurrences(text_example)
#print("Word Occurrences:", word_occurrences)



# Fonction 4
def calculate_idf(directory):
    """Get the list of file names in the directory"""
    num_documents = len(speech_files)
    idf_scores = {}
    for file_name in speech_files:
        file_path = path.join(directory, file_name)

        with open(file_path, 'r') as file:
            content = file.read()
        words = content.split()

        for word in words:
            idf_scores[word] = idf_scores.get(word, 0) + 1

    for word, count in idf_scores.items():
        idf_scores[word] = math.log(num_documents / count)

    return idf_scores


speeches_directory = "./cleaned"
idf_scores = calculate_idf(speeches_directory)
print("IDF Scores:", idf_scores)


'''
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
'''
