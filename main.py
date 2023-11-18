import os

def extract_president_names(file_names):
    '''returns a list of the names of each president from the speech file'''
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        name = parts[1]
        if ord(name[-1]) >= 48 and ord(name[-1]) <= 57:
            name = ''
            for i in range(len(parts[1])-1):
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

