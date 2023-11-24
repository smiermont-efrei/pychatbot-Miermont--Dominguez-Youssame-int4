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


#fonction 2
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

#We fill speech_files to make it a list of all speech files
speech_files = []
for file in listdir(speeches_directory):
    if file.endswith('.txt'):
        speech_files.append(file)


# We make a path for each speech to use it as the input directory, and cleaned_directory is the path the 'cleaned' files will be put into
for speech_file in speech_files:
    speech_file_path = path.join(speeches_directory, speech_file)
    clean_text(speech_file_path, cleaned_directory)

# on a des problèmes ici //////////////////////////////////////////


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


#Fonction 5 

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


#Fonction 6
def display_least_important_words(tf_idf_matrix):
    least_important_words = set()

    # Iterate over each word in the TF-IDF matrix
    for document in tf_idf_matrix:
        for word, tf_idf_score in document.items():
            # Check if the TF-IDF score is 0 in all documents
            if tf_idf_score == 0:
                least_important_words.add(word)

    # Display the list of least important words
    print("Least Important Words:", least_important_words)

speeches_directory = "./cleaned"
tf_idf_matrix = calculate_tf_idf_matrix(speeches_directory)
display_least_important_words(tf_idf_matrix)

#Fonction 7
def display_highest_tfidf_words(tf_idf_matrix):
    highest_tfidf_words = {}

    for document in tf_idf_matrix:
        # Find the word(s) with the highest TF-IDF score in the document
        max_tfidf_word = max(document, key=document.get)
        max_tfidf_score = document[max_tfidf_word]

        # Add the word(s) and their score to the result dictionary
        highest_tfidf_words[max_tfidf_word] = max_tfidf_score

    print("Word(s) with Highest TF-IDF Score:", highest_tfidf_words)

speeches_directory = "./cleaned"
tf_idf_matrix = calculate_tf_idf_matrix(speeches_directory)
display_highest_tfidf_words(tf_idf_matrix)

# Fonction 8

def calculate_tf_idf_matrix_with_presidents(directory, president_names):

    tf_idf_matrix = []

    for file_name, president in zip(file_names, president_names):
        file_path = os.path.join(directory, file_name)

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        word_occurrences = count_word_occurrences(content)

        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        tf_idf_matrix.append({'president': president, 'tf_idf_scores': tf_idf_scores})

    return tf_idf_matrix

# Fonction 8.1
def most_repeated_words_by_president(tf_idf_matrix, president_name):
    president_documents = [doc['tf_idf_scores'] for doc in tf_idf_matrix if doc['president'] == president_name]

    if not president_documents:
        print(f"No documents found for President {president_name}")
        return

    combined_scores = {}
    for document in president_documents:
        for word, tf_idf_score in document.items():
            combined_scores[word] = combined_scores.get(word, 0) + tf_idf_score

    most_repeated_word = max(combined_scores, key=combined_scores.get)
    most_repeated_score = combined_scores[most_repeated_word]

    print(f"Most Repeated Word(s) by President {president_name}:")
    print(f"Word: {most_repeated_word}, TF-IDF Score: {most_repeated_score}")

# Example usage:
speeches_directory = "./cleaned"
president_names = ['Chirac', 'Giscard d\'Estaing', 'Hollande', 'Mitterrand', 'Macron', 'Sarkozy']
tf_idf_matrix = calculate_tf_idf_matrix_with_presidents(speeches_directory, president_names)
most_repeated_words_by_president(tf_idf_matrix, 'Chirac')

# Fonction 9

def calculate_tf_idf_matrix_with_target_word(directory, president_names, target_word):

    tf_idf_matrix = []

    for file_name, president in zip(file_names, president_names):
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r') as file:
            content = file.read()

        word_occurrences = count_word_occurrences(content)

        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        target_word_count = word_occurrences.get(target_word, 0)
        tf_idf_matrix.append({'president': president, 'tf_idf_scores': tf_idf_scores, 'target_word_count': target_word_count})

    return tf_idf_matrix

# Fonction 9.1
# the parameters in the prevoius function has been changed to add a parameter to this function to specify the target word
def president_speaking_about_nation_the_most(tf_idf_matrix):
    combined_counts = {}
    for document in tf_idf_matrix:
        president = document['president']
        target_word_count = document['target_word_count']
        combined_counts[president] = combined_counts.get(president, 0) + target_word_count

    max_count = max(combined_counts.values())
    presidents_with_max_count = [president for president, count in combined_counts.items() if count == max_count]

    print(f"President(s) who spoke about 'Nation' the most:")
    for president in presidents_with_max_count:
        print(f"{president}: {max_count} occurrences")

# Example usage:
speeches_directory = "./cleaned"
president_names = ['Chirac', 'Giscard d\'Estaing', 'Hollande','Mitterrand', 'Macron', 'Sarkozy']
target_word = 'Nation'
tf_idf_matrix_with_nation = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, target_word)
president_speaking_about_nation_the_most(tf_idf_matrix_with_nation)

# Fonction 10
def calculate_tf_idf_matrix_with_first_mention(directory, president_names, target_words):

    tf_idf_matrix = []

    for file_name, president in zip(file_names, president_names):
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r') as file:
            content = file.read()

        word_occurrences = count_word_occurrences(content)

        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        first_mentions = {}
        for target_word in target_words:
            if target_word in word_occurrences and target_word not in first_mentions:
                first_mentions[target_word] = president

        tf_idf_matrix.append({'president': president, 'tf_idf_scores': tf_idf_scores, 'first_mentions': first_mentions})

    return tf_idf_matrix
#Fonction 10.1
def first_president_to_mention_climate_or_ecology(tf_idf_matrix, target_words):
    first_mentions = {}

    for document in tf_idf_matrix:
        president = document['president']
        mentions = document['first_mentions']

        for target_word in target_words:
            if target_word in mentions:
                # Record the first mention
                if target_word not in first_mentions:
                    first_mentions[target_word] = (president, document['tf_idf_scores'][target_word])

    for target_word, (president, tf_idf_score) in first_mentions.items():
        print(f"The first president to talk about '{target_word}': {president} (TF-IDF Score: {tf_idf_score})")

# Example usage:
speeches_directory = "./cleaned"
president_names = ['Chirac', 'Giscard d\'Estaing', 'Hollande' ,'Mitterrand', 'Macron', 'Sarkozy']
target_words = ['climate', 'ecology']
tf_idf_matrix_with_first_mention = calculate_tf_idf_matrix_with_first_mention(speeches_directory, president_names, target_words)
first_president_to_mention_climate_or_ecology(tf_idf_matrix_with_first_mention, target_words)

#Fonction 11
def words_mentioned_by_all_presidents(tf_idf_matrix, unimportant_words):
    common_words = set(tf_idf_matrix[0]['tf_idf_scores'].keys())

    for document in tf_idf_matrix[1:]:
        current_words = set(word for word, score in document['tf_idf_scores'].items() if score > 0)
        common_words.intersection_update(current_words)
        common_words.difference_update(unimportant_words)

    return common_words


speeches_directory = "./cleaned"
president_names = ['Chirac', 'Giscard d\'Estaing', 'Hoolande', 'Mitterrand', 'Macron', 'Sarkozy']
unimportant_words = {'the', 'and', 'to', 'of', 'in', 'a', 'is', 'it', 'that', 'with', 'for', 'on', 'this', 'as', 'by', 'an'}

# Calculates the  TF-IDF matrix with unimportant words removed
tf_idf_matrix_without_unimportant = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, 'Nation')
common_words = words_mentioned_by_all_presidents(tf_idf_matrix_without_unimportant, unimportant_words)

print("Words Mentioned by All Presidents (Except Unimportant Words):", common_words)


#Main Programe

def main():
    speeches_directory = "./cleaned"
    president_names = ['Chirac', 'Giscard d\'Estaing','Hollande', 'Mitterrand', 'Macron', 'Sarkozy']
    unimportant_words = {'the', 'and', 'to', 'of', 'in', 'a', 'is', 'it', 'that', 'with', 'for', 'on', 'this', 'as', 'by', 'an'}

    while True:
        print("\nMenu:")
        print("1. Display List of Least Important Words")
        print("2. Display Word(s) with Highest TF-IDF Score")
        print("3. Indicate the Most Repeated Word(s) by President Chirac")
        print("4. Indicate the President(s) who Spoke about 'Nation' the Most")
        print("5. Identify the First President to Talk about 'Climate' and/or 'Ecology'")
        print("6. Identify Words Mentioned by All Presidents (Except Unimportant Words)")
        print("0. Exit")

        choice = input("Enter your choice (0-6): ")

        if choice == '1':
            tf_idf_matrix = calculate_tf_idf_matrix(speeches_directory)
            display_least_important_words(tf_idf_matrix)
        elif choice == '2':
            tf_idf_matrix = calculate_tf_idf_matrix(speeches_directory)
            display_highest_tfidf_words(tf_idf_matrix)
        elif choice == '3':
            tf_idf_matrix = calculate_tf_idf_matrix_with_presidents(speeches_directory, president_names)
            most_repeated_words_by_president(tf_idf_matrix, 'Chirac')
        elif choice == '4':
            tf_idf_matrix_with_nation = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, 'Nation')
            president_speaking_about_nation_the_most(tf_idf_matrix_with_nation)
        elif choice == '5':
            target_words = ['climate', 'ecology']
            tf_idf_matrix_with_first_mention = calculate_tf_idf_matrix_with_first_mention(speeches_directory, president_names, target_words)
            first_president_to_mention_climate_or_ecology(tf_idf_matrix_with_first_mention, target_words)
        elif choice == '6':
            tf_idf_matrix_without_unimportant = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, 'Nation')
            common_words = words_mentioned_by_all_presidents(tf_idf_matrix_without_unimportant, unimportant_words)
            print("Words Mentioned by All Presidents (Except Unimportant Words):", common_words)
        elif choice == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 0 and 6.")

if __name__ == "__main__":
    main()



