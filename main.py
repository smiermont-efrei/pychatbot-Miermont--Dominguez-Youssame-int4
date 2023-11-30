import math
import os
from collections import *


# Function 1
def extract_president_names(file_names):
    """returns a list of the names of each president from the speech file"""
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        name1 = ''
        for i in range(len(parts[1]) - 4):
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
files_names = os.listdir(speeches_directory)
president_names = extract_president_names(files_names)
print("President Names:", president_names)


# Function 2
def clean_text(path, speech):
    """converts the speeches to lowercase and removes the punctuation"""
    with (open(path, 'r', encoding='utf-8')as f):
        text = f.read()
    text = text.lower()

    punctuation = (',', "'", ";", ':', '!', '?', '-', '_', '(', ')', '/', '.')
    text1 = ''
    for word in text:
        for char in word:
            if char not in punctuation:
                text1 += char
            else:
                text1 += ' '
    text = text1

    outdir = 'cleaned' + "\ " + speech
    if not os.path.exists('./cleaned'):
        os.makedirs('cleaned')

    out = open(outdir, 'w', encoding='utf-8')
    out.write(text)


nb_docs = 0
for speech in os.listdir(speeches_directory):
    path = speeches_directory + '/' + speech
    clean_text(path, speech)
    nb_docs += 1


# Function 3

def count_word_occurrences(text):
    # ''counts the number of times a word is present in a text''
    words = text.split()
    word_counts = Counter(words)
    return word_counts

speeches_directory = 'cleaned'

# Function 4
def calculate_tf(path):
    '''Returns the Term frequency of each word in a text'''
    words = {}        # This will store the words and their frequency
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    speech = content.split()

    for word in speech:         # Puts every word into the words dictionary and initializes them at 0
        if word not in words:
            words[word] = 0

    for word in speech:      # Now we go through the speech again and add 1 to the words if they are in the speech
        if word in words:
            words[word] += 1
    return words


for speech in os.listdir(speeches_directory):
    path = speeches_directory + '/' + speech
    #print(calculate_tf(path))


# Function 5
def in_doc(directory):
    """Returns a dictionary with the number of speeches each word is present in, used in calculate_idf()"""
    Doc = {}        # This will store the number of documents each word is present in
    Speech = []     # Here we have each word of a speech
    nb_docs = 8
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        L = []
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        words = content.split()

        for word in words:
            if word not in L:
                L.append(word)
        Speech.append(L)
    for i in range(nb_docs):      # We are going to put every word into Doc and give to each a value of 0
        for j in range(len(Speech[i])):
            if Speech[i][j] not in Doc:
                Doc[Speech[i][j]] = 0

    for i in range(nb_docs):      # Now we go through every speech list again and add 1 to the words if they are in the speech
        for j in range(len(Speech[i])):
            if Speech[i][j] in Doc:
                Doc[Speech[i][j]] += 1
    return Doc


# Function 6
def calculate_idf(path):
    """Returns a dictionary with the idf score of each word for a given speech"""
    occurence_numbers = in_doc('cleaned')
    nb_docs = 8
    idf_scores = {}
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    words = content.split()
    for word in words:
        idf_scores[word] = 0
    for word, nb in idf_scores.items():
        idf_scores[word] = math.log((nb_docs/occurence_numbers[word]) + 1)
    return idf_scores


for speech in os.listdir(speeches_directory):
    path = speeches_directory + '/' + speech
    #print("IDF scores:", calculate_idf(path))
    #print(in_doc('cleaned'))


# Function 7
def calculate_tf_idf_matrix_all(directory):
    file_names = os.listdir(directory)

    for speech in file_names:
        path = directory + '/' + speech
    idf_scores = calculate_idf(path)

    tf_idf_matrix = []

    for speech in file_names:
        file_path = os.path.join(directory, speech)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        word_occurrences = calculate_tf(file_path)
        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        tf_idf_matrix.append(tf_idf_scores)

    return tf_idf_matrix


#Function 8
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
    return least_important_words
#display_least_important_words(tf_idf_matrix)


#Function 9
def display_highest_tfidf_words(tf_idf_matrix):
    highest_tfidf_words = {}

    for document in tf_idf_matrix:
        # Find the word(s) with the highest TF-IDF score in the document
        max_tfidf_word = max(document, key=document.get)
        max_tfidf_score = document[max_tfidf_word]

        # Add the word(s) and their score to the result dictionary
        highest_tfidf_words[max_tfidf_word] = max_tfidf_score

    print("Word(s) with Highest TF-IDF Score:", highest_tfidf_words)


#display_highest_tfidf_words(tf_idf_matrix)



# Function 10
def calculate_tf_idf_matrix_with_presidents(directory, president):
    """Returns a tf-idf matrix of each speech for a given president in a given directory"""
    idf_scores = []                                 # List of idf value dictionaries
    for speech in os.listdir(directory):
        if president in speech:
            path = directory + '/' + speech
            idf_scores.append(calculate_idf(path))

    tf_idf_scores = {}
    tf_idf_matrix = []

    for speech in os.listdir(directory):
        if president in speech:
            file_path = os.path.join(directory, speech)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            word_occurrences = count_word_occurrences(content)       # Is a dictionary of words and their number of occurence
            #print(word_occurrences)

            for i in range(len(idf_scores)):
                for word, tf in word_occurrences.items():
                    tf_idf_scores[word] = tf * idf_scores[i].get(word, 0)
            tf_idf_matrix.append(tf_idf_scores)

    return tf_idf_matrix


#for president in extract_president_names(os.listdir(speeches_directory)):
#    calculate_tf_idf_matrix_presidents(speeches_directory, president)

#tf_idf_matrix = calculate_tf_idf_matrix_all(speeches_directory)
#print(tf_idf_matrix)



# Function 10.1
def most_repeated_words_by_president(directory, president_name):
    """Returns the most repeated word said by a given president"""
    #president_documents = [doc['tf_idf_scores'] for doc in tf_idf_matrix if doc['president'] == president_name]
    # Donc president_documents est une liste des matrices tf-idf des discours d'un président donné

    president_documents = calculate_tf_idf_matrix_with_presidents(directory, president_name)
    '''
    for speech in os.listdir(directory):    
        if president in speech:
            file_path = os.path.join(directory, speech)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            president_documents.append(content)

    if not president_documents:
        print(f"No documents found for President {president_name}")
        return
'''

    combined_scores = {}
    for document in president_documents:
        for word, tf_idf_score in document.items():
            combined_scores[word] = combined_scores.get(word, 0) + tf_idf_score

    most_repeated_word = max(combined_scores, key=combined_scores.get)
    most_repeated_score = combined_scores[most_repeated_word]

    #print(f"Most Repeated Word by President {president_name}:")
    #print(f"Word: {most_repeated_word}, TF-IDF Score: {most_repeated_score}")


# Example usage:
#for president in extract_president_names(os.listdir(speeches_directory)):
#    tf_idf_matrix = calculate_tf_idf_matrix_with_presidents(speeches_directory, president)
#    most_repeated_words_by_president(speeches_directory, president)


# Function 11
def word_frequence_comparison(directory, target_word):
    """Returns a list of who says the target word more in its speeches."""  #Doesn't work
    i = 0
    L = []
    for speech in os.listdir(speeches_directory):
        path = speeches_directory + '/' + speech
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        #content =
        matrix = calculate_tf(path)
        for word, n in matrix.items():
            w = []
            if word == target_word:
                pres_names = os.listdir(speeches_directory)
                w.append(pres_names[i])
                w.append(matrix[target_word])
                L.append(w)
        i += 1

    numbers = '0123456789'
    for i in range(len(L)):
        parts = L[i][0].split('_')
        name1 = ''
        name = ''
        for j in range(len(parts[1]) - 4):
            name1 += parts[1][j]

        if name1[-1] in numbers:
            name = ''
            for k in range(len(name1) - 1):
                name += str(name1[k])
        else:
            name = name1
        L[i][0] = name
        #print(L[i])
    L2 = []
    m2 = []
    b = False
    for i in range(len(L)-1):
        m2 = []
        for n in L2:
            if L[i][0] == n:
                b = True
        if b is False:
            m2.append(L[i][0])
        if L[i][0] == L[i+1][0]:
            m2.append(L[i][1] + L[i+1][1])
        else:
            m2.append(L[i][1])
        L2.append(m2)
    L2.append(L[-1])
    print(L2)




def calculate_tf_idf_matrix_with_target_word(directory, president, target_word):
    tf_idf_matrix = []
    for speech in os.listdir(directory):
        file_path = os.path.join(directory, speech)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        word_occurrences = count_word_occurrences(content)
        idf_scores = calculate_idf(file_path)
        tf_idf_scores = {}
        for word, tf in word_occurrences.items():
            tf_idf_scores[word] = tf * idf_scores.get(word, 0)

        target_word_count = word_occurrences.get(target_word, 0)
        tf_idf_matrix.append({'president': president, 'tf_idf_scores': tf_idf_scores, 'target_word_count': target_word_count})

    return tf_idf_matrix

#for president in extract_president_names(os.listdir(speeches_directory)):
#word_frequence_comparison(speeches_directory, 'nation')

# Function 11.1
# the parameters in the previous function has been changed to add a parameter to this function to specify the target word
def president_speaking_about_nation_the_most(tf_idf_matrix, president_name):
    combined_counts = {}
    for document in tf_idf_matrix:
        president = document[president_name]
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

tf_idf_matrix_with_nation = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, 'nation')
#for president in extract_president_names(os.listdir(speeches_directory)):
    #print(president_speaking_about_nation_the_most(tf_idf_matrix_with_nation, president))


# Function 12
def calculate_tf_idf_matrix_with_first_mention(directory, president, target_words):

    tf_idf_matrix = []

    for speech in os.listdir(directory):
        file_path = os.path.join(directory, speech)

        idf_scores = calculate_idf(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
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
    

#Function 12.1
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


#Function 13
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


#Main Program

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
