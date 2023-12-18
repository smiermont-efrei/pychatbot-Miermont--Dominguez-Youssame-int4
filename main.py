import math
import os
from collections import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#print("President Names:", president_names)


# Function 2
def clean_text(path, speech):
    """converts the speeches to lowercase and removes the punctuation"""
    with (open(path, 'r', encoding='utf-8')as f):
        text = f.read()
    text = text.lower()

    punctuation = (',', "'", '"', ";", ':', '!', '?', '-', '_', '(', ')', '/', '.')
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
    return

'''
for speech in os.listdir(speeches_directory):
    path = speeches_directory + '/' + speech
    print(clean_text(path, speech))
'''

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



# Function 5
def in_doc(directory):
    """Returns a dictionary with the number of speeches each word is present in, used in calculate_idf()"""
    Doc = {}        # This will store the number of documents each word is present in
    Speech = []     # Here we will have each word of a speech
    nb_docs = 8
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        L = []
        with open(path, 'r', encoding='utf-8') as file:
            words = file.read().split()

        words = list(set(words))
        Speech.append(words)

    for i in range(len(Speech)):      # We are going to put every word into Doc and give to each a value of 0
        for j in range(len(Speech[i])):
            if Speech[i][j] not in Doc:
                Doc[Speech[i][j]] = 0

    for i in range(nb_docs):      # Now we go through every speech list again and add 1 to the words if they are in the speech
        for j in range(len(Speech[i])):
            if Speech[i][j] in Doc:
                Doc[Speech[i][j]] += 1
    return Doc                    # Final result : A dictionary with the number of occurences of each word


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
        idf_scores[word] = 1    #math.log((nb_docs/occurence_numbers[word]) + 1)

    return idf_scores



# TF fix

# Il va nous falloir un dictionnaire tf et un dictionnaire idf tel que [tf[de]: [1,2,3,4, les tf dans chaque document],...] et [idf[de]: [1,2,3,4, les idf dans chaque document],...] ensuite on fait

# tf_idf = {}
# for word in [dictionnaire de tous les mots]:
#    tf_idf[word] = tf[word]*idf[word]          pour un tf_idf général


# Function 7
def calculate_tf_idf_matrix_all(directory):
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        idf_scores = calculate_idf(path)

    tf_idf_matrix = []

    for speech in os.listdir(directory):
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


#display_highest_tfidf_words(calculate_tf_idf_matrix_all(speeches_directory))



# Function 10
def calculate_tf_idf_matrix_with_presidents(directory, president):
    """Returns a tf-idf matrix of each speech for a given president in a given directory"""
    idf_scores = []                                 # List of idf value dictionaries
    for speech in os.listdir(directory):
        if president in speech:
            path = directory + '/' + speech
            idf_scores.append(calculate_idf(path))
    #for i in idf_scores:
        #print(i)
    tf_idf_scores = {}
    tf_idf_matrix = []

    for speech in os.listdir(directory):
        if president in speech:
            file_path = os.path.join(directory, speech)

            word_occurrences = calculate_tf(file_path)
            #print(word_occurrences)

            for i in range(len(idf_scores)):
                for word, tf in word_occurrences.items():
                    tf_idf_scores[word] = tf * idf_scores[i].get(word, 0)
                    tf_idf_scores[word] = '{:.4f}'.format(tf_idf_scores[word])
            tf_idf_matrix.append(tf_idf_scores)

#    pathlist = []

#    for speech in os.listdir(directory):
#            if president in speech:
#                file_path = os.path.join(directory, speech)
#                pathlist.append(file_path)

#    for fpath in pathlist:
#        word_occurrences = calculate_tf(fpath)
        #print(word_occurrences)

#        for i in range(len(idf_scores)):
#            for word, tf in word_occurrences.items():
#                tf_idf_scores[word] = tf * idf_scores[i].get(word, 0)
#                tf_idf_scores[word] = '{:.4f}'.format(tf_idf_scores[word])
#                tf_idf_matrix.append(tf_idf_scores)

    return tf_idf_matrix        # returns a matrix containing the tf-idf dictionaries of every speech of the given president


#for president in extract_president_names(os.listdir(speeches_directory)):
    #print(calculate_tf_idf_matrix_with_presidents(speeches_directory, 'president'))

#tf_idf_matrix = calculate_tf_idf_matrix_all(speeches_directory)
#print(tf_idf_matrix)



# Function 10.1
def most_repeated_words_by_president(directory, president):
    """Returns the most repeated word said by a given president"""
    president_real = False

    for speech in os.listdir(directory):
        if president in speech:
            president_real = True
            #file_path = directory + '/' + speech
            #with open(file_path, 'r', encoding='utf-8') as file:
                #content = file.read()
            #president_documents.append(content)
    president_documents = calculate_tf_idf_matrix_with_presidents(directory, president)
    for d in president_documents:
        print(d)
    if not president_real:
        print(f"No documents found for President {president}")
        return

    combined_scores = {}            # New dictionary. This is where we are going to merge the tf-idf dictionaries in the given list into a single dictionary
    if len(president_documents) == 2:
        wordlist = {}
        for i in range (len(president_documents)):
            for word in president_documents[i]:
                wordlist.add(word)
        print(wordlist)

# checkpoint

        for sp_word, sp_score in president_documents[0].items():
            combined_scores[sp_word] = combined_scores.get(sp_word, 0) + sp_score

    most_repeated_word = max(combined_scores, key=combined_scores.get)
    most_repeated_score = combined_scores[most_repeated_word]

    print(f"Most Repeated Word by President {president}:")
    print(f"Word: {most_repeated_word}, TF-IDF Score: {most_repeated_score}")



# Example usage:
#for president in extract_president_names(os.listdir(speeches_directory)):
#    tf_idf_matrix = calculate_tf_idf_matrix_with_presidents(speeches_directory, president)
#most_repeated_words_by_president(speeches_directory, 'Chirac')


# Function 11
def word_frequence_comparison(directory, target_word):
    """Returns a list of who says the target word more in its speeches."""  #Doesn't work
    i = 0
    L = []
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        #content =
        matrix = calculate_tf(path)
        for word, n in matrix.items():
            w = []
            if word == target_word:
                pres_names = os.listdir(directory)
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
#tf_idf_matrix_without_unimportant = calculate_tf_idf_matrix_with_target_word(speeches_directory, president_names, 'Nation')
#common_words = words_mentioned_by_all_presidents(tf_idf_matrix_without_unimportant, unimportant_words)

#print("Words Mentioned by All Presidents (Except Unimportant Words):", common_words)


# Main Program

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
            tf_idf_matrix = calculate_tf_idf_matrix_with_presidents(speeches_directory)
            display_least_important_words(tf_idf_matrix)
        elif choice == '2':
            tf_idf_matrix = calculate_tf_idf_matrix_all(speeches_directory)
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

#if __name__ == "__main__":
#    main()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function 1
def question_token(question):
    """Takes a question as a string in parameter and returns a cleaned version of it as a list of words (lowercase and with no punctuation), thus tokenizing the question."""
    question = str(question)
    text = question     # I put the question in lowercase manually after taking away the punctuation

    # We take away the punctuation
    punctuation = (',', "'", ";", ':', '!', '?', '-', '_', '(', ')', '/', '.')
    text1 = ''
    for word in text:
        for char in word:
            if char not in punctuation:
                text1 += char
            else:
                text1 += ' '
    text = text1

    # We turn letters into lowercase when needed
    text1 = ''
    for word in text:
        for letter in word:
            if 64 < ord(letter) and ord(letter) < 91 :
                text1 += chr(ord(letter) + 32)
            else:
                text1 += letter
    text = text1
    lquestion = text.split()
    return lquestion

# Testing the function :
#print(question_token("Comment s'appelle-le président ?"))


# Function 2
def allcorpus_text(directory):
    """Function that returns all the speeches as a single string."""
    corpus = ''
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        with open(path, 'r', encoding='utf-8') as file:
            corpus = corpus + file.read()
    return corpus

#print(allcorpus(speeches_directory))

# Function 2.1
def allcorpus_list(directory):
    """Function that returns all the speeches as a list of strings."""
    corpus = []
    for speech in os.listdir(directory):
        path = directory + '/' + speech
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        text = text.split()
        noenter = ''
        for word in text:
            noenter = noenter + word + ' '

        corpus.append(noenter)
    return corpus

#for speech in allcorpus_list((speeches_directory)):
#    print(speech)

# Function 3
def tokens_in_corpus(token_list, directory):
    """Takes a question as a list of words, tokens, and returns a list of the tokens that are in the corpus (?)"""
    approved_tokens = {}
    text = allcorpus_text(directory).split()
    for i in range(len(token_list)):       # Loop for all the words in the question
        if token_list[i] in text:
            approved_tokens.add(token_list[i])     # Puts the word from the question into the set if the word is in the corpus

    approved_tokens = list(approved_tokens)
    return approved_tokens


# Maybe we will need a function to get the tf-idf score of the tokens in each document


# Function 4
def tf_idf_matrix_fix_separated_by_document(directory):
    """Returns a tf-idf matrix that depends on each independent document."""
    final = []
    for speech in os.listdir(directory):        # Loop to repeat the same process with all the speeches
        path = directory + '/' + speech
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read().split()

        tf = calculate_tf(path)
        idf = calculate_idf(path)
        doc_words = set()
        for word in text:                       # Creating a set with all the words in the document to avoid duplicates in the next step
            if word not in doc_words:
                doc_words.add(word)

        row = []
        for word in doc_words:                  # Putting every word in the text with its corresponding tf-idf into a list
            row.append([word, '{:.5}'.format(tf[word]*idf[word])])

        final.append(row)                       # Fills progressively the final tf-idf matrix and returns it at the end
    return final

#for i in tf_idf_matrix_fix_separated_by_document(speeches_directory):
#    print(i)

# Function 4
def tf_question(question):
    """Returns a matrix associating each word to its tf value in the question"""
    question = question_token(question)
    q_set = set(question)       # Makes a set of the words to avoid duplicates

    tf = []
    for word in q_set:       # First a loop to give a tf score to each word
        count = 0
        for i in range(len(question)):         # Then another loop to count each occurrence of the word in the question
            if question[i] == word:
                count += 1
        tf.append([word, count])

    return tf

#print(tf_question("Le président donné est-il le plus vieux président ?"))


# All part 2 is functional this far

# Funtion 5
def tf(directory):
    """Returns a matrix where each speech has a matrix with each word and their number of occurences in the speech, the TF values"""
    corpus_list = allcorpus_list(directory)         # The text in the form of a list of words
    final = []

    for speech in corpus_list:      # A loop for each speech
        words = speech.split()
        speech_voc = []
        c_voc = list(set(words))
        occu = []

        # Creating an occurence matrix
        for word in c_voc:
            occu.append([word, 0])

        # Filling the occurence values for this speech
        for word in c_voc:
            if word in words:
                for i in range(len(occu)):
                    if word == occu[i][0]:
                        occu[i][1] += 1
        final.append(occu)

    return final


#p = occurence(speeches_directory)
#for i in range(len(p)):
#    print(i)

# Function 5.1
def occurence_in_docs(directory):
    final = []
    speeches = allcorpus_list(directory)
    speeches_list = []
    count = 0
    # Preparing the list of words for each text
    for speech in speeches:
        speeches_list.append(speech.split())

    # Making a matrix of all words
    words = allcorpus_text(directory).split()
    word_list = set()
    all_words = []
    for i in range(len(words)):
        if words[i] not in word_list:
            all_words.append([words[i],0])
            word_list.add(words[i])

    # Changing all_words to have the number of documents each word is present in
    for target in range(len(all_words)):
        for speech in speeches_list:
            if all_words[target][0] in speech:
                all_words[target][1] += 1


    return all_words


#o = occurence_in_docs(speeches_directory)
#print(o)
#for i in o:
#    print(i)

# Function 5.2
def idf_allcorpus(directory):
    """Returns a matrix of the idf that depends on each document"""
    corpus = allcorpus_text(directory).split()
    idf = []
    c_set = list(set(corpus))                     # list of every word in all the texts
    separate_corpus = allcorpus_list(directory)
    occu = occurence_in_docs(directory)

    # Loop to get each word in the text a tf-idf score
    for word in c_set:
        # Repeats the process for all speeches
        for i in range(len(occu)):
            if word == occu[i][0]:
                score = math.log((8/(occu[i][1])) + 1)
                idf.append([word, score])
    return idf


#idf_allcorpus(speeches_directory)

def tf_idf_allcorpus(directory):
    """Returns a tf-idf matrix"""
    idf = idf_allcorpus(directory)
    tf_m = tf(directory)
    final = []

    for i in range(len(tf_m)):
        row = []
        for j in range(len(tf_m[i])):
            for word in idf:
                if tf_m[i][j][0] == word[0]:
                    score = tf_m[i][j][1] * word[1]
                    row.append([word[0], score])

        final.append(row)
    return final

#tf_idf_allcorpus(speeches_directory)



# Function to calculate TF-IDF vector for the question
def calculate_question_tfidf(directory, question):
    """Returns the TF-IDF vector of the question"""
    corpus_words = allcorpus_text(directory).split()

    corpus_tfidf = tf_idf_allcorpus(directory)
    # Tokenize the question
    question_words = question_token(question)

    # Initialize TF-IDF vector for the question
    question_tfidf = {}

    # Calculate TF for each word in the question
    for word in question_words:
        tf = question_words.count(word) / len(question_words)
        # Set TF to 0 for words not in the corpus
        if word in corpus_words:
            # Use precalculated IDF from the corpus
            idf = corpus_tfidf[word]['idf']
            question_tfidf[word] = tf * idf
        else:
            question_tfidf[word] = 0

    return question_tfidf


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    """Returns the cosine similarity between two vectors"""
    dot_product = sum(vector1[word] * vector2[word] for word in set(vector1) & set(vector2))
    norm_vector1 = math.sqrt(sum(vector1[word] ** 2 for word in vector1))
    norm_vector2 = math.sqrt(sum(vector2[word] ** 2 for word in vector2))

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0

    return dot_product / (norm_vector1 * norm_vector2)


# Function to find the most relevant document
def find_most_relevant_document(question_tfidf, corpus_tfidf_matrix, file_names):
    """Returns the most relevant document """
    similarities = {}

    # Calculate cosine similarity with each document in the corpus
    for file_name, document_tfidf in corpus_tfidf_matrix.items():
        similarities[file_name] = cosine_similarity(question_tfidf, document_tfidf)

    # Identify the document with the highest similarity
    most_relevant_document = max(similarities, key=similarities.get)

    # Return the most relevant document name
    return most_relevant_document


# Fonction 6
def generate_response(question, most_relevant_document):
    """Returns a response"""
    # Locate the first occurrence of the word with the highest TF-IDF score in the relevant document
    # Return the sentence containing the identified word as the answer
    # Implement your logic here to locate the word and generate the response

    # Step 1: Calculate TF-IDF vector for the question
    question_tfidf = calculate_question_tfidf(question, corpus_tfidf_matrix, corpus_words)

    # Step 2: Find the most relevant document
    most_relevant_document = find_most_relevant_document(question_tfidf, corpus_tfidf_matrix, file_names)

    # Step 3: Generate the response
    response = generate_response(question, most_relevant_document)

    print("Generated Response:", response)

# Example usage
question = "Peux-tu me dire comment une nation peut-elle prendre soin du climat?"
corpus_tfidf_matrix = {}  # Replace with your TF-IDF matrix for the corpus
corpus_words = []  # Replace with the list of words in the corpus
file_names = []  # Replace with the list of file names in the corpus



# Function 7

def refine_answer(response, question):
    """Refines the answer for it to have a proper structure."""
    # Dictionary of possible text forms and associated model responses
    question_starters = {"Comment": "Après analyse, {}", "Pourquoi": "Car, {}", "Peux-tu": "Oui, bien sûr! {}"}

    # Capitalizes the first letter of the response
    refined_response = response.capitalize()

    # Add a period at the end of the response if it doesn't have one
    if not refined_response.endswith('.'):
        refined_response += '.'

    # Checks for specific question starters and uses corresponding model responses
    for starter, model_response in question_starters.items():
        if question.startswith(starter):

            refined_response = model_response.format(refined_response)

    return refined_response


# Example of usage:
question = "Peux-tu me dire comment une nation peut-elle prendre soin du climat?"
response = "et je songe bien sûr à françois hollande, faisant oeuvre de précurseur avec l'accord de paris sur le climat et protégeant les français dans un monde frappé par le terrorisme."

# Refine the answer
refined_response = refine_answer(response, question)
print("Refined Response:", refined_response)







# Function for Part 1 functionalities
def part_1_functionalities():
    # Part one functionalities here
    print("You selected Part 1 functionalities.")
    # Code sending to part 1

# Function for chatbot mode
def chatbot_mode():
    # Chatbot mode functionalities here
    print("You selected Chatbot mode.")
    # Code sending to chatbot functions


# Main program
while True:
    # Display menu options
    print("\nMenu:")
    print("1. Access Part I functionalities")
    print("2. Access Chatbot mode")
    print("3. Exit")

    # Get user choice
    choice = input("Enter your choice (1, 2, or 3): ")

    # Process user choice
    if choice == "1":
        part_1_functionalities()
    elif choice == "2":
        chatbot_mode()
    elif choice == "3":
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")