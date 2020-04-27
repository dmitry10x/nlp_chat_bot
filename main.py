import pymorphy2
import pymystem3
import pandas as pd
import numpy as np
import nltk
import re

m1 = pymorphy2.MorphAnalyzer()
m2 = pymystem3.Mystem()

def lemmatize_word(word):
    try:
        word = m2.analyze(word)[0]['analysis'][0]['lex']
    except:
        pass
    return word

def create_vector(corpus, sentence):
    # lemmatized_words = [lemmatize_word(word) for word in sentence]
    vector = {}
    vector['term_id'] = sentence['term_id']
    # vector = np.array([])
    for word in corpus:
        if word in sentence['question']:
            vector[word] = 1
            # vector = np.append(vector, 1)
        else:
            vector[word] = 0
            # vector = np.append(vector, 0)
    vector = pd.Series(vector) #, name='vector'
    print(vector)
    return vector

def get_list_of_lists_of_tokenize_sentences(questions_df):
    all_questions = []
    for index, row in questions_df.iterrows():
        all_questions.append({'term_id':row["term_id"],
                              'question':row["question"]})
    t_sentences = []

    for i in all_questions:
        t_sentense = {'term_id':i['term_id'],
                              'question':nltk.word_tokenize(i['question'])}
        t_sentences.append(t_sentense)

    return t_sentences

def create_corpus(t_sentences):
    corpus = set()
    for sent in t_sentences:
        for token in sent['question']:
            token = token.lower()
            token = re.sub(r'\W','',token)
            token = re.sub(r'\s+','',token)
            token = lemmatize_word(token)
            if len(token) != 0 and len(token) != ' ':
                corpus.add(token)
    corpus = list(corpus)
    for i in corpus:
        if i == '':
            corpus.remove(i)
    print(corpus)
    return corpus


def create_dataset(questions_df):
    t_sentences = get_list_of_lists_of_tokenize_sentences(questions_df)
    corpus = create_corpus(t_sentences)

    columns = [i for i in corpus]
    # columns.insert(0,'ques_id')
    columns.append('term_id')
    dataset = pd.DataFrame(columns=columns)

    for sentence in t_sentences:
        vector = create_vector(corpus, sentence)
        dataset = dataset.append(vector,ignore_index=True)
    return dataset


def create_trained_vectors(df):
    trained_vectors = {}
    columns = [column for column in df.columns]
    columns.remove('term_id')
    columns.remove('Unnamed: 0')
    for index, row in df.iterrows():
        trained_vectors[index] = {'term_id':row['term_id'],
                                    'vector':np.array([])}
        for column in columns:
            trained_vectors[index]['vector'] = np.append(trained_vectors[index]['vector'], row[column])
    return trained_vectors

def handle_question():
    marks_to_remove = [' ','',',','.',':','/',';','?','!','-','—']
    question = input('What is your question?:')
    #tokenize
    question_tokens = nltk.word_tokenize(question)
    #remove punctuation marks
    for i in range(len(question_tokens) - 1):
            question_tokens[i] = question_tokens[i].lower()
            question_tokens[i] = re.sub(r'\W','',question_tokens[i])
            question_tokens[i] = re.sub(r'\s+','',question_tokens[i])

            if str(question_tokens[i]) in marks_to_remove:
                question_tokens.pop(i)
            else:
                #lemmatize
                question_tokens[i] = lemmatize_word(question_tokens[i])
    return question_tokens

def find_nearest_neighbor(question_tokens, dataset):
    trained_vectors = create_trained_vectors(dataset)
    features = [feature for feature in dataset.columns]
    features.remove('term_id')
    features.remove('Unnamed: 0')

    #vector creation
    vector = np.array([])
    for feature in features:
        if feature in question_tokens:
            vector = np.append(vector, 1)
        else:
            vector = np.append(vector, 0)

    #finding nearest neighbor
    scores = []
    for t_vector in trained_vectors:
        distance = sum(t_vector - vector)
        scores.append(distance)
    max_value_index = scores.index(max(scores))

    print(vector)
    print('max_value_index', max_value_index)

    print(trained_vectors[max_value_index])

################
questions_df = pd.read_csv('questions.csv', sep=';', names=['term_id','question'])
dataset = create_dataset(questions_df)
dataset.to_csv('dataset_full.csv')

# ################
# questions_df = pd.read_csv('dataset.csv', sep=',')
# trained_vectors = create_trained_vectors(questions_df)
# q = handle_question()
# print(q)

# find_nearest_neighbor(q,questions_df)





# for i in range(len(all_questions)):
#     all_questions[i] = all_questions[i].lower()
#     all_questions[i] = re.sub(r'\W',' ',all_questions[i])
#     all_questions[i] = re.sub(r'\s+',' ',all_questions[i])

# all_words = set()
# for sentence in all_questions:
#     tokens = nltk.word_tokenize(sentence)
#     for token in tokens:
#         all_words.add(token)

# print(all_words)



# lemmatized_words = [lemmatize_word(word) for word in all_words]
# print(lemmatized_words)

# corpus = ['портфель', 'рынок', 'что', 'инвестиция', 'фондовый', 'на', 'etf', 'такой', 'голубой', 'фишка', 'етф', 'зарабатывать', 'биржевой', 'заработок', 'в', 'как', 'диверсифицировать', 'фонд', 'инвестиционный', 'диферсификация']


# for sentence in all_questions:
#     tokens = nltk.word_tokenize(sentence)
#     lemmatized_sentence = []
#     for token in tokens:
#         lemmatized_sentence.append(lemmatize_word(token))
#     print (lemmatized_sentence)



# for sentence in all_questions:
#     vector = create_vector(corpus, sentence)
#     print(vector)
