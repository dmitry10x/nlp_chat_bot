import pymorphy2
import pymystem3
import pandas as pd
import nltk
import re

m1 = pymorphy2.MorphAnalyzer()
m2 = pymystem3.Mystem()

parse = m1.parse('сидел')
analyze = m2.analyze('сидели')
# print('parse:',parse)
# print('analyze:',analyze[0]['analysis'][0]['lex'])


def lemmatize_word(word):
    try:
        word = m2.analyze(word)[0]['analysis'][0]['lex']
    except:
        pass
    return word


questions_df = pd.read_csv('questions.csv', sep=';', names=['term_id','question'])

all_questions = []
for index, row in questions_df.iterrows():
    all_questions.append(row["question"])

for i in range(len(all_questions)):
    all_questions[i] = all_questions[i].lower()
    all_questions[i] = re.sub(r'\W',' ',all_questions[i])
    all_questions[i] = re.sub(r'\s+',' ',all_questions[i])

all_words = set()
for sentence in all_questions:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        all_words.add(token)

print(all_words)

# for word in all_words:
    # print(lemmatize_word(word))
lemmatized_words = [lemmatize_word(word) for word in all_words]
print(lemmatized_words)
