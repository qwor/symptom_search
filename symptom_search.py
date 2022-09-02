import os

import pandas as pd
import numpy as np
import json
from thefuzz import process
from gensim.models.fasttext import FastText


def create_small_data(df):
    df_main = df['"Прочие симптомы"']
    small_df = pd.DataFrame(np.array_split(df_main, 100)[0])
    small_df.to_csv("data_small.csv")


def create_words(df_main):
    words = []
    for row in df_main:
        words += row.strip('"').replace(',', ' ').replace(';', ' ').replace('.', ' ').split(' ')
    words = set(words)

    def filter_func(s):
        all(i.isalpha() for i in s) and len(s) > 2

    words = sorted(list(filter(filter_func, words)))
    words_df = pd.DataFrame(words)
    words_df.to_csv('words.csv')
    return words


def create_similarity_dictionary(words, min_sim_index):
    dict_sims = {}
    for word in words:
        # filtered_list is of structure [(word, similarity_index)...]
        filtered_list = filter(lambda tup: tup[1] >= min_sim_index, process.extract(word, words, limit=100))
        dict_sims[word] = [word] + list(map(lambda tup: tup[0], filtered_list))
    if not os.path.isdir('dictionaries'):
        os.mkdir('dictionaries')
    with open(f'dictionaries/dict_{min_sim_index}.json', 'w') as outfile:
        json.dump(dict_sims, outfile)


def load_dictionaries():
    dictionaries = []
    for file in os.listdir('dictionaries'):
        if file.endswith('.json'):
            with open(f'dictionaries/{file}', 'r') as infile:
                dictionaries.append(json.load(infile))
    return dictionaries


def create_model(words, dictionaries):
    model = FastText(dictionaries, sg=1, min_count=1)
    model.save('symptoms.model')
    return model


def find_similar_words_thefuzz(words, word, limit):
    return process.extract(word, words, limit=limit)


def find_similar_words_fasttext(model, word, limit):
    return model.wv.most_similar_cosmul(word, topn=limit)


def find_patients(df, df_main, similar_words):
    patient_set = set()
    for word, index in similar_words:
        patient_set.update(df.index[df_main.str.contains(word)].tolist())
    return patient_set


def find_patients_thefuzz(df, df_main, words, search_word, limit):
    similar_words = [(search_word, 100)] + find_similar_words_thefuzz(words, search_word, limit)
    return find_patients(df, df_main, similar_words)


def find_patients_fasttext(df, df_main, model, search_word, limit):
    similar_words = [(search_word, 1)] + find_similar_words_fasttext(model, search_word, limit)
    return find_patients(df, df_main, similar_words)


def test_similar_words(words, model, search_word, limit):
    print(f'thefuzz list for word {search_word}: {find_similar_words_thefuzz(words, search_word, limit)}')
    print(f'fasttext list for word {search_word}: {find_similar_words_fasttext(model, search_word, limit)}')


def test_patients_list(df, df_main, words, model, search_word, limit):
    print(f'thefuzz patient set length: {len(find_patients_thefuzz(df, df_main, words, search_word, limit))}')
    print(f'fasttext patient set length: {len(find_patients_fasttext(df, df_main, model, search_word, limit))}')


def main():
    df = pd.read_csv('data.csv', index_col=0)
    df_main = df['"Прочие симптомы"']
    words = pd.read_csv('words.csv')['0'].values.tolist()
    # create_similarity_dictionary(words, 60)
    # create_similarity_dictionary(words, 80)
    # dictionaries = load_dictionaries()
    # create_model(words_df, dictionaries)
    model = FastText.load('symptoms.model')
    test_similar_words(words, model, 'кашель', 10)
    test_patients_list(df, df_main, words, model, 'кашель', 5)
    print('done')


if __name__ == '__main__':
    main()
