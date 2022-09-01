import os

import pandas as pd
import numpy as np
import json
from thefuzz import process
from gensim.models.word2vec import Word2Vec


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


def create_similarity_dictionary(words_df, min_sim_index):
    words = words_df['0'].values.tolist()
    dict_sims = {}
    for word in words:
        # filtered_list is of structure [(word, similarity_index)...]
        filtered_list = filter(lambda tup: tup[1] >= min_sim_index, process.extract(word, words, limit=100))
        dict_sims[word] = list(map(lambda tup: tup[0], filtered_list))
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


def create_word2Vec_model(words, dictionaries):
    model = Word2Vec(dictionaries, min_count=1)
    # model.build_vocab(words, min_count=1)
    # model.train(dictionaries)
    model.save('symptoms.model')
    return model


def main():
    # df = pd.read_csv('data.csv', index_col=0)
    # df_main = df['"Прочие симптомы"']
    words_df = pd.read_csv('words.csv')
    # create_similarity_dictionary(words_df, 60)
    # create_similarity_dictionary(words_df, 80)
    dictionaries = load_dictionaries()
    print(dictionaries[0]['кашель'])
    create_word2Vec_model(words_df, dictionaries[0])
    model = Word2Vec.load('symptoms.model')
    print(model.wv.most_similar('кашель'))
    print('done')


if __name__ == '__main__':
    main()
