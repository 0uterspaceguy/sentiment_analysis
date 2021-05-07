import os
from os import path as p
import pandas as pd
import numpy as np
import json
import random
import argparse

def get_mokoron():
    """
    Function to load and prepare mokoron-sentiment dataset

    Returns:
        dict: {"positive": positive sentiments, "negative": negative sentiments}

    """
    mokoron_negative = pd.read_csv(p.join(data_dir, 'mokoron', 'negative.csv'), sep=';', comment='#')
    mokoron_positive = pd.read_csv(p.join(data_dir, 'mokoron', 'positive.csv'), sep=';', comment='#')

    mokoron_negative = mokoron_negative.iloc[:, 3]
    mokoron_positive = mokoron_positive.iloc[:, 3]

    mokoron_negative.columns = ['text']
    mokoron_positive.columns = ['text']

    mokoron_negative = mokoron_negative.values
    mokoron_positive = mokoron_positive.values

    mokoron = {"positive": mokoron_positive, "negative": mokoron_negative}
    return mokoron


def get_rusentiment():
    """
    Function to load and prepare rusentiment dataset

    Returns:
        dict: {"positive": positive sentiments, "negative": negative sentiments, "neutral": neutral sentiments}

    """
    rusentiment_random_posts = pd.read_csv(p.join(data_dir, 'rusentiment', 'rusentiment_random_posts.csv'))
    rusentiment_test = pd.read_csv(p.join(data_dir, 'rusentiment', 'rusentiment_test.csv'))
    rusentiment_preselected_posts = pd.read_csv(p.join(data_dir, 'rusentiment', 'rusentiment_preselected_posts.csv'))

    rusentiment_positive = pd.concat([rusentiment_random_posts[rusentiment_random_posts['label'] == 'positive'], \
                                      rusentiment_test[rusentiment_test['label'] == 'positive'], \
                                      rusentiment_preselected_posts[
                                          rusentiment_preselected_posts['label'] == 'positive']]).drop('label',
                                                                                                       axis=1).values.squeeze(
        -1)

    rusentiment_negative = pd.concat([rusentiment_random_posts[rusentiment_random_posts['label'] == 'negative'], \
                                      rusentiment_test[rusentiment_test['label'] == 'negative'], \
                                      rusentiment_preselected_posts[
                                          rusentiment_preselected_posts['label'] == 'negative']]).drop('label',
                                                                                                       axis=1).values.squeeze(
        -1)

    rusentiment_neutral = pd.concat([rusentiment_random_posts[rusentiment_random_posts['label'] == 'neutral'], \
                                     rusentiment_test[rusentiment_test['label'] == 'neutral'], \
                                     rusentiment_preselected_posts[
                                         rusentiment_preselected_posts['label'] == 'neutral']]).drop('label',
                                                                                                     axis=1).values.squeeze(
        -1)

    rusentiment = {"positive": rusentiment_positive, "negative": rusentiment_negative, "neutral": rusentiment_neutral}
    return rusentiment


def get_rureviews():
    """
    Function to load and prepare rureviews-sentiment dataset

    Returns:
        dict: {"positive": positive sentiments, "negative": negative sentiments, "neutral": neutral sentiments}

    """
    rureviews = pd.read_csv(p.join(data_dir, 'rureviews', 'women-clothing-accessories.3-class.balanced.csv'), sep='\t')

    rureviews_positive = (rureviews[rureviews['sentiment'] == 'positive']).drop(['sentiment'], axis=1).values.squeeze(
        -1)
    rureviews_negative = (rureviews[rureviews['sentiment'] == 'negative']).drop(['sentiment'], axis=1).values.squeeze(
        -1)
    rureviews_neutral = (rureviews[rureviews['sentiment'] == 'neautral']).drop(['sentiment'], axis=1).values.squeeze(-1)

    rureviews = {"positive": rureviews_positive, "negative": rureviews_negative, "neutral": rureviews_neutral}
    return rureviews


def get_medical():
    """
    Function to load and prepare medical-sentiment dataset

    Returns:
        dict: {"positive": positive sentiments, "negative": negative sentiments, "neutral": neutral sentiments}

    """
    medical_comments = pd.read_csv(p.join(data_dir, 'medical_comments', 'medical_comments.csv'))

    medical_positive = (medical_comments[
        (medical_comments['sentiment'] == ' Отлично') | (medical_comments['sentiment'] == ' Хорошо')]).drop(
        ['sentiment'], axis=1).values.squeeze(-1)
    medical_negative = (
        medical_comments[
            (medical_comments['sentiment'] == ' Ужасно') | (medical_comments['sentiment'] == ' Плохо')]).drop(
        ['sentiment'], axis=1).values.squeeze(-1)
    medical_neutral = (medical_comments[(medical_comments['sentiment'] == ' Нормально')]).drop(['sentiment'],
                                                                                               axis=1).values.squeeze(
        -1)

    medical = {"positive": medical_positive, "negative": medical_negative, "neutral": medical_neutral}
    return medical


def get_kaggle():
    """
    Function to load and prepare kaggle-sentiment dataset

    Returns:
        dict: {"positive": positive sentiments, "negative": negative sentiments, "neutral": neutral sentiments}

    """
    with open(p.join(data_dir, 'kaggle-sentiment', 'train.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    kaggle_positive = []
    kaggle_negative = []
    kaggle_neutral = []

    for line in data:
        if line['sentiment'] == 'positive':
            kaggle_positive.append(line['text'])
        elif line['sentiment'] == 'negative':
            kaggle_negative.append(line['text'])
        else:
            kaggle_neutral.append(line['text'])

    kaggle_positive = np.array(kaggle_positive)
    kaggle_negative = np.array(kaggle_negative)
    kaggle_neutral = np.array(kaggle_neutral)

    kaggle = {"positive": kaggle_positive, "negative": kaggle_negative, "neutral": kaggle_neutral}

    return kaggle


def collect_dataset(chosen_datasets):
    """
    Function to collect all chosen datasets in one
    Args:
        chosen_datasets (list): List of names of choden datasets

    Returns:
        sought_data (list): [{"text": text_data, "label": label of sentiment},]
    """
    datasets_f = {'mokoron': get_mokoron,
                  'rureviews': get_rureviews,
                  'medical': get_medical,
                  'rusentiment': get_rusentiment,
                  'kaggle': get_kaggle}

    sought_data = []

    for dataset_name in chosen_datasets:
        dataset = datasets_f[dataset_name]()
        for sentiment in dataset:
            for sample in dataset[sentiment]:
                sought_data.append({'text': sample, 'label': sentiment})

    random.shuffle(sought_data)
    return sought_data


def split_dataset(data, val_split=0.01):
    """
    Function to split dataset into train and validation
    Args:
        data (list): [{"text": text_data, "label": label of sentiment},]

    Returns:
        train (list): [{"text": text_data, "label": label of sentiment},]
        test (list): [{"text": text_data, "label": label of sentiment},]
    """
    treshold = int(len(data) * val_split)
    return data[treshold:], data[:treshold]


def save_dataset(data, path):
    """
    Function to save dataset
    Args:
        data (list): [{"text": text_data, "label": label of sentiment},]
        path (str): path to saving directiory

    """
    with open(path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glueing and splitting datasets')
    parser.add_argument('--data_directory', type=str, default='sentiment-datasets', help='Path to data directory')
    parser.add_argument('--val_split', type=float, default=0.01, help='Validation proportion')
    parser.add_argument("--mokoron", type=bool, const='True', nargs='?', default=False, help="Use mokoron dataset")
    parser.add_argument("--rureviews", type=bool, const='True', nargs='?', default=False, help="Use rureviews dataset")
    parser.add_argument("--medical", type=bool, const='True', nargs='?', default=False, help="Use medical dataset")
    parser.add_argument("--rusentiment", type=bool, const='True', nargs='?', default=False,
                        help="Use rusentiment dataset")
    parser.add_argument("--kaggle", type=bool, const='True', nargs='?', default=False, help="Use kaggle dataset")

    args = parser.parse_args()

    data_dir = args.data_directory

    datasets_list = ['mokoron', 'rureviews', 'medical', 'rusentiment', 'kaggle']
    datasets_mask = [args.mokoron, args.rureviews, args.medical, args.rusentiment, args.kaggle]
    chosen_datasets = [datasets_list[i] for i in range(len(datasets_list)) if datasets_mask[i]]

    dataset = collect_dataset(chosen_datasets)
    train, validation = split_dataset(dataset, args.val_split)

    save_dataset(train, p.join(data_dir,'train.jsonl'))
    save_dataset(validation, p.join(data_dir, 'val.jsonl'))
