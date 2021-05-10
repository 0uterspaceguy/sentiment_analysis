import json
from transformers import AutoTokenizer
import numpy as np
from utils import *
import random


class Dataset():
    def __init__(self,
                 tokenizer,
                 to_lower=True,
                 do_filter=True,
                 remove_repetitions=True,
                 remove_stopwords=False,
                 remove_punktuations=True,
                 lemmatize=False,
                 stemming=False,
                 n_classes=3):

        self.sentiment_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.to_lower = to_lower
        self.do_filter = do_filter
        self.remove_repetitions = remove_repetitions
        self.remove_stopwords = remove_stopwords
        self.remove_punktuations = remove_punktuations
        self.lemmatize = lemmatize
        self.stemming = stemming
        self.n_classes = n_classes

    def load_dataset(self, path2jsonl):
        """
        This function load dataset from jsonl file with next structure:
            [{text: text, label: sentiment_label},]

        and then save it to self.texts and self.labels arrays.

        sentiment_label is one of:
            - negative
            - positive
            - neutral

        Args:
            path2jsonl (str): path to jsonl file

        """
        with open(path2jsonl, 'r', encoding='utf-8') as json_file:
            list_data = list(json_file)

        self.texts = []
        self.labels = []

        for line in list_data:
            data = json.loads(line)
            text = data['text']
            label = data['label']

            self.labels.append(self.sentiment_labels[label])
            self.texts.append(text)

        self.shuffle_dataset()

    def shuffle_dataset(self):
        """
        This function shuffle whole dataset (self.texts, self.labels)
        """
        combined = list(zip(self.texts, self.labels))

        random.shuffle(combined)
        self.texts[:], self.labels[:] = zip(*combined)

    def prepare_text(self, text):
        """
        Function for preparing text before training model:

        Args:
            sentence (str): text sentence

        Returns:
            (str): prepared text
        """
        if self.to_lower:
            text = text.lower()
        if self.do_filter:
            text = filter_symbols(text)
        if self.remove_repetitions:
            text = remove_repetitions(text)
        if self.remove_stopwords:
            text = remove_stopwords(text)
        if self.remove_punktuations:
            text = remove_punktuations(text)
        if self.stemming:
            text = stemming(text)
        if self.lemmatize:
            text = lemmatize(text)
        return text

    def tokenize_text(self, sentence):
        """
        Function for tokenize text:

        Args:
            sentence (str): text sentence

        Returns:
            ids (np.ndarray): input ids for model
            masks (np.ndarray): input mask for model
        """
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))

        ids = pad_sequences(
            ids,
            maxlen=512,
            dtype="long",
            truncating="pre",
            padding="post"
        )

        mask = [(x > 0).astype('float') for x in ids]
        return ids[0], mask[0]

    def __getitem__(self, i):
        """
        Function returns one sample for training model:

        Args:
            i (int): sample index

        Returns:
            ids (np.ndarray): input tokens ids
            masks (np.ndarray): input mask
            label (np.ndarray): label in one hot encoding
        """

        prepared_text = self.prepare_text(self.texts[i])
        ids, masks = self.tokenize_text(prepared_text)

        label = np.zeros(self.n_classes)
        label[self.labels[i]] = 1

        return np.array(ids), np.array(masks), label

    def __len__(self):
        """
        Function return length of dataset
        """
        return len(self.texts)
