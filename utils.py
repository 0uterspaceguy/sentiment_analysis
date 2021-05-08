from pymorphy2 import MorphAnalyzer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from CONST import *

stemmer = SnowballStemmer("russian")
morph = MorphAnalyzer()

def remove_stopwords(sentence):
    """
    Function removes stopwords from sentence:

    Args:
        sentence (str): text sentence

    Returns:
        (str): input sentence without stopwords
    """
    words = word_tokenize(sentence)
    filtered_words = [w for w in words if not w in stopwords.words('russian')]
    return ' '.join(filtered_words)


def lemmatize(sentence):
    """
    Function for lemmatize words in sentence:

    Args:
        sentence (str): text sentence

    Returns:
        (str): input sentence with lemmatize words
    """
    words = word_tokenize(sentence)
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)


def remove_punktuations(sentence):
    """
    Function removes punctuations from sentence:

    Args:
        sentence (str): text sentence

    Returns:
        (str): input sentence without punkts
    """
    no_punkt = ''
    for char in sentence:
        if not char in punkts:
            no_punkt += char
    return no_punkt


def stemming(sentence):
    """
    Function use stemming for words in sentence:

    Args:
        sentence (str): text sentence

    Returns:
        (str): input sentence with stem words
    """
    words = word_tokenize(sentence)
    stemms = [stemmer.stem(word) for word in words]
    return ' '.join(stemms)


def filter_symbols(sentence):
    """
    Function removes all symbols which not in dictionary from input sentence:

    Args:
        sentence (str): text sentence

    Returns:
        (str): filtered sentence
    """
    del_s = set(sentence) - dictionary
    for s in del_s:
        sentence = sentence.replace(s, '')
    return sentence.strip()


def remove_repetitions(sentence):
    """
    Function removes repeating characters from sentence:
       - allows two letters in a row

    Args:
        sentence (str): text sentence

    Returns:
        (str): input sentence without repetitions
    """
    new_text = sentence[:2]
    for x in sentence[2:]:
        if x.isalpha():
            if new_text[-2] != x:
                new_text += x
                continue
            else:
                if new_text[-1] != x:
                    new_text += x
                    continue
        else:
            if new_text[-1] != x:
                new_text += x
                continue
    return new_text
