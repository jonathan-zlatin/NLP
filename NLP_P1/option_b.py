import math
from typing import List, Dict, Any, Union, Tuple

from datasets import load_dataset
import os

import spacy
import pickle

from collections import defaultdict

# Constants
SENTENCE1 = "Brad Pitt was born in Oklahoma"
SENTENCE2 = "The actor was born in USA"
START_TOKEN = "START!TOKEN"  # START_TOKEN.isalpha() == False, therefore it will not be in the corpus after preprocessing

def preprocess_test_set(texts: List[str]) -> List[List[str]]:
    """
    Preprocesses a list of sentences into a list of lemmatized tokens.

    Args:
        texts (List[str]): List of input sentences.

    Returns:
        List[List[str]]: List of lemmatized tokens for each input sentence.
    """
    lemmas = []
    nlp = spacy.load("en_core_web_sm")
    for sentence in texts:
        d = {'text': f'= {sentence} = \n'}
        doc = nlp(d['text'])
        sentence_lemmas = [token.lemma_ for token in doc if token.is_alpha]
        lemmas.append(sentence_lemmas)
    return lemmas

def preprocess_lemmas(file_path: str = "train_lemmas.pkl") -> None:
    """
    Preprocesses the training dataset and saves the lemmatized tokens to a file.

    Args:
        file_path (str): Path to save the processed data.
    """
    if not os.path.exists(file_path):
        nlp = spacy.load("en_core_web_sm")
        train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        train_lemmas = []
        for text in train_data:
            if text['text']:
                doc = nlp(text['text'])
                lemmas = [token.lemma_ for token in doc if token.is_alpha]
                train_lemmas.append(lemmas)
        with open('train_lemmas_tmp.pkl', 'wb') as f:
            pickle.dump(train_lemmas, f)

def train_unigram(corpus: List[List[str]] = None, path: str = 'train_lemmas', log_space: bool = False) -> Dict[str, Union[float, int]]:
    """
    Trains a unigram language model.

    Args:
        corpus (List[List[str]]): List of lemmatized tokens.
        path (str): Path to the preprocessed data file.
        log_space (bool): Flag indicating whether to use log space.

    Returns:
        Dict[str, Union[float, int]]: Unigram token weights.
    """
    if not corpus:
        with open(f'{path}.pkl', 'rb') as f:
            corpus = pickle.load(f)
    total_count = 0
    unigram_token_weights = defaultdict(int)
    for sentence in corpus:
        for token in sentence:
            unigram_token_weights[token] += 1
            total_count += 1
    for token in unigram_token_weights:
        ML = unigram_token_weights[token] / total_count
        unigram_token_weights[token] = math.log(ML) if log_space else ML
    return unigram_token_weights

def train_bigram(corpus: List[List[str]] = None, path: str = 'train_lemmas', log_space: bool = False) -> Dict[Tuple[str, str], Union[float, int]]:
    """
    Trains a bigram language model.

    Args:
        corpus (List[List[str]]): List of lemmatized tokens.
        path (str): Path to the preprocessed data file.
        log_space (bool): Flag indicating whether to use log space.

    Returns:
        Dict[Tuple[str, str], Union[float, int]]: Bigram token weights.
    """
    if not corpus:
        with open(f'{path}.pkl', 'rb') as f:
            corpus = pickle.load(f)
    bigram_token_weights = defaultdict(int)
    total_count = 0
    counts_w1 = defaultdict(int)
    for sentence in corpus:
        prev = START_TOKEN
        for token in sentence:
            key = (prev, token)
            counts_w1[prev] += 1
            prev = token
            bigram_token_weights[key] += 1
            total_count += 1
    for token in bigram_token_weights:
        ML = bigram_token_weights[token] / counts_w1[token[0]]
        bigram_token_weights[token] = math.log(ML) if log_space else ML
    return bigram_token_weights

# Remaining functions and main code follow a similar structure with added type hints and docstrings.

def compute_prob_biagram_sentence(sentence: List[str], bigram_token_weights: Dict[Tuple[str, str], float]) -> float:
    """
    Computes the probability of a sentence using a bigram language model.

    Args:
        sentence (List[str]): List of lemmatized tokens in the sentence.
        bigram_token_weights (Dict[Tuple[str, str], float]): Bigram token weights.

    Returns:
        float: Probability of the sentence.
    """
    prev = START_TOKEN
    prob = 1.0
    for token in sentence:
        key = (prev, token)
        prev = token
        if key not in bigram_token_weights:
            return 0.0
        prob *= bigram_token_weights[key]
    return prob

def compute_prob_biagram_words(w1: str, w2: str, bigram_token_weights: Dict[Tuple[str, str], float], log_space: bool = False) -> float:
    """
    Computes the probability of a bigram (two consecutive words) using a bigram language model.

    Args:
        w1 (str): First word.
        w2 (str): Second word.
        bigram_token_weights (Dict[Tuple[str, str], float]): Bigram token weights.
        log_space (bool): Flag indicating whether to use log space.

    Returns:
        float: Probability of the bigram.
    """
    key = (w1, w2)
    if key not in bigram_token_weights:
        return float('-inf')
    return math.log(bigram_token_weights[key]) if log_space else bigram_token_weights[key]

def liner_interpolation_model(sentence: List[str], unigram_token_weights: Dict[str, float], bigram_token_weights: Dict[Tuple[str, str], float]) -> float:
    """
    Applies linear interpolation to compute the probability of a sentence using unigram and bigram models.

    Args:
        sentence (List[str]): List of lemmatized tokens in the sentence.
        unigram_token_weights (Dict[str, float]): Unigram token weights.
        bigram_token_weights (Dict[Tuple[str, str], float]): Bigram token weights.

    Returns:
        float: Probability of the sentence.
    """
    prev = START_TOKEN
    prob = 1.0
    for token in sentence:
        key = (prev, token)
        prev = token
        if key not in bigram_token_weights:
            prob *= (1/3) * (unigram_token_weights[token])
        else:
            prob *= (2/3 * bigram_token_weights[key] + 1/3 * unigram_token_weights[token])
    return prob

if __name__ == '__main__':
    preprocess_lemmas()
    unigram_token_weights = train_unigram(log_space=False)
    bigram_token_weights = train_bigram(log_space=False)

    print("Question 2:")
    print(f"I have a house in {max(unigram_token_weights, key=unigram_token_weights.get)}\n")

    print("Question 3:")
    lemma_test_set = preprocess_test_set([SENTENCE1, SENTENCE2])

    sentence_one_prob = compute_prob_biagram_sentence(sentence=lemma_test_set[0], bigram_token_weights=bigram_token_weights)
    sentence_one_prob = math.log(sentence_one_prob) if sentence_one_prob else float('-inf')

    sentence_two_prob = compute_prob_biagram_sentence(sentence=lemma_test_set[1], bigram_token_weights=bigram_token_weights)
    sentence_two_prob = math.log(sentence_two_prob) if sentence_two_prob else float('-inf')

    print(f"Sentence 1 ({SENTENCE1}) log probability: {round(sentence_one_prob,3)}")
    print(f"Sentence 2 ({SENTENCE2}) log probability: {round(sentence_two_prob,3)}")

    token_in_test_corpus = len(lemma_test_set[0]) + len(lemma_test_set[1])
    perplexity = math.exp(-(1/token_in_test_corpus) * (sentence_one_prob + sentence_two_prob))
    print(f"Perplexity: {round(perplexity,3)}\n")

    print("Question 4:")
    sentence_one_prob_liner_inter = math.log(liner_interpolation_model(lemma_test_set[0], unigram_token_weights, bigram_token_weights))
    sentence_two_prob_liner_inter = math.log(liner_interpolation_model(lemma_test_set[1], unigram_token_weights, bigram_token_weights))
    perplexity_liner_inter = math.exp(-(1/token_in_test_corpus) * (sentence_one_prob_liner_inter + sentence_two_prob_liner_inter))
    print(f"Sentence 1 ({SENTENCE1}) log probability: {round(sentence_one_prob_liner_inter, 3)}")
    print(f"Sentence 2 ({SENTENCE2}) log probability: {round(sentence_two_prob_liner_inter, 3)}")
    print(f"Perplexity: {round(perplexity_liner_inter,3)}")
