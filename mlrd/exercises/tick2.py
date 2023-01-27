import math
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    dictionary = dict()
    positive = 0
    negative = 0
    for i in range(len(training_data)):
        if training_data[i]['sentiment'] == 1:
            positive += 1
        elif training_data[i]['sentiment'] == -1:
            negative += 1
    dictionary[1] = math.log(positive / len(training_data))
    dictionary[-1] = math.log(negative / len(training_data))
    # print(dictionary[1], dictionary[-1])
    return dictionary


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    dictionary = dict()
    dictionary[-1] = dict()
    dictionary[1] = dict()
    for i in range(len(training_data)):  # each text review
        sentiment = training_data[i]['sentiment']
        text_list = training_data[i]['text']
        for j in range(len(text_list)):  # keyword in each review
            if text_list[j] in dictionary[sentiment]:
                dictionary[sentiment][text_list[j]] += 1
            else:
                dictionary[sentiment][text_list[j]] = 1
    value1 = sum(dictionary[1].values())
    value2 = sum(dictionary[-1].values())
    for k in dictionary[1]:
        if dictionary[1][k] != 0:
            dictionary[1][k] = math.log(dictionary[1][k] / value1)

    for k in dictionary[-1]:
        if dictionary[-1][k] != 0:
            dictionary[-1][k] = math.log(dictionary[-1][k] / value2)
    return dictionary


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    dictionary = dict()
    dictionary[-1] = dict()
    dictionary[1] = dict()
    for i in range(len(training_data)):  # each text review
        sentiment = training_data[i]['sentiment']
        text_list = training_data[i]['text']
        for j in range(len(text_list)):  # keyword in each review
            if text_list[j] in dictionary[sentiment]:
                dictionary[sentiment][text_list[j]] += 1
            else:
                dictionary[sentiment][text_list[j]] = 1
            if not text_list[j] in dictionary[-sentiment]:
                dictionary[-sentiment][text_list[j]] = 1  # default
    value1 = sum(dictionary[1].values())
    value2 = sum(dictionary[-1].values())
    value = len(dictionary[1].keys() | dictionary[-1].keys())
    # is vocabulary of all distinct words, no matter which class c a word w occurred with.
    for k in dictionary[1]:
        dictionary[1][k] = math.log((dictionary[1][k] + 1) / (value + value1))

    for k in dictionary[-1]:
        dictionary[-1][k] = math.log((dictionary[-1][k] + 1) / (value + value2))
    return dictionary


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    pos = class_log_probabilities[1]
    neg = class_log_probabilities[-1]
    for i in range(len(review)):
        if review[i] in log_probabilities[1]:
            pos += log_probabilities[1][review[i]]
        if review[i] in log_probabilities[-1]:
            neg += log_probabilities[-1][review[i]]
    if pos >= neg:
        return 1
    else:
        return -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")


if __name__ == '__main__':
    main()
