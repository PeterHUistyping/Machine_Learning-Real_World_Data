import math
import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    dictionary = dict()
    positive = 0
    negative = 0
    neu = 0
    for i in range(len(training_data)):
        if training_data[i]['sentiment'] == 1:
            positive += 1
        elif training_data[i]['sentiment'] == -1:
            negative += 1
        else:
            neu += 1
    dictionary[1] = math.log(positive / len(training_data))
    dictionary[-1] = math.log(negative / len(training_data))
    dictionary[0] = math.log(neu / len(training_data))
    # print(dictionary[1], dictionary[-1])
    return dictionary


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    dictionary = dict()
    dictionary[-1] = dict()
    dictionary[1] = dict()
    dictionary[0] = dict()
    for i in range(len(training_data)):  # each text review
        sentiment = training_data[i]['sentiment']
        text_list = training_data[i]['text']
        for j in range(len(text_list)):  # keyword in each review
            if text_list[j] in dictionary[sentiment]:
                dictionary[sentiment][text_list[j]] += 1
            else:
                dictionary[sentiment][text_list[j]] = 1
            if not text_list[j] in dictionary[-1]:
                dictionary[-1][text_list[j]] = 1  # default
            if not text_list[j] in dictionary[0]:
                dictionary[0][text_list[j]] = 1
            if not text_list[j] in dictionary[1]:
                dictionary[1][text_list[j]] = 1
    value1 = sum(dictionary[1].values())
    value2 = sum(dictionary[-1].values())
    value3 = sum(dictionary[0].values())
    value = len(dictionary[1].keys() | dictionary[-1].keys()|dictionary[0].keys())
    # is vocabulary of all distinct words, no matter which class c a word w occurred with.
    for k in dictionary[1]:
        dictionary[1][k] = math.log((dictionary[1][k] + 1) / (value + value1))

    for k in dictionary[-1]:
        dictionary[-1][k] = math.log((dictionary[-1][k] + 1) / (value + value2))
    for k in dictionary[0]:
        dictionary[0][k] = math.log((dictionary[0][k] + 1) / (value + value3))
    return dictionary


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = 0
    for i in range(len(true)):
        if pred[i] == true[i]:
            correct += 1
    return correct / len(true)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    pos = class_log_probabilities[1]
    neg = class_log_probabilities[-1]
    net = class_log_probabilities[0]
    for i in range(len(review)):
        if review[i] in log_probabilities[1]:
            pos += log_probabilities[1][review[i]]
        if review[i] in log_probabilities[-1]:
            neg += log_probabilities[-1][review[i]]
        if review[i] in log_probabilities[0]:
            net += log_probabilities[0][review[i]]
    max_ = max(pos, neg, net)
    if max_ == neg:
        return -1
    elif max_ == pos:
        return 1
    else:
        return 0


def calculate_kappa(agreement_table: Dict[int, Dict[int, int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    key=list(agreement_table.keys())[0]
    num_views= len(agreement_table.keys())
    num_ann = agreement_table[key][1] + agreement_table[key][-1]
    pairs = (num_ann - 1) * num_ann
    numer = 0
    pe_pos = 0
    pe_neg = 0
    for i in agreement_table.keys():
        numer += (agreement_table[i][1] * (agreement_table[i][1] - 1) + agreement_table[i][-1] * (
                agreement_table[i][-1] - 1)) / pairs
        pe_pos += agreement_table[i][1] / num_ann
        pe_neg += agreement_table[i][-1] / num_ann
    p_a = numer / num_views
    p_e = (pe_pos / num_views) ** 2 + (pe_neg / num_views) ** 2

    return (p_a - p_e) / (1 - p_e)  # kappa


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int, int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    first_pos = 0
    first_neg = 0
    sec_pos = 0
    sec_neg = 0
    thr_pos = 0
    thr_neg = 0
    for_pos = 0
    for_neg = 0

    dict_ = dict()

    for pre in review_predictions:
        if pre[0] == 1:
            first_pos += 1
        else:
            first_neg += 1
        if pre[1] == 1:
            sec_pos += 1
        else:
            sec_neg += 1
        if pre[2] == 1:
            thr_pos += 1
        else:
            thr_neg += 1
        if pre[3] == 1:
            for_pos += 1
        else:
            for_neg += 1
    dict_[0] = {1: first_pos, -1: first_neg}
    dict_[1] = {1: sec_pos, -1: sec_neg}
    dict_[2] = {1: thr_pos, -1: thr_neg}
    dict_[3] = {1: for_pos, -1: for_neg}

    return dict_


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i + 1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(
        os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x: y for x, y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x: y for x, y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(
        os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")


if __name__ == '__main__':
    main()
