import random
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, read_lexicon, predict_sentiment
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test

def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    Input = training_data  # only creates a reference to an object
    random.shuffle(Input)
    start = 0
    res = []
    each_len = len(Input) // n
    for i in range(0, n):
        res.append(Input[i * each_len:(i + 1) * each_len])
    return res


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    positive = []
    negative = []
    for review in training_data:
        if review['sentiment'] == -1:
            negative.append(review)
        else:
            positive.append(review)
    res = []
    random.shuffle(positive)
    random.shuffle(negative)
    each_len = len(training_data) // (2* n)
    for i in range(0, n):
        res.append(positive[i * each_len:(i + 1) * each_len]+ negative[i * each_len:(i + 1) * each_len])
    return res


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracy_ = []
    for i in range(len(split_training_data)):
        # training = []
        prediction=[]
        training_ = split_training_data.copy()
        training_.remove(split_training_data[i])
        training_list=[m for item in training_ for m in item]
        # for it in training_:
        #     for item in it:
        #         training+=item['text']
        class_pr = calculate_class_log_probabilities(training_list)
        smoothed_log_pr = calculate_smoothed_log_probabilities(training_list)
        for r in split_training_data[i]:
            prediction.append(predict_sentiment_nbc(r['text'], smoothed_log_pr, class_pr)) # cal outside, not need to traverse
        accuracy_.append(accuracy(prediction, [x['sentiment'] for x in split_training_data[i]]))  # validation
        training_list.clear()
    return accuracy_


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies) / len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    ave = cross_validation_accuracy(accuracies)
    variance = []
    for i in range(len(accuracies)):
        variance.append((accuracies[i] - ave) ** 2 )

    return sum(variance) / len(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    matrix = [[0 for i_ in range(2)] for j_ in range(2)] # [[0] * 2] * 2
    for i in range(len(predicted_sentiments)):
        if predicted_sentiments[i] == 1 and actual_sentiments[i] == 1:
            matrix[0][0] += 1
        elif predicted_sentiments[i] == 1 and actual_sentiments[i] == -1:
            matrix[0][1] += 1
        elif predicted_sentiments[i] == -1 and actual_sentiments[i] == 1:
            matrix[1][0] += 1
        elif predicted_sentiments[i] == -1 and actual_sentiments[i] == -1:
            matrix[1][1] += 1
    return matrix


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))
    # 2016
    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    # simple classifier performance
    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')
    # before
    pred_sim_test = [predict_sentiment(t, lexicon) for t in test_tokens]
    acc_sim_test = accuracy(pred_sim_test, [x['sentiment'] for x in test_data])
    print(f"Simple Sentiment Classifier performance accuracy on held-out data: {acc_sim_test}")

    # after
    pred_sim_recent = [predict_sentiment(t, lexicon) for t in recent_tokens]
    acc_sim_recent = accuracy(pred_sim_recent,recent_sentiments)
    print(f"Simple Sentiment Classifier performance accuracy on 2016 data: {acc_sim_recent}")

    p_v = sign_test(recent_sentiments, preds_recent, pred_sim_recent)
    print(f"P-value of significance test between NB and Simple Classifier on 2016 data: {p_v}")

if __name__ == '__main__':
    main()
