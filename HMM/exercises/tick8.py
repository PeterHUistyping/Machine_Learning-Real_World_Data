import math

from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random, sys

from typing import List, Dict, Tuple

sys.setrecursionlimit(int(1e4))
states = ['F', 'W', 'B', 'Z']


def log(n):
    if n == 0:
        return -math.inf
    return math.log(n)


def calculate_delta(sj, t, transition_probs, emission_probs, observed_sequence, table, previous_best_table):  # dp
    if t == 0:
        table[t][sj] = log(emission_probs[(sj, observed_sequence[t])])

    if sj in table[t]:
        return table[t][sj]  # reduce duplicate calculation
    maxDelta = -math.inf  # initialize
    maxState = ''  # phi
    for si in states:
        previous_delta = calculate_delta(si, t - 1, transition_probs, emission_probs, observed_sequence, table,
                                         previous_best_table) + \
                         log(transition_probs[(si, sj)]) + log(emission_probs[(sj, observed_sequence[t])])
        if previous_delta > maxDelta:
            maxState = si
            maxDelta = max(previous_delta, maxDelta)
    table[t][sj] = maxDelta
    previous_best_table[t][sj] = maxState
    return maxDelta


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float],
            emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    observed_sequence = ['B'] + observed_sequence + ['Z']  # create the complete sequence
    memoization_table = [{} for _ in range(len(observed_sequence))]
    previous_best_table = [{} for _ in range(len(observed_sequence))]

    for si in states:
        calculate_delta(si, len(observed_sequence) - 1, transition_probs, emission_probs, observed_sequence,
                        memoization_table, previous_best_table)

    res = [None] * (len(observed_sequence) - 2)
    s = 'Z'  # backtrace
    for t in range(len(res)):
        s = previous_best_table[len(observed_sequence) - 1 - t][s]  # 368
        res[-1 - t] = s  # start from the last one
    return res


def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    Truth = Sys = 0
    for (pr, tr) in zip(pred, true):
        for i in range(len(pr)):
            if pr[i] == tr[i] and pr[i] == 1:
                Truth += 1
            if pr[i] == 1:
                Sys += 1
    return Truth / Sys


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    correct = Truth = 0
    for (p, t) in zip(pred, true):
        # if not len(p) == len(t): print(f"pred and true length not equal {len(p)}, {len(t)}")
        for i in range(len(p)):
            if p[i] == t[i] and p[i] == 1:
                correct += 1
            if t[i] == 1:
                Truth += 1
    return correct / Truth


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over **all** the test sequences.
    """
    precision = precision_score(pred, true)
    recall = recall_score(pred, true)

    return precision * recall * 2 / (recall + precision) if precision + recall > 0 else 0


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    n = 10
    dic = {}
    folds = []
    each_len = int(len(data) / n)
    random.shuffle(data)
    for i in range(n):
        folds.append(data[i * each_len:i * each_len + each_len])

    for fold in folds:
        training_data = []
        for train_da in folds:
            if not fold == train_da:
                training_data += train_da

        transition_prob_table, emission_prob_table = estimate_hmm(training_data)

        # get all predictions and truth for the test fold
        each_pred = []
        each_true = []
        for seq in fold:
            # get prediction for one observed sequence
            pred = viterbi(seq['observed'], transition_prob_table, emission_prob_table)
            each_pred.append(pred)
            each_true.append(seq['hidden'])

        fold_pred = [[1 if x == 'W' else 0 for x in p] for p in each_pred]
        fold_true = [[1 if x == 'W' else 0 for x in t] for t in each_true]

        precision = precision_score(fold_pred, fold_true)
        recall = recall_score(fold_pred, fold_true)
        f1 = f1_score(fold_pred, fold_true)

        dic['recall'] = dic.get('recall', 0) + recall / n
        # default -> set to 0
        dic['precision'] = dic.get('precision', 0) + precision / n
        dic['f1'] = dic.get('f1', 0) + f1 / n
    return dic


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")


if __name__ == '__main__':
    main()
