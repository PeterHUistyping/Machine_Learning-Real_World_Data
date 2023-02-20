import math

from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score, viterbi

from typing import List, Dict, Tuple

hidden_states = ['B', 'Z', 'i', 'o', 'M']
observations = ['B,', 'Z', 'P', 'A', 'T', 'K', 'S', 'N', 'L', 'Q', 'M', 'G', 'Z', 'B', 'D', 'H', 'I', 'C', 'W', 'E',
                'R', 'V', 'Y', 'F']


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    dic = {}
    for state1 in hidden_states:
        for state2 in hidden_states:
            state_tuple = (state1, state2)
            if state1 == 'B' and state2 == 'Z':
                dic[state_tuple] = 0
            elif state1 == 'Z':
                dic[state_tuple] = 0
            elif state2 == 'B':
                dic[state_tuple] = 0
            else:
                numerator = 0
                denominator = 0
                for sequence in hidden_sequences:
                    previous = None  # for each sequence of FW, reset previous as None
                    for s in sequence:
                        if s == state2 and previous == state1:  # F->W
                            numerator += 1
                        if s == state1:  # F->
                            denominator += 1
                        previous = s
                dic[state_tuple] = numerator / denominator
    return dic


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[
    Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    dic = {}
    for state1 in hidden_states:
        for observation2 in observations:
            info_tuple = (state1, observation2)
            numerator = 0
            denominator = 0
            for (s_singleseq, o_singleseq) in zip(hidden_sequences, observed_sequences):
                for (s, o) in zip(s_singleseq, o_singleseq):
                    if s == state1 and o == observation2:
                        numerator += 1
                    if s == state1:
                        denominator += 1
            dic[info_tuple] = numerator / denominator
    return dic


def estimate_hmm_bio(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


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
    for si in hidden_states:
        previous_delta = calculate_delta(si, t - 1, transition_probs, emission_probs, observed_sequence, table,
                                         previous_best_table) + \
                         log(transition_probs[(si, sj)]) + log(emission_probs[(sj, observed_sequence[t])])
        if previous_delta > maxDelta:
            maxState = si
            maxDelta = max(previous_delta, maxDelta)
    table[t][sj] = maxDelta
    previous_best_table[t][sj] = maxState
    return maxDelta


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float],
                emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    observed_sequence = ['B'] + observed_sequence + ['Z']  # create the complete sequence
    memoization_table = [{} for _ in range(len(observed_sequence))]
    previous_best_table = [{} for _ in range(len(observed_sequence))]

    for si in hidden_states:
        calculate_delta(si, len(observed_sequence) - 1, transition_probs, emission_probs, observed_sequence,
                        memoization_table, previous_best_table)

    res = [None] * (len(observed_sequence) - 2)
    s = 'Z'  # backtrace
    for t in range(len(res)):
        s = previous_best_table[len(observed_sequence) - 1 - t][s]  # 368
        res[-1 - t] = s  # start from the last one
    return res


def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
                      unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    scores = []
    additional_data = []

    for iterate in range(0, num_iterations + 1):  # at least 1
        predictions = []  # on unlabeled data ; almost same as main()
        transition_probs, emission_probs = estimate_hmm_bio(training_data + additional_data)
        for sample in unlabeled_data:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        additional_data = [{'observed': o, 'hidden': pred} for (o, pred) in zip(unlabeled_data, predictions)]

        dev_predictions = []
        dev_observed_sequences = [x['observed'] for x in dev_data]
        dev_hidden_sequences = [x['hidden'] for x in dev_data]
        for sample in dev_observed_sequences:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            dev_predictions.append(prediction)
        predictions_binarized = [[1 if x == 'M' else 0 for x in pred] for pred in dev_predictions]
        dev_hidden_sequences_binarized = [[1 if x == 'M' else 0 for x in dev] for dev in dev_hidden_sequences]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

        print(f"Precision for iteration {iterate} : {p}")
        print(f"Recall for iteration {iterate} : s{r}")
        print(f"F1 for iteration {iterate} : {f1}")
        print("\n")

        scores.append({'recall': r, "precision": p, "f1": f1})
    return scores


def visualize_scores(score_list: List[Dict[str, float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    recall_sc = [(i, score_list[i]['recall']) for i, iter in enumerate(score_list)]
    precision_sc  = [(i, score_list[i]['precision']) for i, iter in enumerate(score_list)]
    f1_sc = [(i, score_list[i]['f1']) for i, iter in enumerate(score_list)]
    chart_plot(recall_sc, "recall score", "iteration", "recall_sc")
    clean_plot()
    chart_plot(precision_sc, "precision score", "iteration", "precision_sc")
    clean_plot()
    chart_plot(f1_sc, "F1 score", "iteration", "F1_sc")
    clean_plot()


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x == 'M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)


if __name__ == '__main__':
    main()
