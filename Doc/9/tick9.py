from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score

from typing import List, Dict, Tuple


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    pass


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    pass


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    pass


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    pass




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
    pass



def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    pass


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
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

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
