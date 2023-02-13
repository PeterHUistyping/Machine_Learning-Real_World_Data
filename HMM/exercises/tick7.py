from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    states = ['F', 'W', 'B', 'Z']
    dic = {}
    for state1 in states:
        for state2 in states:
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


def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[
    Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    observations = ['1', '2', '3', '4', '5', '6', 'B', 'Z']
    states = ['F', 'W', 'B', 'Z']
    dic = {}
    for state1 in states:
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


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)


if __name__ == '__main__':
    main()
