import math
from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from typing import List, Tuple, Callable
from utils.sentiment_detection import read_tokens
import os
import glob


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words.

    Use this to create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    chart_plot(token_frequencies_log, "log-log graph of The 10,000 highest-ranked words, ", "Frequency (log)",
               "Rank (log)")
    [m, c] = best_fit(token_frequencies_log, token_frequencies)
    chart_plot([(x, m * x + c) for x in range(0, 5)], "Estimation of log-log", "Frequency (log)", "Rank (log)")

    clean_plot()

    def cal_fre_from_rank(rank: int):
        rank = math.log10(rank)
        return m * rank + c

    return cal_fre_from_rank


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    reviews = glob.glob(os.path.join(dataset_path, '*'))
    # reviews = glob.glob('data/sentiment_detection/reviews/reviews/*')
    reviews.sort()
    list_of_frequencies = [tuple()]
    dict_token = dict()
    for x in reviews:
        tokenized_data = read_tokens(x)
        for token in tokenized_data:
            if token in dict_token:
                dict_token[token] += 1
            else:
                dict_token[token] = 1
    list_of_frequencies = [(k, v) for k, v in dict_token.items()]
    list_of_frequencies.sort(key=lambda a: a[1], reverse=True)
    return list_of_frequencies


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    i = 1
    data = []
    for fre in frequencies:
        if i <= 10000:
            data.append((i, fre[1]))
            i += 1
    chart_plot(data, "The 10,000 highest-ranked words", "Frequency", "Rank")
    clean_plot()


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    i = 1
    data = []
    task1 = []
    chosen = ["best", "well", "satisfying", "bland", "lacking", "ironic", "escapist", "fun", "like", "gory"]
    for fre in frequencies:
        for word in chosen:
            if fre[0] == word:
                task1.append((i, fre[1]))
        if i <= 10000:
            data.append((i, fre[1]))
            i += 1

    chart_plot(data, "The 10,000 highest-ranked words", "Frequency", "Rank")
    chart_plot(task1, "Selected word", "Frequency", "Rank")
    clean_plot()


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    i = 1
    data = []
    data2 = []
    for fre in frequencies:
        if i <= 10000:
            data.append((math.log10(i), math.log10(fre[1])))
            data2.append((i, fre[1]))
            i += 1
    chart_plot(data, "log-log graph of The 10,000 highest-ranked words, ", "Frequency (log)", "Rank (log)")
    [m, c] = best_fit(data, data2)
    chart_plot([(x, m * x + c) for x in range(0, 5)], "Estimation of log-log", "Frequency (log)", "Rank (log)")

    clean_plot()


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    reviews = glob.glob('data/sentiment_detection/reviews/reviews/*')
    reviews.sort()
    list_of_frequencies = [tuple()]
    dict_token = dict()
    count=0
    i=0
    list_of_frequencies = []
    for x in reviews:
        tokenized_data = read_tokens(x)
        for token in tokenized_data:
            if token in dict_token:
                dict_token[token] += 1
            else:
                dict_token[token] = 1
            count+=1
            if count == 2<<i:
                list_of_frequencies.append((count,len(dict_token)))
                i+=1
    return list_of_frequencies


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    chart_plot(type_counts,"Numbers of Words","Total words",'Distinct words')


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
