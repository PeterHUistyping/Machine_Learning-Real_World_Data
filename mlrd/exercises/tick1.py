from typing import List, Dict
import os
from utils.sentiment_detection import load_reviews, read_tokens


def read_lexicon(filename: str) -> Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    with open(filename, "r") as f2:  # a resource is "cleaned up"
        dictionary = {}
        for line in f2.readlines():
            each_item = line.split()
            index1 = each_item[0].find("=") + 1
            index2 = each_item[2].find("=") + 1
            # print(line)
            if each_item[2][index2:] == "negative":
                dictionary[each_item[0][index1:]] = -1
                # print("-1", each_item[0][index:])
            else:
                dictionary[each_item[0][index1:]] = 1
                # print("1", each_item[0][index:])
    return dictionary


def predict_sentiment(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the
    test set is positive or negative based on whether there are more positive or negative words.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    score_temp = 0
    for each_review in review:
        if lexicon.get(each_review):
            score_temp += lexicon.get(each_review)
        else:
            score_temp += 0  # DEFAULT
    if score_temp >= 0:
        return 1
    else:
        return -1


def accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    length_ = min(len(pred), len(true))
    correct = 0
    for i in range(length_):
        if pred[i] == true[i]:
            correct += 1
    return correct / length_


def read_intensity(filename: str) -> Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    with open(filename, "r") as f2:  # a resource is "cleaned up"
        dictionary = {}
        for line in f2.readlines():
            each_item = line.split()
            index1 = each_item[0].find("=") + 1
            index2 = each_item[2].find("=") + 1
            index3 = each_item[1].find("=") + 1
            # print(line)
            if each_item[2][index2:] == "negative":
                if each_item[1][index3:] == "strong":
                    dictionary[each_item[0][index1:]] = -2
                else:
                    dictionary[each_item[0][index1:]] = -1
                # print("-1", each_item[0][index:])
            else:
                if each_item[1][index3:] == "strong":
                    dictionary[each_item[0][index1:]] = 2
                else:
                    dictionary[each_item[0][index1:]] = 1
                # print("1", each_item[0][index:])

    return dictionary


def predict_sentiment_improved(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    score_temp = 0
    for each_review in review:
        if lexicon.get(each_review):
            score_temp += lexicon.get(each_review)
        else:
            score_temp += 0  # DEFAULT change to 0
    if score_temp >= 10:
        return 1
    else:
        return -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [read_tokens(x['filename']) for x in review_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    print(f"Your accuracy: {acc1}")
    lexicon = read_intensity(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    print(f"Your improved accuracy: {acc2}")


if __name__ == '__main__':
    main()
