# Machine Learning and Real World data
https://www.cl.cam.ac.uk/teaching/2122/MLRD/ \
PyCharm
## Dependency
NLTK-Natural Language Toolkit
## Sentiment Lexicon Database
Given a list of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether the sentiment of each review in the test set is positive or negative based on whether there are more positive or negative words.\
Classification: label Lexicons into postive and negative.\
Evaluation: Based on reviews on IMDb. \
Improve the classifier using thresholds for decision bounds.
```
Your accuracy: 0.6355555555555555
Your improved accuracy: 0.6888888888888889
```
## Naive Bayes Classifier
Parameter estimation\
How to deal with a word in a review was not present in the training dataset?\ 
Ignore its contribution or using add-one (Laplace) Smoothing 
```
Your accuracy using simple classifier: 0.63
Your accuracy using unsmoothed probabilities: 0.49
Your accuracy using smoothed probabilities: 0.795
```
## Zipf’s Law and Heaps’ Law
Zipf’s law says that there is a reverse exponential relationship between the frequency of a word (fw) in a large natural language text, and its relative frequency rank (rw; the ranking of its frequency in comparison with other words’ frequencies) \
![Zipf](mlrd/figures/sentiment_detection/Estimation%20of%20log-log.png)
Heaps’ law relates the number of distinct words in a text to the overall number of words in the text.
![Heaps](mlrd/figures/sentiment_detection/Numbers%20of%20Words.png)

## Statistical Significance Testing
Modify the simple classifier to include the information about the magnitude of a sentiment.\
A word with a strong intensity should be weighted *four* times as high for the evaluator.\
Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than     another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least common sign total should be used to calculate the probability.
```
The p-value of the two-sided sign test for classifier_a "classifier simple" and classifier_b "classifier magnitude": 0.6722499772048186
The p-value of the two-sided sign test for classifier_a "classifier magnitude" and classifier_b "naive bayes classifier": 0.07683763213126037
```