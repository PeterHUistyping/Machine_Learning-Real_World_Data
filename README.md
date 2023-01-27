# Machine Learning and Real World data
PyCharm
## Dependency
NLTK-Natural Language Toolkit
## Sentiment Lexicon Database
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