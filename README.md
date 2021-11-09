# Classify Real and Fake News Headlines



## Datasets

The data are chosen 1298 [fake news headlines](https://www.kaggle.com/mrisdal/fake-news/data) and 1968 [real news headlines](https://www.kaggle.com/therohk/million-headlines). The data were cleaned by removing words from fake news titles that are not a part of the headline, removing special characters from the headlines, and restricting real news headlines to those after October 2016 containing the word “trump”.



## Training

- We train the decision tree classifier using 5 different values of maximum depth, as well as two different split criteria (information gain and Gini coefficient), and evaluate the performance of each one on the validation set.
- Hyperparameter tuning: pick the hyperparameter that achieves the highest accuracy on the validation dataset.



## Testing

The accuracy for Gini coefficient criterion with max_depth = 8: 73.06%
The accuracy for information gain criterion with max_depth = 8: 71.84%
The accuracy for Gini coefficient criterion with max_depth = 13: 76.73%
The accuracy for information gain criterion with max_depth = 13: 74.29%
The accuracy for Gini coefficient criterion with max_depth = 21: 77.96%
The accuracy for information gain criterion with max_depth = 21: 73.47%
The accuracy for Gini coefficient criterion with max_depth = 34: 80.0%
The accuracy for information gain criterion with max_depth = 34: 79.18%
The accuracy for Gini coefficient criterion with max_depth = 55: 80.0%
The accuracy for information gain criterion with max_depth = 55: 75.1%



## Compute Information Gain

Computes the information gain of the topmost split(keyword is "the") on the training data as well as several other keywords - "donald", "trumps", "hillary":

The information gain for attribute "the" is 0.051
The information gain for attribute "donald" is 0.0499
The information gain for attribute "trumps" is 0.0426
The information gain for attribute "hillary" is 0.0352



## Decision Tree Output

![Decision Tree](/Users/steveny/Desktop/PEY/cls-news-headlines/decision_tree.png)
