import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model.predict(X_test)
print(y_test)
print(y_test_p)

print('Precision: %.3f' % precision_score(y_test, y_test_p))
print('Recall: %.3f' % recall_score(y_test, y_test_p))
print('F1: %.3f' % f1_score(y_test, y_test_p))
print('Accuracy: %.3f' % accuracy_score(y_test, y_test_p))
