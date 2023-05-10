'''Zadatak 5.5.1 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
ulazne velicine. Podaci su podijeljeni na skup za ucenje i skup za testiranje modela. 
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a) Prikažite podatke za ucenje u ˇ x1 -x2 ravnini matplotlib biblioteke pri cemu podatke obojite ˇ
#s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
#marker (npr. 'x'). Koristite funkciju scatter koja osim podataka prima i parametre c i
#cmap kojima je moguce definirati boju svake klase.

plt.scatter(X_train[:,0],X_train[:,1], c=y_train, s=15, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.scatter(X_test[:,0],X_test[:,1], marker="x", c=y_test, s=15, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.show()

#b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa poda- ´
#taka za ucenje. 

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)


#c) Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke ¯
#naucenog modela u ravnini x1 - x2 zajedno s podacima za ucenje. Napomena: granica ˇ
#odluke u ravnini x1 -x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.

th0  = LogRegression_model.intercept_[0]
th1 = LogRegression_model.coef_[0,0]
th2 = LogRegression_model.coef_[0,1]
a=-th0/th1
b=-th2/th1

x_lin = np.linspace(-4,4,100)
y_lin = b*x_lin+a
plt.plot(x_lin, y_lin)
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.scatter(X_train[:,0],X_train[:,1], c=y_train, s=15, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.show()


#d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke 
#regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunate tocnost, 
#preciznost i odziv na skupu podataka za testiranje.

y_test_p = LogRegression_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Precision: %.3f' % precision_score(y_test, y_test_p))
print('Recall: %.3f' % recall_score(y_test, y_test_p))
print('Accuracy: %.3f' % accuracy_score(y_test, y_test_p))


#e) Prikažite skup za testiranje u ravnini x1-x2. Zelenom bojom oznacite dobro klasificirane
#primjere dok pogrešno klasificirane primjere oznacite crnom bojom

plt.scatter(X_test[:,0],X_test[:,1], marker="o", c=y_test==y_test_p, s=25, cmap=mcolors.ListedColormap(["black", "green"]))
plt.show()