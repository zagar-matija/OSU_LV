'''
6.5.3 Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje? ˇ
Mijenjajte tip kernela koji se koristi. Što primjecujete? ´

6.5.4 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ C i γ
algoritma SVM za problem iz Zadatka 1.'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn . model_selection import cross_val_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

'''6.5.2 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra  K
algoritma KNN za podatke iz Zadatka 1.'''

KNN_model = KNeighborsClassifier ()
k_range = list(range(1, 101))
param_grid = dict(n_neighbors=k_range)

knn_gscv = GridSearchCV (KNN_model, param_grid, scoring ='accuracy')

knn_gscv.fit ( X_train_n , y_train)
print ( knn_gscv . best_params_)
print ( knn_gscv . best_score_ )
print ( knn_gscv . cv_results_ )


#zad3
SVM_model = svm.SVC(kernel='rbf', gamma=1, C=0.1)
SVM_model.fit(X_train_n, y_train)

y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost(SVM): " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

#zad4
param_grid = {'C': [0.1, 1, 10, 100, 1000],
                'gamma': [100, 10, 1, 0.1, 0.01]}

SVM_model_default = svm.SVC()

svm_gscv = GridSearchCV(estimator=SVM_model_default, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)

svm_gscv.fit(X_train_n, y_train)

print(svm_gscv.best_params_)
#print(svm_gscv.best_score_)
#print(svm_gscv.cv_results_)
