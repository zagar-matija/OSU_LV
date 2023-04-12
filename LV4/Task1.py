from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot as plt

# a)

data = pd.read_csv('data_C02_emission.csv')

inputColumns = ['Engine Size (L)',
                'Cylinders',
                'Fuel Consumption City (L/100km)',
                'Fuel Consumption Hwy (L/100km)',
                'Fuel Consumption Comb (L/100km)',
                'Fuel Consumption Comb (mpg)']
outputColumn = ['CO2 Emissions (g/km)']

X = data[inputColumns].to_numpy()
y = data[outputColumn].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# b)

inputTrain = X_train[:, 0]
inputTest = X_test[:, 0]
plt.scatter(inputTrain, y_train, c='#0000ff')
plt.scatter(inputTest, y_test, c='#FF0000')
plt.xlabel('Engine size (L)')
plt.ylabel('CO2 Emission (g/km)')
plt.show()

# c)

scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

plt.hist(X_train[:, 0],  bins=15)
plt.title('Unscaled engine size histogram')
plt.show()

plt.hist(X_train_transformed[:, 0], bins=15)
plt.title('Unscaled engine size histogram')
plt.show()

# d)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_transformed, y_train)

print(linearModel.coef_)
# koeficijenti ovog modela su vrijednosti theta parametara u linearnoj fuinkciji opisanoj u 4.6.

# e)

y_test_prediction = linearModel.predict(X_test_transformed)

plt.scatter(y_test, y_test_prediction, c='#0000ff', s=2)
plt.xlabel('stvarna vrijednost izlaza')
plt.ylabel('predikcija izlaza')
plt.show()

# f)

print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))

# g)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_transformed[:, 2:], y_train)
y_test_prediction = linearModel.predict(X_test_transformed[:, 2:])

print('Metrike nakon izostavljanja prva dva ulazna parametra:')
print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))

# metrike pokazuju da je novi model losije fittao podatke
