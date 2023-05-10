# 2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku ˇ
# varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih ˇ
# velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate. ˇ
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
# vozila radi?
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# a)

data = pd.read_csv('data_C02_emission.csv')

inputColumnsNumeric = ['Model',
                       'Engine Size (L)',
                       'Cylinders',
                       'Fuel Consumption City (L/100km)',
                       'Fuel Consumption Hwy (L/100km)',
                       'Fuel Consumption Comb (L/100km)',
                       'Fuel Consumption Comb (mpg)',
                       'D', 'E', 'X', 'Z'
                       ]

outputColumn = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data[ohe.categories_[0]]=X_encoded 
#kategorije su imena stupaca, pohranjene u listu lista kategorija atributa categories_, zato [0], tj. jedina lista kategorija iz prosle linije dobivena


X = data[inputColumnsNumeric].to_numpy()
y = data[outputColumn].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


linearModel = lm.LinearRegression()
linearModel.fit(X_train[:,1:], y_train)

y_test_prediction = linearModel.predict(X_test[:,1:])

print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))

# 
y_error = abs(y_test_prediction - y_test)
max_error_index = np.argmax(y_error)
print("Maksimalna pogreska(MAE):", y_error[max_error_index])
print("Model s najvecom greskom:", X_test[max_error_index,0])
print("Predvidjeno:",y_test_prediction[max_error_index],"\nstvarno:", y_test[max_error_index])

