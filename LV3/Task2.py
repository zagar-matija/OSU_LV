import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#a
data = pd.read_csv("data_C02_emission.csv")
plt.hist(data["co2 emissions (g/km)"])
plt.xlabel('co2 emisija')
plt.ylabel('broj auta')
plt.show()

#b
colordict = {'x': 'green', 'z':"red", 'd':"blue", 'e':"black", 'n':"yellow"}
plt.scatter(data['fuel consumption city (l/100km)'], data["co2 emissions (g/km)"], c=[colordict[x] for x in data['fuel type']])
plt.show()

#c

df = data[["fuel consumption hwy (l/100km)", "fuel type"]]

df.boxplot(by='fuel type')
plt.show()

#d
df = data.groupby("fuel type")
df['make'].count().plot(kind="bar")
plt.show()

#e
df = data.groupby("Cylinders")
df['CO2 Emissions (g/km)'].mean().plot(kind="bar")
plt.show()