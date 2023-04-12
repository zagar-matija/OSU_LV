import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#a
data = pd.read_csv("data_C02_emission.csv")
#plt.hist(data["CO2 Emissions (g/km)"])
#plt.xlabel('co2 emisija')
#plt.ylabel('broj auta')
#plt.show()

##b
#colorDict = {'X': 'green', 'Z':"red", 'D':"blue", 'E':"black", 'N':"yellow"}
#plt.scatter(data['Fuel Consumption City (L/100km)'], data["CO2 Emissions (g/km)"], c=[colorDict[x] for x in data['Fuel Type']])
#plt.show()

##c

#df = data[["Fuel Consumption Hwy (L/100km)", "Fuel Type"]]

#df.boxplot(by='Fuel Type')
#plt.show()

##d
#df = data.groupby("Fuel Type")
#df['Make'].count().plot(kind="bar")
#plt.show()

#e
df = data.groupby("Cylinders")
df['CO2 Emissions (g/km)'].mean().plot(kind="bar")
plt.show()