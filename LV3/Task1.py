'''Zadatak 3.4.1 Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja: 
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili 
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip 
category.
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
ime proizvodaca, model vozila i kolika je gradska potrošnja. 
c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija 
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosjecna emisija C02 
plinova automobila proizvodaca Audi koji imaju 4 cilindara? 
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na 
broj cilindara?
f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila 
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? 
h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)? 
i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.'''

import numpy as np
import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#a)
print("Broj mjerenja: ", len(data))
print("Tipovi podataka: ", data.info())
print("Broj izostalih vrijednosti: ", data.isnull().sum())
print("Broj dupliciranih vrijednosti: ", data.duplicated().sum())
data.dropna(axis=0)
data.drop_duplicates()
data = data.reset_index(drop = True)

for col in data.columns[data.dtypes == "object"]:
    data[col] = data[col].astype("category")

#b)
'''Koja tri automobila imaju najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
ime proizvodaca, model vozila i kolika je gradska potrošnja. '''

sorted = data.sort_values(by=['Fuel Consumption City (L/100km)'])

print("Najveca potrosnja: \n", sorted[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))
print("Najmanja potrosnja: \n", sorted[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))

#c)
'''Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija 
plinova za ova vozila?'''

print("Broj vozila izmedu 2.5 i 3.5 l motorima: ", len(data[(data['Engine Size (L)']>2.5) & (data['Engine Size (L)']<3.5)]))

#d)
'''d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02 
plinova automobila proizvodaca Audi koji imaju 4 cilindara? '''

print("Broj auta proizvodaca Audi: ", len(data["Make"]=="Audi"))
print("Prosjecna emisija C02 plinova auta proizvodaca Audi s 4 cilindra: ", data[(data["Make"]=="Audi") & (data['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean())

#e)
'''Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na 
broj cilindara?'''

print("Broj auta s parnim brojem cilindara : ", len(data[data["Cylinders"]%2==0]))

print("Prosjecna emisija C02 plinova s obzirom na cilindre: ", data.groupby(by="Cylinders")['CO2 Emissions (g/km)'].mean())

#f)[['Fuel Type','Fuel Consumption City (L/100km)']].mean()
'''Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila 
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?'''

print("Prosjecna gradska potrošnja s obzirom na gorivo: ", data[(data['Fuel Type'] == 'X') | (data['Fuel Type'] == 'D')].groupby(by="Fuel Type")[['Fuel Type', 'Fuel Consumption City (L/100km)']].mean())
print("Medijalna gradska potrošnja s obzirom na gorivo: ", data[(data['Fuel Type'] == 'X') | (data['Fuel Type'] == 'D')].groupby(by="Fuel Type")[['Fuel Type', 'Fuel Consumption City (L/100km)']].median())



'''
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? '''

print("vozilo s 4 cilindra koje koristi dizelski motor i ima najvecu gradsku potrošnju goriva : ", data[(data['Cylinders']==4) & (data['Fuel Type']=='D')].sort_values(by=["Fuel Consumption City (L/100km)"]).tail(1))


#h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)? 

print("Broj auta s rucni tip mjenjaca : ", len(data[data["Transmission"].str.startswith("M")]))

'''
i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.'''

print("korelacije velicina", data.corr(numeric_only = True))