'''Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je masa u kg, a treci´
stupac polja je visina u cm.
a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
b) Prikažite odnos visine i mase osobe pomocu naredbe  matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1)'''

import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt("data.csv",delimiter=",", dtype=float, skiprows=1)

#a)
print("Broj osoba:", arr.shape[0])

#b)
plt.scatter(arr[:,2], arr[:,1])
plt.xlabel("visina")
plt.ylabel("težina")
plt.title("Odnos visine i težine svih osoba")
plt.show()

#c)
plt.scatter(arr[:,2][::50], arr[:,1][::50])
plt.xlabel("visina")
plt.ylabel("težina")
plt.title("Odnos visine i težine svake 50. osobe")
plt.show()

#d)
print("minimalna visina:", min(arr[:,1]))
print("maksimalna visina:", max(arr[:,1]))
print("prosjecna visina:", arr[:,1].mean())

#e)
women = arr[np.where(arr[:,0] == 0)]

print("prosjecna visina žene:", women[:,1].mean())

men = arr[np.where(arr[:,0] == 1)]

print("prosjecna visina muškarca:", men[:,1].mean())