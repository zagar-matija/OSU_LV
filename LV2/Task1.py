import numpy as np
import matplotlib.pyplot as plt

'''Pomocu funkcija ´ numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
sl.'''

x= [1,3,3,2,1]
y= [1,1,2,2,1]
plt.plot(x,y, marker=".", linewidth=2, color="g", markersize=10)
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("Primjer")
plt.axis ([0 ,4 ,0 , 4])
plt.show()