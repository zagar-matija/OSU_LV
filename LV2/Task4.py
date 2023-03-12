'''Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno ˇ
bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
u odgovarajuci oblik koristite numpy funkcije ´ hstack i vstack'''


import numpy as np
import matplotlib.pyplot as plt

whiteSquare = np.ones((50,50))*255
blackSquare = np.zeros((50,50))

image = np.hstack((blackSquare,whiteSquare))
image = np.vstack((image, np.hstack((whiteSquare,blackSquare))))

plt.imshow(image, cmap="gray")
plt.show()