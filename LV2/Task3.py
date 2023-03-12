'''
Skripta zadatak_3.py ucitava sliku 'road.jpg'. Manipulacijom odgovarajuce
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu cetvrtinu slike po širini, ˇ
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku
'''

import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("road.jpg")
image=image[:,:,0]

#a)
plt.imshow(image, cmap="gray")
plt.show()
brightened= np.where(image<200, image +55, image)
plt.imshow(brightened, cmap="gray")
plt.show()

#b)
cut = image[:, round(len(image[:,1])/4):round(len(image[:,1])/2)]
plt.imshow(cut, cmap="gray")
plt.show()

#c)

rotated = np.rot90(np.rot90(np.rot90(image)))
plt.imshow(rotated, cmap="gray")
plt.show()

#d)

mirrored = np.flip(image, axis=1)
plt.imshow(mirrored, cmap="gray")
plt.show()