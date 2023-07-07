from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

#ucitavanje modela
model = load_model('Model/')
model.summary()

#ucitavanje slike, u MNIST-u su brojevi bijelom bojom na crnoj pozadini pa je potrebna zamjena vrijednosti crne i bijele boje jer je ova slika crna olovka bijela pozadina
img = plt.imread('test.png')[:,:,0]*255   #bude izmedu 0 i 1 pa treba mnozit s 255, bude rgb, pa treba odstranit nepotrebno jer je crno bijelo
img = img.astype('uint8')
img = np.where(img != 255, 255, 0)       #zamjena, da 255 simbolizira gdje nesto pise, a 0 gdje nema nista (kada ucita je obrnuto jer je crni tekst(0) na bijeloj pozadini(255))
img_reshaped = np.reshape(img, (1,img.shape[0]*img.shape[1]))    #mora biti shape (n, broj ulaznih velicina), tj. u ovom slucaju (1,784)

#predikcija
img_prediction = model.predict(img_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (1,10)
img_prediction = np.argmax(img_prediction, axis=1) #izdvaja max index u svakom retku 

#prikaz slike
plt.imshow(img)
plt.title(f'Stvarni broj:2, predikcija:{img_prediction[0]}')
plt.show()