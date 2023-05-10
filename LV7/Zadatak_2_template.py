'''Zadatak 7.5.2 Kvantizacija boje je proces smanjivanja broja razlicitih boja u digitalnoj slici, ali
uzimajuci u obzir da rezultantna slika vizualno bude što slicnija originalnoj slici. Jednostavan
nacin kvantizacije boje može se postici primjenom algoritma K srednjih vrijednosti na RGB
vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
kvantizacije i koja sadrži samo 5 boja koje su odredene algoritmom K srednjih vrijednosti.

1. Otvorite skriptu zadatak_2.py. Ova skripta ucitava originalnu RGB sliku test_1.jpg
te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu 7.2 pri cemu je  n
broj elemenata slike, a m je jednak 3. Koliko je razlicitih boja prisutno u ovoj slici?
2. Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima
elemenata originalne slike.
3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom.
4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
rezultate.
5. Primijenite postupak i na ostale dostupne slike.
6. Graficki prikažite ovisnost ˇ J o broju grupa K. Koristite atribut inertia objekta klase
KMeans. Možete li uociti lakat koji upu ˇ cuje na optimalni broj grupa?
7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
primjecujete?'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
#plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()
plt.imshow(img)
plt.tight_layout()
#plt.show()

#1
unique_values=np.unique(img_array_aprox)
print("unikatnih boja: ",unique_values.size)

#2
# distortions = []
# Ks=range(1,11)
# for k in Ks:
#     km = KMeans (n_clusters =k, init ='random', n_init =5, random_state =0 )
#     km.fit(img_array_aprox)
#     distortions.append(km.inertia_)
# plt.plot(Ks, distortions)
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
#mozemo primjetiti da vec s 2 boje mozemo vjerno prikazati sliku

km = KMeans(n_clusters =2, init ='random', n_init =5, random_state =0 )
km.fit(img_array_aprox)
predicted_values = km.predict(img_array_aprox)



#3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom.
groupedImg = km.cluster_centers_[predicted_values]
groupedImg = np.reshape(groupedImg, img.shape)
plt.imshow(groupedImg)
plt.show()

#4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene rezultate.

for i in range(1,6):
    km = KMeans(n_clusters =i, init ='random', n_init =5, random_state =0 )
    km.fit(img_array_aprox)
    predicted_values = km.predict(img_array_aprox)
    groupedImg = km.cluster_centers_[predicted_values]
    groupedImg = np.reshape(groupedImg, img.shape)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img)
    axs[1].imshow(groupedImg)
    plt.title(f"Broj boja: {i}")
    plt.show()

#5. Primijenite postupak i na ostale dostupne slike.
for i in range (2, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")        
    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    newimg = km.cluster_centers_[labels]
    newimg = np.reshape(newimg, (img.shape))
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    axarr[1].imshow(newimg)
    plt.show()

#6. Graficki prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase
#KMeans. Možete li uociti lakat koji upucuje na optimalni broj grupa?
img = Image.imread("imgs\\test_1.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()
Ks = range(1, 11)
Js = []
for i in Ks:
    km = KMeans(n_clusters=i, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    Js.append(km.inertia_)
plt.plot(Ks, Js)
plt.show()

#7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
#primjecujete?

img = Image.imread("imgs\\test_1.jpg")

img = img.astype(np.float64) / 255

w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

img_array_aprox = img_array.copy()

km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)

km.fit(img_array_aprox)

labels = km.predict(img_array_aprox)

unique_labels = np.unique(labels)
print(unique_labels)

f, axarr = plt.subplots(2, 2)

for i in range(len(unique_labels)):
    values = labels==[i]
    bit_img = np.reshape(values, (img.shape[0:2]))
    bit_img = bit_img*1
    x=int(i/2)
    y=i%2
    axarr[x, y].imshow(bit_img)

plt.show()
