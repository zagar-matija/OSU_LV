'''Napišite Python skriptu koja ce ucitati tekstualnu datoteku naziva  song.txt.
Potrebno je napraviti rjecnik koji kao kljuceve koristi sve razlicite rijeci koje se pojavljuju u ˇ
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (kljuc) pojavljuje u datoteci. ˇ
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.'''

file = open("song.txt")

words = {}

for line in file:
    for word in line.split():
        if word in words.keys():
            words[word] += 1
        else:
            words[word] = 1

for word in words.keys():
    if words[word] == 1:
        print(word)
