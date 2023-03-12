'''Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke oznacene kao spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
a) Izracunajte koliki je prosjecan broj rijeci u SMS porukama koje su tipa ham, a koliko je ˇ
prosjecan broj rijeci u porukama koje su tipa spam. ˇ
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ? '''

file = open("SMSSpamCollection.txt")
hamMessages = []
spamMessages = []
hamWordCount = 0
spamWordCount = 0

spamExclamationMarkCount = 0

for line in file:
    line = line.strip().split()
    if line[0] == "ham":
        hamMessages.append(line[1:])
        hamWordCount += len(line[1:])
    else:
        spamMessages.append(line[1])
        spamWordCount += len(line[1:])
        if line[-1].endswith("!"):
            spamExclamationMarkCount += 1

print(
    f"Prosjecan broj rijeci u ham poruci: {float(hamWordCount)/len(hamMessages)}")
print(
    f"Prosjecan broj rijeci u spam poruci: {float(spamWordCount)/len(spamMessages)}")

print(
    f"broj spam poruka koje završavaju uskličnikom: {spamExclamationMarkCount}")
