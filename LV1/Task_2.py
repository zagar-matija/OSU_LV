'''Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju 
sljedecih uvjeta: ´
>= 0.9 A
>= 0.8 B
>= 0.7 C
>= 0.6 D
< 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovarajucu poruku.'''

while True:
    value = input("Unesite ocjenu: ")

    try:
        value = float(value)
    except ValueError:
        print("Unos nije broj.")
        continue

    if value > 1.0 or value < 0:
        print("vrijednost mora biti u rasponu 0-1")
        continue
    break

if value >= 0.9:
    print("A")
elif value >= 0.8:
    print("B")
elif value >= 0.7:
    print("C")
elif value >= 0.6:
    print("D")
else:
    print("F")
