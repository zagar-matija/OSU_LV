'''Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovarajucu poruku.'''

numbers = []


while True:
    userInput = input()
    if userInput == "Done":
        break
    try:
        userInput = float(userInput)
    except ValueError:
        print("Krivi unos.")
        continue
    numbers.append(userInput)

print(f"Broj unosa: {len(numbers)}")

if numbers != []:
    print(f"Minimum liste: {min(numbers)}")
    print(f"Maksimum liste: {max(numbers)}")
    print(f"Prosjek liste: {sum(numbers)/len(numbers)}")

    numbers.sort()
    print(f"Sortirana lista: {numbers}")
else:
    print("Lista prazna.")
