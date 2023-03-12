# Zadatak 1.4.1 Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen
# po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
# je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
# izracunavate u zasebnoj funkciji naziva total_euro.
# Primjer:
# Radni sati: 35 h
# eura/h: 8.5
# Ukupno: 297.5 eura


hoursWorked = input("Radni sati:  ")
hoursWorked = int(hoursWorked.split()[0])

hourPrice = float(input("eura/h: "))

totalPay = hoursWorked*hourPrice

print(f"Ukupno: {totalPay}")
