from datetime import datetime, timedelta
# data di nascita utente
birth = input("Inserire data di nascita formattata del tipo gg.mm.yyyy HH:MM:SS: ")
birth = datetime.strptime(birth, "%d.%m.%Y %H:%M:%S")
# data attuale
datenow = datetime.now()
# calcolo del tempo passato secondo format data
timepassed = datenow - birth
print("La tua età é:\n")

timediff = timedelta(days=timepassed.days, seconds=timepassed.seconds, microseconds=timepassed.microseconds)
print("in Anni", timediff.days // 365)
print("in Giorni", timepassed.days)
print("in Secondi", timepassed.total_seconds())