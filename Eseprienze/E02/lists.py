from datetime import datetime, timedelta

weekdays = {0:'Domenica', 1:'Lunedì', 2:'Martedì', 3:'Mercoledì', 4:'Giovedì', 5:'Venerdì', 6:'Sabato'}
giorni1 = [weekdays[i] for i in range(7)]
print(giorni1, "\n")

date0 = datetime.strptime("11.09.2003", "%d.%m.%Y")
datenow = datetime.now()
datestart = datenow - timedelta(days=datenow.day - 1)

month_lenght = 1
giorni_num = []
while datestart.month == datenow.month:
    giorni_num.append(datestart.weekday())
    datestart += timedelta(days=1)
    month_lenght += 1
    
print(giorni_num, "\n")
    
giorni2 = [weekdays[i] for i in giorni_num]
giorni3 = dict(zip(range(1, month_lenght), giorni2))

print(giorni2, "\n")
print(giorni3)