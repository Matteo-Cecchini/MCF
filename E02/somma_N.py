n_int = input("Inserire numero naturale: ")

while type(n_int) != int:
    try:
        n_int = int(n_int)
    except:
        n_int = input("Non è un numero, inserire numero naturale: ")


print("Farò la somma dei primi ", n_int, " numeri naturali")

sm = 0
for i in range(1, n_int + 1):
    sm += i

print("La somma dei primi ", n_int, "numeri naturali é: ", sm)

# alternativa
if n_int % 2 == 0:
    sm = (n_int + 1) * n_int // 2
else:
    sm = (n_int**2 + n_int) // 2

print("La somma dei primi ", n_int, "numeri naturali con formula é: ", sm)