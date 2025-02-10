sm = 0
for i in range(1, 101):
    sm += i

print("La somma dei primi 100 numeri naturali é: ", sm)

# alternativa
sm = (100 + 1) * 100 // 2
print("La stessa somma ma con formula: ", sm)