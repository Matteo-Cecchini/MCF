import somme

a = int(input())
b = somme.somma(a)
print(b)

c = somme.sommaRad(a)
print(c)

d = somme.sumNDot(a)
print(d)

e = somme.sumPow(a, 0)
print(e)

print(somme.somma.__doc__)