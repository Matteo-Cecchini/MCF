import numpy as np
import math


def somma(n):
    '''
    funzione che somma i primi numeri naturali sfruttando la formula di Gauss
    
    parametri
    --------------
        n: numero naturale di cui fare la somma nell' intervallo [1;n]
        
    retituisce
    --------------
        sum(i)_{i=1}^{n}
    '''
    return (n + 1) * (n / 2)

def sommaRad(n):
    '''
    funzione che somma le radici dei primi numeri naturali.
    Invece di fare un ciclo su tutti i numeri naturali, la funzione sfrutta numpy per generare un array
    con le radici dei numeri dell'intervallo selezionato, e ne fa la somma; la scelta deriva dall'osservazione che
    il ciclo prende più tempo più l'intervallo è grande, mentre l'efficienza di numpy, nonostante il costo in memoria,
    rimane con tempi ridotti anche oltre il milione di numeri.
    
    parametri
    --------------
        n: numero naturale di cui fare la somma delle radici nell' intervallo [1;n]
        
    retituisce
    --------------
        sum(sqrt(i))_{i=1}^{n}
    '''
    m = np.sqrt(np.arange(1, n + 1))
    return np.sum(m)

def sommaRadApprox(n):
    '''
    funzione che somma le radici dei primi numeri naturali.
    fino a 150 sfutta lo stesso metodo della funzione sommaRad(), oltre fa un'approssimazione con la formula dell'integrale.
    Oltre 150 infatti la formula integrale scende sotto lo 0.5% di scarto dal valore vero.
    
    parametri
    --------------
        n: numero naturale di cui fare la somma delle radici nell' intervallo [1;n]
        
    retituisce
    --------------
        +-sum(sqrt(i))_{i=1}^{n}
    '''
    if n < 150:
        m = np.sum(np.sqrt(np.arange(1,n+1)))
    else:
        m = (2/3) * (n**1.5)
    return m

def sumNDot(n):
    '''
    Funzione che riporta in lista prima la somma dei primi n numeri naturali utilizzando la formula di Gauss,
    poi il prodotto dei primi n numeri naturali, ovvero il fattoriale di n; per il fattoriale si sfrutta numpy per
    la sua efficienza.
    
    parametri
    --------------
        n: estremo destro dell'intervallo di cui fare somma/prodotto [1;n]
    
    restituisce
    --------------
        ( sum(i)_{i=0}^{n} , n! )
    '''
    p = math.factorial(n)
    s = (n + 1) * (n / 2)
    return s, p

def sumPow(n, *k):
    '''
    Funzione che calcola la somma delle potenze dei primi n numeri naturali
    
    parametri
    --------------
        n: estremo destro dell'intervallo di cui fare la somma di potenze [1;n]
        a: eventuale esponente della somma (se non immesso la funzione fa una semplice somma)
    altri eventuali parametri non vengono gestiti dalla funzione
    
    restituisce
    --------------
        sum(i^a)_{i=1}^{n}
    '''
    if len(k) > 0 and k[0] != 1:
        m = 0
        for i in range(1, n + 1):
            m += i**k[0]
        return m
    else:
        return (n + 1) * (n / 2)