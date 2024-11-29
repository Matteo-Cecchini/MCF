import numpy as np

'''
Modulo con alcune funzioni di risoluzione di problemi di Cauchy.    
Le funzioni fornite sono:                                           
    - Euler (metodo di Eulero)                                      
    - RK2   (metodo di Runge-Kutta al second'ordine)                
    - RK4   (metodo di Runge-Kutta al quart'ordine)                 
                                                                    
    Nota: ad ora la funzione da immettere nei metodi risolutivi deve    
          avere come parametri prima le variabili dipendenti, poi quelle
          indipendenti                                                        
'''

def Euler(fun: callable, y0: float, x_start: float, x_end: float, *funargs, iterations: int = 100):
    '''
    Funzione per applicare il metodo di Eulero ad una funzione per un problema di Cauchy del tipo:
        dy/dx = fun(x, y) , con y(x0) = y0
    
    Parametri
    ----------
        fun: funzione argomento per il metodo di Eulero (i parametri vanno nell'ordine y,x)
        y0: parametro iniziale per la variabile dipendente
        x_start: limite sinistro dell'intervallo su cui applicare il metodo di Eulero
        x_end: limite sinistro dell'intervallo su cui applicare il metodo di Eulero
        *funargs: eventuali parametri della funzione
        iterations: numero di iterazioni da fare nell'intervallo selezionato (100 di default), chiamabile solo tramite keyword
        
    Return
    ----------
        np.ndarray con le variabili dipendenti
    '''
    h = ( x_end - x_start ) / iterations
    yy = np.zeros(iterations)
    xx = np.arange(x_start, x_end, h)
    yy[0] = y0
    for i in range(iterations - 1):
        yy[i + 1] = yy[i] + h*fun(yy[i], xx[i], *funargs)
    return yy, xx

def RK2(fun: callable, y0: float, x_start: float, x_end: float, *funargs, iterations: int = 100):
    '''
    FUnzione per applicare il metodo Runge-Kutta di secondo ordine ad un problema di Cauchy del primo ordine.
    
    Parametri
    ----------
        fun: funzione derivata prima del problema di Cauchy di cui si vuole avere l'approssimazione (i parametri vanno nell'ordine y,x)
        y0: valore iniziale della variabile dipendente
        x_start: limite sinistro dell'intervallo di sui si vuole trovare la soluzione al problema di Cauchy
        x_end: limite destro dell'intervallo di sui si vuole trovare la soluzione al problema di Cauchy
        *funargs: eventuali parametri della funzione
        iterations: numero di iterazioni da fare nell'intervallo di approssimazione, chiamabile solo tramite keyword
        
    Return
    ----------
        np.ndarray con i valori della variabile dipendente usando il metodo RK2
    '''
    h = ( x_end - x_start ) / iterations
    xx = np.arange(x_start, x_end, h)
    yy = np.zeros(iterations)
    yy[0] = y0
    for i in range(iterations - 1):
        k1 = h*fun(yy[i], xx[i], *funargs)
        k2 = h*fun(yy[i] + k1/2, xx[i] + h/2, *funargs)
        yy[i + 1] = yy[i] + k2
    return yy

def RK4(fun: callable, y0: float, x_start: float, x_end: float, *funargs, iterations: int = 100):
    '''
    FUnzione per applicare il metodo Runge-Kutta di quarto ordine ad un problema di Cauchy del primo ordine.
    
    Parametri
    ----------
        fun: funzione derivata prima del problema di Cauchy di cui si vuole avere l'approssimazione (i parametri vanno nell'ordine y,x)
        y0: valore iniziale della variabile dipendente
        x_start: limite sinistro dell'intervallo di sui si vuole trovare la soluzione al problema di Cauchy
        x_end: limite destro dell'intervallo di sui si vuole trovare la soluzione al problema di Cauchy
        *funargs: eventuali parametri della funzione
        iterations: numero di iterazioni da fare nell'intervallo di approssimazione, chiamabile solo tramite keyword
        
    Return
    ----------
        np.ndarray con i valori della variabile dipendente usando il metodo RK4
    '''
    h = ( x_end - x_start ) / iterations
    xx = np.arange(x_start, x_end, h)
    yy = np.zeros(iterations)
    yy[0] = y0
    for i in range(iterations - 1):
        k1 = h*fun(yy[i], xx[i], *funargs)
        k2 = h*fun(yy[i] + k1/2, xx[i] + h/2, *funargs)
        k3 = h*fun(yy[i] + k2/2, xx[i] + h/2, *funargs)
        k4 = h*fun(yy[i] + k3, xx[i] + h, *funargs)
        yy[i + 1] = yy[i] + (k1 + 2*(k2 + k3) + k4) / 6
    return yy