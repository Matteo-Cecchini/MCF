import numpy as np

def gaussNpoly(x, *p):
    '''
    funzione di Gauss + una polinomiale di primo grado
    
    parametri
    -----------
        x: variabili indipendenti
        *p: lista di parametri, solo i primi cinque verranno utilizzati.
            I primi tre sono rispettivamente media, normalizzazione e deviazione della gaussiana;
            gli ultimi due sono pendenza e quota della retta
    return
    -----------
        la somma di una funzione di Gauss più una retta
    '''
    gauss = p[1]*np.exp(-((x - p[0])**2)/(2*p[2]**2))
    poly = p[3]*x + p[4]
    return gauss + poly

def doubleGaussNpoly(x, *p):
    '''
    funzione di Gauss + una polinomiale di primo grado
    
    parametri
    -----------
        x: variabili indipendenti
        *p: lista di parametri, solo i primi sette verranno utilizzati. Il primo è la media delle due gaussiane,
            secondo e terzo normalizzazione e deviazione della prima gaussiana, quarto e quindo le stesse per la seconda,
            gli ultimi due pendenza e quota della retta.
    return
    -----------
        la somma di due funzioni di Gauss con media in comune come parametro più una retta
    '''
    gauss1 = p[1]*np.exp(-((x - p[0])**2)/(2*p[2]**2))
    gauss2 = p[3]*np.exp(-((x - p[0])**2)/(2*p[4]**2))
    poly = p[5]*x + p[6]
    return (gauss1 + gauss2 + poly)

def freeDoubleGaussNpoly(x, *p):
    '''
    funzione di Gauss + una polinomiale di primo grado
    
    parametri
    -----------
        x: variabili indipendenti
        *p: lista di parametri, solo i primi otto verranno utilizzati.
            I primi tre sono media, normalizzazione e deviazione della prima gaussiana;
            i secondi tre sono gli analoghi per la seconda gaussiana;
            gli ultimi due sono pendanza e quota della retta
    return
    -----------
        la somma di due funzioni di Gauss con media in comune come parametro più una retta
    '''
    gauss1 = p[1]*np.exp(-((x - p[0])**2)/(2*p[2]**2))
    gauss2 = p[4]*np.exp(-((x - p[3])**2)/(2*p[5]**2))
    poly = p[6]*x + p[7]
    return (gauss1 + gauss2 + poly)