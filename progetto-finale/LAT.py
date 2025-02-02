import pandas as pd
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import multiprocessing

_lib = np.ctypeslib.load_library("float_conv.so", ".")

_lib.float_conv.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_double)]
_lib.float_conv.restype = None

def to_numeric(arr: np.ndarray|list):
    '''
    Sfrutta l'oggetto condiviso float_conv.so per convertire in numero array di stringhe.
    ----------
    Parametri:
        arr: struttura array-like da immettere, anche se la funzione converte anche valori singoli.
            Il funzionamento della funzione C++ a cui fa capo float_conv.so gestisce tre input: la lunghezza dell'array da convertire, un array della stessa lunghezza dell
    '''
    arrlen = len(arr)
    input = (ct.c_char_p * arrlen)(*[i.encode('utf-8') for i in arr])
    output = (ct.c_double * arrlen)()
    _lib.float_conv(arrlen, input, output)
    return np.frombuffer(output, dtype=float)


class LATdatasheet(pd.DataFrame):
    
    def __init__(self, *args, **kwargs):
        '''
        Costruttore della classe, eredita direttamente da pd.DataFrame
        '''
        super().__init__(*args, **kwargs)
    
    @property
    def _constructor(self):
        '''
        Funzione di proprietà di LATdatasheet, permette di rimandare alla stessa classe e non alla classe padre pd.DataFrame. 
        '''
        return LATdatasheet
    
    def __getitem__(self, cols: str|list[str]):
        '''
        Funzione di accesso alle colonne di LATdatasheet, eredita direttamente da pd.DataFrame.
        -----------
        Parametri:
            cols: stringa o lista di stringhe, nomi delle colonne di cui si vuole avere accesso
        
        Return:
            LATdatasheet con colonne le colonne immesse come parametro e i relativi valori.
        '''
        return super().__getitem__(cols)
    
    @property
    def loc(self):
        '''
        Metodo di localizzazione dei valori della classe tramite condizione/slice.
        '''
        return super().loc
    
    def to_numeric(self, cols: str = None, inplace: bool = False):
        '''
        Funzione di conversione a valori numerici dei dati della classe. Questa funzione risponde allo scopo dell'analisi,
        che prende direttamente i valori limite preceduti da '<' all'interno dello studio in frequenza.
        La conversione elimina i simboli non numerici come '<' e riporta il dato numerico.
        ----------
        Parametri:
            cols: le colonne di cui fare la conversione a dati numerici. Se il parametro non è immesso viene fatto su tutte le colonne
            inplace: parametro booleano, permette la conversione sul posto se True, altrimenti ritorna un altro LATdatasheet
        
        Return:
            LATdatasheet con i valori convertiti in float64 se inplace è False.
        '''
        if cols == None:
            cols = self.columns
        
        df = self if inplace else self.copy()
        
        for col in cols:
            try:
                df.loc[:,col] = to_numeric(self[col].values)
            except Exception as e:
                if df[col].values.dtype == np.float64:
                    continue
                print(f"Conversione non eseguita per la colonna '{col}': {e}")
        if not inplace:
            return df
        
    def fft_analysis(self, col: str|int = None, inplace = False):
        '''
        Funzione per la trasformata di Fourier dei dati.
        ------------
        Parametri:
            col: la colonna dei dati di cui fare la trasformata
            inplace: booleano per sffettuare la funzione in-place
        
        Return:
            Un LATdatasheet con colonne le colonne del LATdatasheet di cui si sta chiamando il metodo 
            più le colonne "Frequencies" e "Coefficients", con rispettivamente frequenze e coefficienti
            della trasformata effettuata.
        '''
        if col == None:
            col = "Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"
        elif isinstance(col, int):
            col = self.columns[col]
        
        data_to_transform = self[col].values
        data_freq = data_to_transform[1]-data_to_transform[0]
        
        coef = fft(data_to_transform)
        freq = fftfreq(len(data_to_transform), d=data_freq)
        
        df = self if inplace else self.copy()
        df["Frequencies"] = freq
        df["Coefficients"] = coef
        return df
    
    def shuffle_analysis(self, n: int = 100, col: str = None):
        if ("Frequencies", "Coefficients") not in self.columns:
            self.fft_analysis(col, inplace=True)
            
        x = self[col].values.copy()
        y = np.zeros(len(self))
        for i in range(n):
            np.random.shuffle(x)
            y += np.absolute(fft(x))**2
        y /= n
        return np.mean(y), np.std(y)
    
    def wf_plot(self, cols: str|list[str] = None, *args, **kwargs):
        '''
        Funzione interna per fare il plot dei dati
        ----------
        Parametri:
            cols: colonne di cui fare il plot, di default sono la data giuliana e il flusso di energia
            *args: altri parametri da passare a plot
            **kwargs: altri parametri da passare a plot
        '''
        if cols == None:
            xcol, ycol = "Julian Date", "Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"
        
        x, y = self[xcol], self[ycol]
        plt.plot(x, y, *args, **kwargs)
        plt.xlabel(xcol, size=12)
        plt.ylabel(ycol, size=12)
        plt.tight_layout()
        plt.show()
        
    def ps_plot(self, *args, **kwargs):
        '''
        Funzione interna per fare il plot dello spettro di potenza della colonna di cui si è fatta la trasformata di Fourier.
        Qualora non sia mai stato chiamato il metodo fft_analysis, il metodo verrà chiamato nella sua forma di default.
        ----------
        Parametri:
            *args: parametri da passare a plot
            **kwargs: parametri con parola chiave da passare a plot
        '''
        if ("Frequencies", "Coefficients") not in self.columns:
            self.fft_analysis(inplace=True)
        
        x, y = self["Frequencies"].values, self["Coefficients"].values
        cut = len(x) // 2 + 1
        plt.plot(x[cut:], np.absolute(y[cut:])**2, *args, **kwargs)
        plt.xlabel("Frequencies (Hz)", size=12)
        plt.ylabel("Power of Fourier's transform's coefficients", size=12)
        plt.tight_layout()
        plt.show()
        

def read_csv(path: str):
    '''
    Usa la funzione built-in di pandas per leggere un file .csv
    '''
    return LATdatasheet(pd.read_csv(path))