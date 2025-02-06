import pandas as pd
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import norm
import multiprocessing
from time import time

_lib = np.ctypeslib.load_library("prova.so", ".")

_lib.float_conv.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
_lib.float_conv.restype =  ct.POINTER(ct.c_double)

def to_numeric(arr):
    '''
    Converte strutture array-like in array di float64
    ---------------
    Parametri:
        arr: l'array-like da convertire
        
    Return:
        Un array di float64
        
    Struttura e funzionamento:
        La funzione sfrutta la funzione C++ ottimizzata nell'oggetto condiviso float_conv.so.
        La funzione in C++ riceve puntatori a puntatori char, la struttura analoga per le liste di stringhe,
        ed effettua un ciclo per ogni elemento ignorando ad ogni iterazione l'elemento più a sinistra
        finché non riesce a convertire efficacemente; qualora si passi una stringa senza un numero che è possibile
        da convertire la funzione dara zero.
        
        Il blocco try-except gestisce la conversione dell'input in un array, e stampa una stringa di notifica
        nel caso la conversione fallisse.
    '''
    try:
        arr = np.asarray(arr, dtype=np.object_)
        
        size = len(arr)
        inp = (ct.c_char_p * size)(*arr.astype(np.bytes_))
        out = ct.POINTER(ct.c_double)
        out = _lib.float_conv(size, inp)
        res = np.array([out[i] for i in range(size)], dtype=np.float64)
    except Exception as e:
        print(f"Conversione dei dati immessi fallita: {e}")
    return res

def shuffle_analysis(arr, n: int = 100, sigmas: float = 3.9, is_raw: bool = False):
    '''
    Effettua un'analisi di significatività statistica delle periodicità di un campione di dati
    attraverso lo shuffling dei dati.
    ---------------
    Parametri:
        - arr : i dati di cui fare l'analisi.
        - n: numero di shuffling da fare.
        - sigmas: numero di deviazioni dalla media dei valori degli shuffle ottenuti.
        - is_raw: gestisce la situazione nel caso in cui si passi alla funzione non l'array dei coefficienti
                della trasformata di Fourier ma i dati in funzione del tempo.
                
    Return:
        Un dizionario con la seguente struttura:
            - "data": un array 1-D con la stessa lunghezza dell'array immesso, con la somma di tutti
                      i coefficienti ottenuti per shuffle ad ogni indice mediati sul numero di shuffle effettuati.
            - "indexes": un array con gli indici di tutti i coefficienti di Fourier la cui potenza supera il numero
                         di deviazioni dalla media di "data".
            - "mean": la media di tutti i coefficienti in "data".
            - "std": la deviazione standard di tutti i coefficienti in "data".
            
    Struttura e funzionamento:
        L'approccio della funzione è quello di stabilire un livello di significatività per la periodicità
        di tutti i coefficienti immessi (o dei coefficienti relativi ai dati immessi) rimescolando i dati;
        in questo modo si rompono tutti i legami temporali, mantenendo il rumore perchè per definizione
        non associato ad un'informazione.
        
        Piuttosto che salvare in memoria n array di shuffle, ogni volta viene sommato il nuovo shuffle ad un array
        di zeri generato apposta, ed infine viene mediata ogni somma per il numero di iterazioni n.
        Ogni indice di quello che era l'array di zeri è quindi la media delle potenze dei coefficienti di n shuffle
        delle potenze originali.
        
        La funzione effettua lo shuffle sulle potenze dei coefficienti anziché sui dati originali per motivi di 
        efficienza, in quanto np.random.shuffle ha complessità O(n), mentre eseguire la FFT sui dati originali 
        comporterebbe una complessità O(n log n) per ogni iterazione. Inoltre, matematicamente, la permutazione 
        delle potenze dei coefficienti non altera l'informazione temporale: l'argomento dell'esponenziale complesso 
        nella trasformata di Fourier dipende dal tempo, ma nelle potenze dei coefficienti ogni fase è uguale a 1, 
        quindi ogni permutazione di $x(t)$ corrisponde a una permutazione analoga di $|X(t)|^2$.       
    '''
    if is_raw:
        arr = fft(arr)
    # apparentemente lo shuffle della potenza dei coefficienti ha lo stesso risultato di fare lo shuffle dei dati prima
    data = np.zeros(len(arr))
    powers = np.absolute(arr)**2
    shuffle_dummy = powers.copy()
    # migliorare il loop se possibile
    for i in range(n):
        np.random.shuffle(shuffle_dummy)
        data += shuffle_dummy
    data /= n
    mean, std = np.mean(data), np.std(data)
    indexes = np.where(powers > mean + sigmas*std)[0]
    return {"data":data, "indexes":indexes, "mean":mean, "std":std}


class Datasheet:
    # array presi dal csv di analisi
    timedata = np.array([])
    fluxdata = np.array([])
    sigmadata = np.array([])
    # array dell'analisi
    frequencies = np.array([])
    coefficients = np.array([])
    coeff_sigmas = np.array([])
    significativity = {}
    # array per il plot
    limitmask = np.array([])
    nameofdata = list
    csvformat = {}
    
    def __init__(self, td: np.ndarray|list, yd: np.ndarray|list = None, sigma = None, names: str = None, **csvformat: dict):
        '''
        Inizializza un oggetto contenente dati temporali, flussi e sigma associati.
        ---------------
        Parametri:
            - td: array-like con dati temporali; se un array bidimensionale, la prima riga rappresenta i tempi,
                  la seconda i flussi e la terza (se presente) le incertezze (sigma).
            - yd: array-like con i dati in funzione del tempo. Se fornito, deve essere di dimensione compatibile 
                  con "td".
            - sigma: array-like con le incertezze di "yd". Se fornito, deve avere la stessa forma di "yd". 
                     Se non fornito o di forma non compatibile, verrà utilizzato un array di `np.nan`.
            - names: Nomi personalizzati per i dati temporali, i flussi e le incertezze. 
                     Default è ["Time", "Data", "Sigma"].
            - csvformat: Parametri aggiuntivi per la gestione del formato CSV.

        L'inizializzazione tenta la conversione diretta dei dati in array NumPy di tipo float64 e verifica la compatibilità dimensionale. 
        Se i dati temporali sono forniti come un array bidimensionale, la prima riga è considerata i tempi, la seconda i flussi e la terza (se presente) le incertezze. In caso di formati non compatibili, viene sollevata un'eccezione.
        '''
        for i in (td, yd, sigma):
            try:
                i = np.asarray(i, dtype=np.float64) if i is not None else None
            except ValueError:
                i = np.asarray(i) if i is not None else None
            except Exception as e:
                raise Exception(f"Dati inseriti non convertibili in np.ndarray: {e}")

        if td.ndim == 2 and td.shape[0] > 1:
            self.timedata = td[0]
            self.fluxdata = td[1]
            self.sigmadata = td[2] if td.shape[0] > 2 else np.full_like(td[1], np.nan)
        elif yd is not None and td.shape == yd.shape:
            self.timedata = td
            self.fluxdata = yd
            self.sigmadata = sigma if sigma is not None and sigma.shape == yd.shape else np.full_like(yd, np.nan)
        else:
            print("Immissione di array di dimensioni non compatibili")
        self.nameofdata = names if names is not None else ["Time", "Data", "Sigma"]
        self.csvformat = csvformat
        
    def __getitem__(self, index):
        '''
        Restituisce l'elemento corrispondente all'indice specificato.
        Il metodo ritorna una matrice NumPy con tutti i dati disponibili della classe al momento della chiamata,
        guardando ai dati esistenti tramite la lista ausialiaria delle "colonne" nameofdata.
        '''
        allitems = [self.timedata, 
                          self.fluxdata, 
                          self.sigmadata, 
                          self.frequencies, 
                          self.coefficients, 
                          self.coeff_sigmas]
        items = np.vstack(allitems[:len(self.nameofdata)])
        try:
            return items[index]
        except IndexError:
            # non ho capito se l'erorre è gestito correttamente
            raise IndexError(f"IndexError: index {index} is out of bounds for axis 0 with size {len(self.nameofdata)}")                  
        
    def __str__(self):
        '''
        Restituisce una rappresentazione testuale formattata dell'oggetto.
        Il metodo prende spunto dal print di un pd.DataFrame, ma al momento il risultato non è quello sperato.
        '''
        names = ["index"] + self.nameofdata
        ln = [len(i) for i in names] # lunghezza minima di 8 caratteri
        header = "\t".join( f"{i:>{j}}" for i,j in zip(names, ln)) + "\n"
        dividedots = "\n" + "\t".join( f"{i:>{j}}" for i,j in zip(["..."]*len(names),ln) ) + "\n"
        
        def structure(n, a):
            part = lambda x: "\t".join( f"{i:>{j}}" for i,j in zip(x,ln[1:]) )
            line = "\n".join(f"{i:>{ln[0]}}\t" + part(j) for i,j in zip(n,a))
            return line


        headrows = structure(range(5), self[:, :5].T)
        endrows = structure(range(len(self.timedata) - 5, len(self.timedata)), self[:, -5:].T)

        # le colonne di cui fare il print sono sempre tre se va tutto bene
        return header + headrows + dividedots + endrows
    
    def copy(self):
        '''
        Metodo di copia profonda dell'oggetto.
        '''
        # NOTA: da verificare che .copy() serva
        df = self[:].copy()
        return Datasheet(df,
                         names=self.nameofdata.copy(),
                         **self.csvcsvformat.copy())
        
    @property
    def dtypes(self):
        '''
        Metodo di proprietà della classe che ha come return un dizionario con i tipi di elementi per ogni "colonna"
        presenta nella classe al momento della chiamata.
        '''
        # si chiama il tipo del primo elemento di ogni array perché usare .dtype da indistintamente <class 'object'>
        # finché non si danno array con elementi di tipo disuniforme non da problemi (si potrebbe implementare nell'inizializzazione)
        return { i:j for i,j in zip(self.nameofdata,
                                    (type(self.timedata[0]),
                                     type(self.fluxdata[0]),
                                     type(self.sigmadata[0]))) }
        
    @property
    def shape(self):
        '''
        Metodo di proprietà della classe per risalire ad una "forma" delle colonne interne.
        '''
        return self[:].shape

    def convert_to_numeric(self, inplace: bool =False, is_limit: bool = True):
        '''
        Converte i dati dell'oggetto in valori numerici, gestendo eventuali valori non numerici.
        ---------------
        Parametri:
            - inplace: se True la conversione avviene direttamente sull'oggetto corrente, 
                   se False viene restituita una copia dell'oggetto con i dati convertiti. 
                   Di default è False.
            - is_limit: Se True, identifica i limiti superiori nei dati di flusso come valori NaN, utilizzabili per il masking o per il plot. Default è True.

        Return:
            - None: se "inplace" è True, non viene restituito nulla.
            - Datasheet: una nuova copia dell'oggetto con i dati numerici, se "inplace" è False.

        Descrizione:
        Il metodo converte le colonne di dati (tempo, flusso e incertezze) in valori numerici, 
        utilizzando "to_numeric" della libreria corrente.
        '''
        # converte se stesso se richiesto in-place
        if not inplace:
            df = self.copy()
        else:
            df = self
        # indici dei limiti superiori per il plot bello
        df.limitmask = np.isnan(pd.to_numeric(df.fluxdata, errors="coerce")) if is_limit else None
        # converte le colonne
        df.timedata = to_numeric(df.timedata)
        df.fluxdata = to_numeric(df.fluxdata)
        df.sigmadata = to_numeric(df.sigmadata)
        # ha un return solo se richiesto in-place
        if not inplace:
            return df
        else:
            return None
    
    def FFT(self, inplace = False):
        '''
        Esegue la trasformata di Fourier discreta (FFT) sui dati.

        Parametri:
            - inplace: se True la trasformata di Fourier viene applicata direttamente sull'oggetto corrente,
                       se False viene restituita una copia dell'oggetto con i risultati della FFT. 
                       Di default è False.

        Restituisce:
            - Datasheet: Una copia dell'oggetto con i risultati della FFT se `inplace` è False, altrimenti non restituisce nulla e modifica direttamente l'oggetto corrente.

        Descrizione:
        Il metodo applica la trasformata di Fourier discreta (FFT) ai dati ("fluxdata") e alle incertezze ("sigmadata"). 
        Se i dati non sono già in formato numerico (float64), viene tentata una conversione tramite il metodo 
        "convert_to_numeric". Successivamente, il metodo calcola i coefficienti della FFT e le frequenze corrispondenti.
        Se "inplace" è False, viene restituita una nuova copia dell'oggetto con le colonne aggiornate: "Frequencies", 
        "FFT coefficients" e "Coeff. sigmas". Se "inplace" è True, l'oggetto viene modificato direttamente.
        '''
        df = self if inplace else self.copy()
        print(any(i != np.float64 for i in df.dtypes.values()))
        if any(i != np.float64 for i in df.dtypes.values()):
            try:
                df.convert_to_numeric(inplace=True)
            except Exception as e:
                print(f"Impossibile effettuare l'analisi: {e}")
        df.nameofdata += ["Frequencies", "FFT coefficients", "Coeff. sigmas"]
        df.coefficients = fft(self.fluxdata)
        df.coeff_sigmas = fft(self.sigmadata)
        # assunzione di array temporale scandito uniformemente
        df.frequencies = fftfreq(len(df.coefficients), d=(df.timedata[1] - df.timedata[0]))
        if not inplace:
            return df
            
    def shuffle_analysis(self, n: int = 100, sigmas=3.9, inplace = False):
        '''
        Esegue un'analisi statistica sui dati o sui relativi coefficienti FFT per valutare la significatività
        periodica di ogni elemento dello studio in frequenza.

        Parametri:
            - n: il numero di iterazioni da eseguire durante l'analisi di shuffle. Di default è 100.
            - sigmas: il numero di deviazioni standard per determinare la soglia di significatività. 
                      Di default è 3.9.
            - inplace: se True l'analisi viene eseguita direttamente sull'oggetto corrente,
                       se False viene restituita una copia dell'oggetto con i risultati dell'analisi. 
                       Di default è False.

        Restituisce:
            - Datasheet: Una copia dell'oggetto con i risultati dell'analisi se "inplace" è False, 
                         altrimenti non restituisce nulla e modifica direttamente l'oggetto corrente.

        Descrizione:
        Il metodo nasce con lo scopo di facilitare l'analisti di significatività statistica delle componenti periodiche di
        segnali astronomici attraverso la generazione di curve di luce sintetiche.
        Se "inplace" è True, l'oggetto viene modificato direttamente. Se "inplace" è False, viene restituita 
        una nuova copia dell'oggetto con i risultati aggiornati.
        '''
        df = self if inplace else self.copy()
        if any(i != np.float64 for i in df.dtypes.values()):
            try:
                df.convert_to_numeric(inplace=True)
            except Exception as e:
                print(f"Impossibile effettuare l'analisi: {e}")
        
        if "FFT coefficients" not in df.nameofdata:
            df.significativity = shuffle_analysis(df.fluxdata, n, sigmas=sigmas, is_raw=True)
        else:
            df.significativity = shuffle_analysis(df.coefficients, n, sigmas=sigmas, is_raw=False)
            
        if not inplace:
            return df
    
    def plot_data(self, timeformat: str = None):
        if any(i != np.float64 for i in self.dtypes.values()):
            try:
                self.convert_to_numeric(inplace=True)
            except Exception as e:
                print(f"Impossibile effettuare l'analisi: {e}")
        
        if self.csvformat["name"] in ("lcr"):
            if timeformat in ("met","MET"):
                timelabel = "Mission Elapsed Time (MET)"
                timedata = 86400*(self.timedata - self.timedata[0]) + self.csvformat["MET0"]
            else:
                timelabel = self.nameofdata[0]
                timedata = self.timedata
        
        if self.limitmask is not None:
            notmask = ~self.limitmask
            plt.scatter(timedata[self.limitmask], self.fluxdata[self.limitmask], marker="v", label="Upper Limit")
        else:
            notmask = np.full(self.timedata.shape, True)
            plt.title("nota: non c'è distinzione tra rilevamenti e limiti superiori")
        
        plt.errorbar(timedata[notmask], self.fluxdata[notmask], self.sigmadata[notmask], 
                     fmt=".", elinewidth=1, ecolor="gray", label="Detection")
        plt.plot(timedata[notmask], self.fluxdata[notmask], lw=1, color="gray", alpha=.5)
        plt.xlabel(timelabel)
        plt.ylabel(self.nameofdata[1])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, see_parts = False):
        if "Frequencies" not in self.nameofdata:
            self.FFT(inplace=True)
        
        cut = len(self.frequencies) // 2
        coeff_power = np.absolute(self.coefficients[:cut])**2
        coeff_power_err = np.absolute(2*self.coefficients[:cut]*self.coeff_sigmas[:cut]) # stima degli errori delle potenze dei coefficienti
        
        fig = plt.figure()
        if see_parts:
            g1 = fig.add_gridspec(1,1, left=.1, right=.45)
            g2 = fig.add_gridspec(2,1, left=.55, right=.9)
            
            re, im = g2.subplots(sharex=True, sharey=True)
            re.plot(self.frequencies[:cut], self.coefficients.real[:cut], color="gray", lw=1)
            im.plot(self.frequencies[:cut], self.coefficients.imag[:cut], color="gray", lw=1)
            re.errorbar(self.frequencies[:cut], self.coefficients.real[:cut], np.absolute(self.coeff_sigmas.real[:cut]), 
                     fmt=".", elinewidth=1, ecolor="gray", label="Parte reale")
            im.errorbar(self.frequencies[:cut], self.coefficients.imag[:cut], np.absolute(self.coeff_sigmas.imag[:cut]), 
                     fmt=".", elinewidth=1, ecolor="gray", label="Parte immaginaria")
            im.set_xlabel("Frequenza [Hz]")
            re.set_title("Parti reale e immaginaria dei coefficienti")
        else:
            g1 = fig.add_gridspec(1,1)
        sp = g1.subplots()
        sp.errorbar(self.frequencies[:cut], coeff_power, coeff_power_err, 
                     fmt=".", elinewidth=1, ecolor="gray")
        sp.plot(self.frequencies[:cut], coeff_power, lw=1, color="gray", alpha=.5)
        sp.set_title("Spettro di potenza dei coefficienti")
        sp.set_xlabel("Frequenza [Hz]")
        sp.set_ylabel("Potenza")
        plt.show()    

    def plot_analysis(self, sigmas: float|int = 3.9, timeformat: str = "", see_shuffle = False):
        if len(self.significativity) == 0:
            self.shuffle_analysis(sigmas=sigmas, inplace=True)
        if self.frequencies.size == 0:
            self.FFT(True)
        
        cut = len(self.timedata) // 2
        line = np.absolute(self.significativity["mean"] + sigmas*self.significativity["std"])
        indexes = [i for i in self.significativity["indexes"] if i < cut] # assumo che siano simmetrici i coefficienti
        sig_freq, sig_pows = self.frequencies[indexes], np.absolute(self.coefficients[indexes])**2
        
        perc = norm.cdf(sigmas)*100
        
        fig = plt.figure(figsize=(13,6))
        gs = fig.add_gridspec(1,2)
        ps, sn = gs.subplots()
        ps.plot(sig_freq, sig_pows, ".", lw=1, label=f"picchi con significatività oltre {round(perc,3)}%")
        ps.plot(self.frequencies[:cut], np.absolute(self.coefficients[:cut])**2, 
                lw=1, color="gray", alpha=.5, label="tutti i dati")
        ps.hlines(np.absolute(self.significativity["mean"]), 0, np.max(self.frequencies[:cut]), 
                  linestyles="dashed", alpha=.25, label="media degli shuffle")
        ps.hlines(line, 0, np.max(self.frequencies[:cut]), 
                  color="#1f77b4", alpha=.4, label=f"media + {round(sigmas,3)} deviazioni")
        
        ps.vlines(sig_freq, ymin=np.zeros(len(indexes)), ymax=sig_pows, color="gray", 
                  linestyles="dashed", alpha=.25)
        
        for i in range(len(indexes)):
            ps.annotate("{:.2e}".format(sig_freq[i]), 
                xy=(sig_freq[i], sig_pows[i]), 
                xytext=(5, 5), textcoords="offset points", ha="left", va="bottom",
                fontsize=8, color="black", fontweight='light')
        
        a = self.frequencies[1] - self.frequencies[0]
        ps.set_xlim(np.min(sig_freq) - a, np.max(sig_freq) + a)
        
        ps.set_title("Spettro di potenza dei coefficienti")
        ps.set_xlabel("Frequenza [Hz]")
        ps.set_ylabel("Potenza")
        ps.legend()
        
        cc = np.zeros(len(self.coefficients), dtype=np.complex128)
        cc[indexes] = self.coefficients[indexes]
        yy = np.absolute(ifft(cc))
        
        # perché ho fatto così?
        if self.csvformat["name"].lower() == "lcr" and timeformat.lower() in ("met", "mission elapsed time"):
            timelabel = "Mission Elapsed Time (MET)"
            timedata = 86400*(self.timedata - self.timedata[0]) + self.csvformat["MET0"]
        else:
            timelabel = self.nameofdata[0]
            timedata = self.timedata
        
        if self.limitmask is not None:
            notmask = ~self.limitmask
            sn.scatter(timedata[self.limitmask], self.fluxdata[self.limitmask], marker="v", label="Upper Limit")
        else:
            notmask = np.full(self.timedata.shape, True)
        
        sn.errorbar(timedata[notmask], self.fluxdata[notmask], self.sigmadata[notmask], 
                     fmt=".", elinewidth=1, ecolor="gray", label="Detection")
        sn.plot(timedata[notmask], self.fluxdata[notmask], lw=1, color="gray", alpha=.5)
        sn.plot(timedata, yy, lw=1, color="plum", label=f"sintesi picchi significativi oltre {perc.round(2)}%")
        if see_shuffle:
            sn.plot(timedata, self.significativity["data"], lw=1, color="#1f77b4", alpha=.5, label="media curve sintetiche")
            
        sn.set_xlabel(timelabel)
        sn.set_ylabel(self.nameofdata[1])
        sn.set_title(f"Plot dati con sintesi su picchi significativi oltre {round(perc,3)}%")
        sn.legend()
        
        plt.tight_layout()
        plt.show()

def read_csv(from_data: str|int, path: str):
    '''
    Funzione analoga al read_csv() di pandas, ma ritorna un Datasheet.
    '''
    df = pd.read_csv(path)
    if isinstance(from_data, int):
        try:
            columns = df.columns[from_data]
            data = Datasheet(df[columns].T.values, names=columns)
        except Exception as e:
            print(f"Colonne non trovate nel CSV: {e}")
    elif from_data.lower() == "lcr":
        columns = df.columns[[1,4,5]].to_list()
        csvformat = {"name" : "lcr", "MET0" : df["MET"][0]}
        data = Datasheet(df[columns].T.values, names=columns, **csvformat)
    else:
        print("Parola chiave non riconosciuta.")    
    return data