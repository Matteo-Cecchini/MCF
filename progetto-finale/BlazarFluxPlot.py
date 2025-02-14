import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fft import ifft
import LCRanalysis as LCR
import os

def load_data(filepath):
    '''
    Inizializza un oggetto Datasheet in base al path immesso.
    '''
    data = LCR.read_csv("lcr", filepath)
    return data

def plot(data: LCR.Datasheet, choice, percentile, shuffles, timeformat):
    '''
    Genera il grafico scelto dall'utente in funzione della parola chiave.
    ---------------
    Parametri:
        data: oggetto Datasheet di cui mostrare il grafico.
        choice: parola chiave scelta dall'utente, le possibili scelte sono ["all", "data", "analysis", "spectrum"].
        percentile: percentile di significatività scelto per la soglia dell'analisi.
        shuffles: numero di permutazioni casuali da fare nell'analisi
        timeformat: sceglie come visualizzare i dati temporali, come Julian Date o MET

    La funzione sfrutta i metodi interni di Datasheet per fare automaticamente il grafico scelto dall'utente.
    Se viene immesso "all" come choice, vengono generati tutti i grafici possibili ("data", "spectrum", "analysis") in successione.
    '''
    data.convert_to_numeric(inplace=True)
    data.FFT(inplace=True)
    data.shuffle_analysis(shuffles=shuffles, percentile=percentile, inplace=True)
    if choice == "analysis":
        data.plot_analysis(percentile=percentile)
    elif choice == "all":
        data.plot_data(timeformat)
        data.plot_spectrum(see_parts=True)
        data.plot_analysis(percentile=percentile, timeformat=timeformat)
    elif choice == "data":
        data.plot_data(timeformat)
    elif choice == "spectrum":
        data.plot_spectrum(see_parts=True)

def dualplot(d1: LCR.Datasheet, d2: LCR.Datasheet, shuffles, percentile, names, timeformat: str = ""):
    '''
    Genera il grafico degli spettri di potenza di daati settimanali e mensili con l'evidenza dei coefficienti significativi,
    e le curve di luce filtrate sui coefficienti significativi a confronto con le curve originali dei segnali (solo la curva mensile, altrimenti
    il grafico diventa affollato)
    ---------------
    Parametri:
        d1, d2: i due oggetti Datasheet di cui fare il plot. I file csv dei Blazar sono opportunamente messi a coppia settimana-mese, quindi il confronto viene di conseguenza tra dati presi sttimanalmente e mensilmente.
        shuffles: numero di permutazioni casuali eseguite nell'analisi
        percentile: percentile di significatività scelto per la soglia dell'analisi
        names: nomi dei file, vengono scelti con os quando viene immesso il path della cartella
        timeformat: sceglie come visualizzare i dati temporali, come Julian Date o MET
    '''
    # Controllo che ci sia tutto in d1
    d1.convert_to_numeric(True, True)
    d1.FFT(True)
    d1.shuffle_analysis(shuffles=shuffles, percentile=percentile, inplace=True)
    
    d2.convert_to_numeric(True, True)
    d2.FFT(True)
    d2.shuffle_analysis(shuffles=shuffles, percentile=percentile, inplace=True)
    
    # DIstinzione delle lunghezze degli array di frequenza e coefficienti FFT
    cut1 = len(d1.timedata) // 2
    cut2 = len(d2.timedata) // 2
    sig_freq_d1, sig_pows_d1 = d1.frequencies, np.absolute(d1.coefficients)**2
    sig_freq_d2, sig_pows_d2 = d2.frequencies, np.absolute(d2.coefficients)**2
    
    # Prendo i dati sopra la soglia di significatività scelta
    indexes_d1 = [i for i in d1.significativity["indexes"] if i < cut1]
    sig_freq_d1 = d1.frequencies[indexes_d1]
    sig_pows_d1 = np.absolute(d1.coefficients[indexes_d1])**2
    
    indexes_d2 = [i for i in d2.significativity["indexes"] if i < cut2]
    sig_freq_d2 = d2.frequencies[indexes_d2]
    sig_pows_d2 = np.absolute(d2.coefficients[indexes_d2])**2
    
    # fontsize legende
    fontsize = 7.5
    
    # Definizione figura
    fig = plt.figure(figsize=(13,6))
    gs = fig.add_gridspec(1, 2)
    
    # Plot a sinistra: spettro di potenza a confronto dati settimanali e mensili
    ps, sn = gs.subplots()
    
    # settimanali
    ps.plot(sig_freq_d1, sig_pows_d1, ".", lw=1, label=f"Picchi sett. oltre {percentile} % ")
    ps.plot(d1.frequencies[:cut1], np.absolute(d1.coefficients[:cut1])**2, lw=1, color="gray", alpha=.5, label=f"Spettro di potenza {names[0]}")
    ps.hlines(d1.significativity["treshold"], 0, np.max(d1.frequencies[:cut1]), color="darkturquoise", alpha=.4, label=f"Soglia del {percentile}% sett.", linestyles="dashed")
    
    # mensili
    ps.plot(sig_freq_d2, sig_pows_d2, ".", lw=1, label=f"Picchi mens.oltre {percentile} %")
    ps.plot(d2.frequencies[:cut2], np.absolute(d2.coefficients[:cut2])**2, lw=1, color="wheat", alpha=1, label=f"Spettro di potenza {names[1]}")
    ps.hlines(d2.significativity["treshold"], 0, np.max(d2.frequencies[:cut2]), color="plum", alpha=.4, label=f"Soglia del {percentile}% mens.", linestyles="dashed")
    
    # scritte delle fequenze sui dati significativi
    
    for i in range(len(indexes_d1)):
        ps.annotate("w {:.2e}".format(sig_freq_d1[i]), xy=(sig_freq_d1[i], sig_pows_d1[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    for i in range(len(indexes_d2)):
        ps.annotate("m {:.2e}".format(sig_freq_d2[i]), xy=(sig_freq_d2[i], sig_pows_d2[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    
    # zoom o su tutto l'array delle potenze FFT dei mensili o fino l'ultima potenza significativa dei settimanali
    try:
        lims = max(d2.frequencies[cut2 - 1], max(sig_freq_d1)) 
    except:
        lims = d2.frequencies[cut2 - 1]
    ps.set_xlim(0, lims)
    ps.set_title("Spettro di potenza delle due strutture")
    ps.set_xlabel("Frequenza [d$^{-1}$]")
    ps.set_ylabel("Potenza (log$_{10}$)")
    ps.set_yscale("log")
    ps.legend(fontsize=fontsize)
    
    # Plot a detra: sintesi a confronto con i dati mensili (sono di meno e si vedono di più)
    # sintesi settimanali
    cc_d1 = np.zeros(len(d1.coefficients), dtype=np.complex128)
    cc_d1[indexes_d1] = d1.coefficients[indexes_d1]
    yy_d1 = np.absolute(ifft(cc_d1))
    
    # sintesi mensili
    cc_d2 = np.zeros(len(d2.coefficients), dtype=np.complex128)
    cc_d2[indexes_d2] = d2.coefficients[indexes_d2]
    yy_d2 = np.absolute(ifft(cc_d2))

    # Formattazione valori temporali se si vuole Julian Date o MET
    if d1.csvformat["name"] == "lcr" and d2.csvformat["name"] == "lcr":
        if timeformat.lower() in ("met", "mission elapsed time"):
            timelabel = "Mission Elapsed Time (MET)"
            timedata_d1 = 86400*(d1.timedata - d1.timedata[0]) + d1.csvformat["MET0"]
            timedata_d2 = 86400*(d2.timedata - d2.timedata[0]) + d2.csvformat["MET0"]
        else:
            timelabel = d1.nameofdata[0]
            timedata_d1 = d1.timedata
            timedata_d2 = d2.timedata

    # vede se in d2 c'è limitmask, altrimenti non fa distinzione con upper limit. potrei toglierla sapendo che non c'è possibilita di scelta in questo script ma non si sa mai
    if d2.limitmask is not None:
        notmask = ~d2.limitmask
        sn.scatter(timedata_d2[d2.limitmask], d2.fluxdata[d2.limitmask], marker="v", label="Limiti superiori (mens.)")
    else:
        notmask = np.full(d2.timedata.shape, True)
    
    sn.errorbar(timedata_d2[notmask], d2.fluxdata[notmask], d2.sigmadata[notmask], 
                    fmt=".", elinewidth=1, ecolor="gray", label="Rilevamenti (mens.)")
    sn.plot(timedata_d2[notmask], d2.fluxdata[notmask], lw=1, color="gray", alpha=.5)

    # plot
    sn.plot(timedata_d1, yy_d1, lw=1, color="darkturquoise", label=f"Sintesi coefficienti oltre {percentile} % (sett.)")
    sn.plot(timedata_d2, yy_d2, lw=1, color="plum", label=f"Sintesi coefficienti oltre {percentile} % (mens.)")

    sn.set_xlabel(timelabel)
    sn.set_ylabel(d2.nameofdata[1])
    sn.set_title(f"Plot sintesi dati con coeff. significativi oltre {percentile}%,\na confronto con la curva di luce dei dati mensili")
    sn.legend(fontsize=fontsize)
    
    # show
    plt.tight_layout()
    plt.show()

def do_things(parser):
    '''
    In base alla scelta dell'utente genera i risultati di un singolo file csv o di una cartella con dati settimanali e mensili.
    '''
    args = parser.parse_args()
    
    if args.dir:
        names = os.listdir(args.filepath)
        paths = [os.path.join(args.filepath,i) for i in names] # qui assumo che ci siano solo due file nella cartella, estendibile a n file anche nella funzione
        paths = paths if "weekly" in names[0] else paths[::-1] # inversione se legge monthly prima di weekly
        names = names if "weekly" in names[0] else names[::-1] # inversione se legge monthly prima di weekly
        data1, data2 = [load_data(i) for i in paths]
        dualplot(data1, data2, args.iterations, args.percentile, names, args.timeformat)
    else:
        data = load_data(args.filepath)
        choice = args.show
                
        if args.timeformat is not None and args.timeformat.lower() in ("met", "mission elapsed time"):
            args.timeformat == "met"
            
        plot(data, choice, args.percentile, args.iterations, args.timeformat)
        
def main():
    parser = argparse.ArgumentParser(description="Analisi di periodicità delle curve di luce dei Blazar")
    parser.add_argument("filepath", type=str, help="Percorso del file CSV della curva di luce")
    parser.add_argument("-d", "--dir", type=bool, default=False, help="Se è una cartella (True) mette a confronto tutti i file nella cartella")
    parser.add_argument("-s", "--show", choices=["all", "data", "spectrum", "analysis"], default="analysis", help="Scelta del tipo di plot")
    parser.add_argument("-p", "--percentile", type=float, default=95, help="Soglia di significatività percentile massimi")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Numero di shuffle da eseguire")
    parser.add_argument("-t", "--timeformat", choices=["JD", "Julian Date", "MET", "Mission Elapsed Time"], 
                        default="JD", help="Scelta del formato data, Julian Date o MET")
    
    do_things(parser)

if __name__ == "__main__":
    main()