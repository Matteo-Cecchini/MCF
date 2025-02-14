Progetto: Analisi delle Curve di Luce dei Blazar

Descrizione

    Questo progetto fornisce strumenti per analizzare le curve di luce dei Blazar, tramite trasmormata di Fourier (FFT) e un test statistico basato sulla generazione di curve sintetiche di luce.
    Il codice permette di stimare periodicità e segnali significativi confrontando lo spettro di potenza del segnale originale con una soglia di significatività (scelta dall'utente) dei massimi spettrali di successive permutazioni casuali del segnale da analizzare.

Struttura del Progetto

    BlazarFluxPlot.py:
        Codice adibito all'utilizzo rapido per l'analisi e la visualizzazione dei risultati del o dei segnali che si vogliono analizzare all'interno di file csv.

        Funzione load_data(filepath): Carica un file CSV e lo converte in un oggetto Datasheet.
        Funzione plot(data, choice, percentile, shuffles, timeformat): Genera diversi tipi di grafici basati sui dati forniti.
        Funzione dualplot(d1, d2, shuffles, percentile, names, timeformat): Confronta due dataset (es. dati settimanali vs mensili) e analizza le loro periodicità.
        Funzione do_things(parser): Analizza gli argomenti forniti da riga di comando ed esegue le funzioni appropriate.
        Funzione main(): Definisce il parser degli argomenti e avvia l'esecuzione del codice.

    LCRanalysis.py
        Modulo di supporto che gestisce l'elaborazione dei dati e l'analisi delle curve di luce. Struttura principale:

        Funzione to_numeric(arr): Converte array di stringhe in float64 utilizzando una libreria C++ esterna.
        Funzione shuffle_analysis(signal, shuffles, percentile): Applica un testi statistico per identificare periodicità significative.
        
        Classe Datasheet:
            Contiene i dati temporali, flussi, frequenze e coefficienti FFT.
            Metodi per elaborare i dati: FFT(), shuffle_analysis(), plot_data(), plot_spectrum(), plot_analysis().
        
        Funzione read_csv(from_data, path): Carica un CSV e lo converte in un oggetto Datasheet.

Requisiti

    I codici scritti utilizzano le seguenti librerie python:
        - NumPy
        - Pandas
        - matplotlib
        - SciPy
        - argparse
        - os
        - ctypes

Utilizzo (BlazarFluxPlot.py)

    BlazarFluxPlot.py è strutturato partendo dalla necessità di dover analizzare file singoli o coppie di file contenenti dati di flusso di Blazar; di conseguenza il codice è orientato all'analisi di singoli file o della cartella contente le coppie di file (su presa dati settimanale-mensile dello stesso oggetto celeste).
    Per l'analisi "a coppia" l'approccio è quello di utilizzare come riferimento da terminale la cartella con entrambi i file.
    
    Se l'analisi riguarda un singolo file:
        
        $ python3 BlazarFluxPlot.py <percorso_al_file_csv>

    Se l'analisi riguarda i dati di una cartella:

        $ python3 BlazarFluxPlot.py <percorso_cartella> -d True (oppure 1)

    Le altre opzioni del parser sono in comune.
    Opzioni disponibili:

        -s, --show : tipo di visualizzazione (all, data, spectrum, analysis).

        -p, --percentile : soglia di significatività del test (default: 95).

        -i, --iterations : numero di shuffle per l'analisi (default: 100).

        -t, --timeformat : formato del tempo (JD o MET).


Output

    In base all'argomento immesso in --show, il codice genera automaticamente i seguenti grafici:

        "data" : curva di luce, Mostra l'andamento del flusso nel tempo.

        "spectrum" : mostra lo spettro della curva di luce analizzata, son accanto parte reale e immaginaria dei coefficienti.

        "analysis" : mostra lo spettro mettendo in evidenza la soglia trovata nel test e tutte le potenze oltre quella soglia. 
                     Accanto mostra la curva di luce del file e la sintesi della curva filtrata sui coefficienti che hanno superato il test.

                     Se l'analisi riguarda una coppia di segnali (le coppie sono sempre di curve con misurazioni settimanali e mensili), il grafico mette a confronto
                     gli spettri con i rispettivi coefficienti che hanno apssato il test, con accanto la curva di luce mensile (altrimenti diventa troppo affollato)
                     e le sintesi filtrate dei due segnali.
        
        "all" : mostra tutte le opzioni, successivamente.

Approfondimenti:

    Datasheet:
        La classe Datasheet in LCRanalysis.py è pensata per le analisi delle curve di luce del Light Curve Repository, ma strutturata perché possa essere flessibile.

        Nella lettura dei csv è possibile immettere parole chiave che rimandano all'origine dei csv stessi, cosicché il Datasheet venga inizializzato con funzionalità pertinenti. Tutto questo a patto che l'origine del csv sia implementata nelle funzioni di lettura del csv e nelle funzioni di plot interne alla classe (ad ora c'è solo la parola chiave "lcr").

    Test statistico:
        Il testi che effettua la classe Datasheet e la funzione shuffle_analysis in LCRanalysis.py funziona nel modo seguente:

            - preso un segnale in funzione del tempo, si fa uno shuffle per n volte

            - per ogni shuffle, si prende il massimo del suo spettro di potenza

            - ottenuti n massimi, si prende un percentile e si mette a confronto con lo spettro di potenza del segnale originale

        I coefficienti relativi alle potenze che superano il percentile sono quelle statisticamente rilevanti.

        Nel caso di un segnale afflitto da forti fenomeni stocastici, come la curva di luce di un Blazar,
        non è molto utile cercare semplicemente i picchi dello spettro di potenza, poiché la presenza di rumore
        può rendere difficile l’individuazione di periodicità reali. 
        
        Mescolando il segnale si distruggono tutte le sue potenziali periodicità, trasformandolo in 
        qualcosa che può essere assimilato a un rumore puro. Questo rumore nel suo spettro di potenza presenta 
        comunque un massimo, che corrisponde alla frequenza più ricorrente nel dominio del tempo di una       distribuzione casuale.

        Permutando più volte il segnale originale in maniera casuale e calcolando i massimi nei rispettivi 
        spettri di potenza, si ottiene una distribuzione statistica di riferimento per il rumore del segnale. 
        
        Infine, stabilendo una soglia di significatività basata su questa distribuzione, si possono identificare 
        i coefficienti spettrali che superano tale soglia, indicando che hanno una probabilità inferiore a un dato 
        livello (ad esempio il 5%) di essere semplicemente dovuti al rumore.