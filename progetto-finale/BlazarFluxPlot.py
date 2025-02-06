import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm
import LCRanalysis as LCR

def load_data(filepath):
    data = LCR.read_csv("lcr", filepath)
    return data

def analyze_periodicity(data: LCR.Datasheet, sigmas = 3.9, num_shuffles=100):
    """Effettua l'analisi di Fourier e verifica la periodicità."""
    data.FFT(inplace=True)
    data.shuffle_analysis(n=num_shuffles, sigmas=sigmas, inplace=True)

def plot_results(data: LCR.Datasheet, choice, sigmas, timeformat):
    """Genera i grafici della curva di luce e spettro di potenza."""
    if choice == "analysis":
        data.plot_analysis(sigmas=sigmas)
    elif choice == "all":
        data.plot_data(timeformat)
        data.plot_spectrum(see_parts=True)
        data.plot_analysis(sigmas=sigmas, see_shuffle=True, timeformat=timeformat)
    elif choice == "data":
        data.plot_data(timeformat)
    elif choice == "apectrum":
        data.plot_spectrum(see_parts=True)

def main():
    parser = argparse.ArgumentParser(description="Analisi di periodicità delle curve di luce dei Blazar")
    parser.add_argument("filepath", type=str, help="Percorso del file CSV della curva di luce")
    parser.add_argument("-p", "--plot", choices=["all", "data", "spectrum", "analysis"], default="analysis", help="Scelta del tipo di plot")
    parser.add_argument("-n", "--shuffles", type=int, default=1000, help="Numero di curve di luce sintetiche")
    parser.add_argument("-s", "--sigmas", type=float, default=3.9, help="Soglia per la significatività del picco in deviazioni")
    parser.add_argument("-c", "--percentage", type=float, help="Soglia per la significatività del picco in percentuale")
    parser.add_argument("-t", "--timeformat", choices=["JD", "Julian Date", "MET", "Mission Elapsed Time"], 
                        default="JD", help="Scelta del formato data, Julian Date o MET")
    
    args = parser.parse_args()
    data = load_data(args.filepath)
    choice = args.plot
    
    if args.percentage is not None:
        args.sigmas = norm.ppf(args.percentage / 100)
    
    analyze_periodicity(data, sigmas=args.sigmas, num_shuffles=args.shuffles)
    
    if args.timeformat is not None and args.timeformat.lower() in ("met", "mission elapsed time"):
        args.timeformat == "met"
        
    plot_results(data, choice, args.sigmas, args.timeformat)
       

if __name__ == "__main__":
    main()