import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import argparse as ap


def main():

    parser = ap.ArgumentParser(description="Selezione Grafico V(t)-t o S(t)-t dal file vel_vs_time.csv", 
                             usage="python3 vel_vs_time.py --opzioni")
    
    parser.add_argument("--vel", "--velocita", "-v", action="store", type=str, help="Genera il grafico velocità-tempo")
    parser.add_argument("--dis", "--distanza", "-d", action="store", type=str, help="Genera il grafico distanza-tempo")
    parser.add_argument("--both", "--entrambi", "-b", action="store", type=str, help="Genera il grafico distanza-tempo")

    args = parser.parse_args()
    
    if args.vel != None:
        path, c = args.vel, 1
    elif args.dis != None:
        path, c = args.dis, 1
    elif args.both != None:
        path, c = args.both, 2
    else:
        print("Nessun file trovato o selezionato")
        return

    df_vel_vs_time = pd.read_csv(path)
    values = df_vel_vs_time.values.transpose()

    fig, ax = plt.subplots(1,c, figsize=(9*c, 8) )
    
    if args.vel != None:
        
        ax.plot(values[0], values[1], color="navy", label="v(t)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Velocità (m/s)")
        ax.set_title("Grafico velocità-tempo")


    elif args.dis != None:
              
        ran = range(1, len(values[0]) + 1)
        distances = map(lambda x: integrate.simpson(values[1][:x], values[0][:x]), ran)
        distances = np.array([values[0], list(distances)])

        ax.plot(distances[0], distances[1], color="tomato", label="x(t)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Distanza (m)")
        ax.set_title("Grafico distanza-tempo")
    
    else:
        ax[0].plot(values[0], values[1], color="navy", label="v(t)")
        ax[0].set_xlabel("Tempo (s)")
        ax[0].set_ylabel("Velocità (m/s)")
        ax[0].set_title("Grafico velocità-tempo")
                      
        ran = range(1, len(values[0]) + 1)
        distances = map(lambda x: integrate.simpson(values[1][:x], values[0][:x]), ran)
        distances = np.array([values[0], list(distances)])

        ax[1].plot(distances[0], distances[1], color="tomato", label="x(t)")
        ax[1].set_xlabel("Tempo (s)")
        ax[1].set_ylabel("Distanza (m)")
        ax[1].set_title("Grafico distanza-tempo")

    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()