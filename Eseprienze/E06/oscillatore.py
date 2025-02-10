import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import argparse

def parse():
    parser = argparse.ArgumentParser(description="visualizza grafico potenziale-posizione e periodo-posizione iniziale",
                                     usage="python3 oscillatore.py const power init.pos mass")
    parser.add_argument("-c", "--custom", action="store", nargs='*', type=float)
    return parser.parse_args()

def main():
    args = parse()
    
    if args.custom != None:
        k = args.custom[0]
        pw = args.custom[1]
        x0 = args.custom[2]
        mass = args.custom[3]
    else:
        k = 0.1
        pw = 6
        x0 = 4.5
        mass = 1
    
    v0 = k*x0**pw
    coeff = np.sqrt(8*mass)
        
    xx = np.linspace(0, x0, 500)
    vv = k*xx**pw
    tt = 1/np.sqrt(v0 - vv)
    tt[np.isinf(tt)] = 0
    t0 = coeff*intg.simpson(tt, xx)
    print(t0)
    
    fig, ax = plt.subplots(1,2, figsize=(10, 5) )

    xx = np.arange(-5, 5.05, 0.1)
    vv = k*xx**pw
    
    ax[0].plot(xx, vv, color='slategray')
    ax[0].axvline(color='k', linewidth=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel(r'V(x)')
    ax[0].plot(x0, v0, 'o', markersize=12, color='tomato')

    xx = np.arange(0, 5.05, 0.02)
    vv = k*xx**pw
    tt = 1/np.sqrt(v0 - vv)
    tt[np.isinf(tt)] = 0
    tt = [coeff*intg.simpson(tt[:i], xx[:i]) for i in range(1, len(xx) + 1)]
    
    ax[1].plot(xx, tt, color='slategray')
    ax[1].axvline(color='k', linewidth=0.5)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel(r't')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()