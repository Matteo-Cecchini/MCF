import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ODE

# funzioni potenziale
uno = lambda x: (-1)**(x//1)
due = lambda x: x

# funzione filtro passa basso
def vfun(vo, t, rc = 4, vinf = due):
    return (vinf(t) - vo) / rc

# valori RC
RC = ( 4, 1, .25 )

# intervallo del tipo [a,b]
a = 0
b = 10
y0=0
h = (b - a) / 100
t = np.arange(a, b, h)
vin = list(map(due, t))

df = {"t": t, "Vin": vin}

for i in RC:
    sol = ODE.Euler(vfun, y0, a, b, i)
    df[f"Vout_{i}"] = sol[0]
    plt.plot(sol[1], sol[0], label=f"curva con RC={i}")
plt.legend()
plt.show()

dataframe = pd.DataFrame(df)
dataframe.to_csv("filtro_passa_basso_due.csv", index=False)