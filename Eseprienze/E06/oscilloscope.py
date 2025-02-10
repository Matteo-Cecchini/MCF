import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def DFC(y, x, n):
    yy = (y[n:] - y[:-n]) / (x[n:] - x[:-n])
    if n > 1:
        y0 = [(y[i] - y[0]) / (x[i] - x[0]) for i in range(int(n/2))]
        y1 = [(y[i - n] - y[-n]) / (x[i - n] - x[-n]) for i in range(int(n/2))]
    else:
        y0 = []
        y1 = [(y[-1] - y[-2]) / (x[-1] - x[-2])]
    return np.concatenate([y0, yy, y1])

df_scope = pd.read_csv("E06/oscilloscope.csv")

d1 = DFC(df_scope["signal1"].values, df_scope["time"].values, 500)
d2 = DFC(df_scope["signal2"].values, df_scope["time"].values, 500)

min_map = [i == 0 for i in d1]
min_map1 = df_scope["signal1"][min_map]
min_map = [i == 0 for i in d2]
min_map2 = df_scope["signal2"][min_map]
print(min_map1, min_map2)
print(df_scope.loc[df_scope["signal1"] == df_scope["signal2"]])

plt.plot(df_scope["time"], df_scope["signal1"], color="gold", label="canale 1")
plt.plot(df_scope["time"], d1, color="tomato", label="derivata canale 1")
plt.plot(df_scope["time"], df_scope["signal1"], color="skyblue", label="canale 1")
plt.plot(df_scope["time"], d2, color="navy", label="derivata canale 2")
plt.xlabel("t (us?)")
plt.ylabel("ddp (V)")
plt.grid()
plt.legend()
plt.show()