import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_hits = pd.read_csv("hit_times_M0.csv")
print(df_hits.columns)

t = df_hits["hit_time"]
dt = np.array(df_hits["hit_time"][1:]) - np.array(df_hits["hit_time"][:-1])
ldt = np.log10(dt)
ldt[np.isinf(ldt)] = np.nan

fig, ax = plt.subplots(1,2, figsize=(12,9))

ax[0].set_title("Grafico tempi/freq")
ax[0].hist(x=t, bins=100, color="tomato")
ax[0].set_xlabel("tempi di hit [ns]")
ax[0].set_ylabel("frequenza")

ax[1].set_title("Grafico Delta(tempi)/freq")
ax[1].hist(x=ldt, bins=100, color="darkturquoise")
ax[1].set_xlabel("differenza tempi di hit consecutivi [log10(ns)]")
ax[1].set_ylabel("frequenza")

plt.show()