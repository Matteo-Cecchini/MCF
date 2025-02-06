import LCRanalysis as LCR
import pandas as pd
import numpy as np
from time import time

df = pd.read_csv("monthly/4FGL_J0137.0+4751_monthly_12_27_2024.csv")

t = time()
mask = df["Energy Flux Error"] == "-"
df.loc[mask, " Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"] = df.loc[mask, "Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"].str.replace("<", "").astype(float)
df.loc[mask, "Photon Flux Error"] = np.nan
t = time() - t

a = LCR.read_csv("lcr","monthly/4FGL_J0137.0+4751_monthly_12_27_2024.csv")
a.convert_to_numeric(inplace=True)

print(t)
print(df["Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"].values)
print(a)