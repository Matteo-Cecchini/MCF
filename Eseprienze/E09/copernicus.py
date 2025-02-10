import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.fft import rfft, rfftfreq, irfft

df_data = pd.read_csv("https://raw.githubusercontent.com/s-germani/metodi-computazionali-fisica-2024/refs/heads/main/dati/trasformate_fourier/copernicus_PG_selected.csv")
print(df_data)

# il csv ha come formati per la data o la data convenzionale o la data giuliana; quest'ultima rende il codice più veloce
'''
plt.plot(df_data["date"], df_data[df_data.columns[2:]], label=df_data.columns[2:])
plt.legend()
plt.show()
'''
# elaborazione per inquinante CO
four = rfft(df_data["mean_co_ug/m3"].values)
freq = rfftfreq(len(four))
'''
plt.plot(freq[1:], np.absolute(four[1:len(freq)])**2, '*')
#plt.xscale('log')
plt.yscale('log')
plt.show()
'''

four_p = four.copy()
mask = np.absolute(four_p)**2 < .8e7
four_p[mask] = 0
y = irfft(four_p, n=len(df_data["date"]))

plt.plot(df_data["date"], df_data["mean_co_ug/m3"])
plt.plot(df_data["date"], y)

plt.show()

period = 1/freq

plt.plot(period[1:], np.absolute(four[1:len(period)])**2, '*')
plt.xscale('log')
plt.yscale('log')
plt.show()