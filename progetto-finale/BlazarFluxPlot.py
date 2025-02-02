import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LAT import LATdatasheet
import LAT
from scipy.fft import rfft, irfft, rfftfreq


wdfs = LAT.read_csv("weekly/4FGL_J0137.0+4751_weekly_12_27_2024.csv")
mdfs = LAT.read_csv("monthly/4FGL_J0137.0+4751_monthly_12_27_2024.csv")

used_cols = ["Julian Date", "Energy Flux [0.1-100 GeV](MeV cm-2 s-1)"]

a = wdfs[used_cols]
a = a.to_numeric()
a.wf_plot(c="r")
print(a)
a = a.fft_analysis()
a.ps_plot()
print(a)