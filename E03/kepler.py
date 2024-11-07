import pandas as pd
import matplotlib.pyplot as plt

df_kepler = pd.read_csv('kplr010666592-2011240104155_slc.csv')
df_keplerF = df_kepler.loc[(df_kepler['TIME'] < 954.85) & (df_kepler['TIME'] > 954.65)] #per KeplerPNG 4

print(df_kepler.columns)
x, y = df_kepler['TIME'], df_kepler['SAP_FLUX']
yeb = df_kepler['SAP_FLUX_ERR']
xf, yf = df_keplerF['TIME'], df_keplerF['SAP_FLUX']
yebf = df_keplerF['SAP_FLUX_ERR']

title = 'Time - Electron_flux/Seconds graph (dot w errorbars)'

#KeplerPNG_1
'''
n = 1
fig, ax = plt.subplots(figsize=(12.5, 9))
ax.plot(x, y, color='gold')
ax.set_xlabel('TIME', fontsize=15)
ax.set_ylabel('ELECTRON FLUX PER SECOND', fontsize=15)
ax.set_title(title, fontsize=20)
'''

#KeplerPNG_2
'''
n = 2
fig, ax = plt.subplots(figsize=(12.5, 9))
ax.plot(x, y, 'o', color='violet', markersize=2.5)
ax.set_xlabel('TIME', fontsize=15)
ax.set_ylabel('ELECTRON FLUX PER SECOND', fontsize=15)
ax.set_title(title, fontsize=20)
'''

#KeplerPNG_3
'''
n = 3
fig, ax = plt.subplots(figsize=(12.5, 9))
ax.errorbar(x, y, yeb, None, 'o', color='darkturquoise', markersize=2.5)
ax.set_xlabel('TIME', fontsize=15)
ax.set_ylabel('ELECTRON FLUX PER SECOND', fontsize=15)
ax.set_title(title, fontsize=20)
'''

#KeplerPNG_4
'''
n = 4
fig, ax = plt.subplots(figsize=(12.5, 9))
ax.errorbar(xf, yf, yebf, None, 'o', color='limegreen', markersize=2.5)
ax.set_xlabel('TIME', fontsize=15)
ax.set_ylabel('ELECTRON FLUX PER SECOND', fontsize=15)
ax.set_title(title, fontsize=20)
'''

#KeplerPNG_5

n = 5
fig, ax = plt.subplots(figsize=(12.5, 9))
ins_ax = ax.inset_axes([0.065, 0.065, 0.2, 0.6])
ins_ax.errorbar(xf, yf, yebf, None, 'o', color='orange', markersize=1.5)
ax.errorbar(x, y, yeb, None, 'o', color='indianred', markersize=2.5)
ax.set_xlabel('TIME', fontsize=15)
ax.set_ylabel('ELECTRON FLUX PER SECOND', fontsize=15)
ax.set_title(title, fontsize=20)
ax.indicate_inset_zoom(ins_ax, )


plt.savefig(f'KeplerPNG_{n}')
plt.show()