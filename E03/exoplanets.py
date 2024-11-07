import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_exoplanets = pd.read_csv("ExoplanetsPars_2024.csv", comment='#')
print(df_exoplanets.columns)
print(df_exoplanets)


# (massa - periodo orbitale) & (semiasse maggiore orbita quadro su massa stellare - periodo orbitale)
'''
exoMass = df_exoplanets['pl_bmassj']
orbPer = df_exoplanets['pl_orbper']
rMaxOnStar = (df_exoplanets['pl_orbsmax']**2)/df_exoplanets['st_mass']

fig, (g1, g2) = plt.subplots( figsize=(12, 9), sharex=True)

g1.scatter(orbPer, exoMass, s=3, c='orangered')
g1.set_xlabel('Periodo Orbitale', fontsize=15)
g1.set_ylabel('Massa esopianeta', fontsize=15)
g1.set_title('Grafico Per.Orb.-Massa', fontsize=20)
g1.set_xscale('log')
g1.set_yscale('log')

g2.set_xlabel('Periodo Orbitale', fontsize=15)
g2.scatter(orbPer, rMaxOnStar, s=3, c='limegreen')
g2.set_ylabel('Rapporto semiasse maggiore d\'orbita - massa stella relativa', fontsize=15)
g2.set_title('Grafico Per.Orb.-Semiasse magg.^2/Massa stellare', fontsize=20)
g2.set_xscale('log')
g2.set_yscale('log')
'''

# grafico massa - periodo orbitale distinto su metodo scoperta
'''
perOrb_T = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Transit']['pl_orbper']
perOrb_RV = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Radial Velocity']['pl_orbper']

mass_T = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Transit']['pl_bmassj']
mass_RV = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Radial Velocity']['pl_bmassj']

fig, g = plt.subplots(figsize=(12, 9))
g.scatter(perOrb_T, mass_T, s=3, c='gold', alpha=0.5)
g.scatter(perOrb_RV, mass_RV, s=3, c='violet', alpha=0.7)

g.set_xlabel('Periodo orbitale esopianeta', fontsize=15)
g.set_ylabel('Massa esopianeta', fontsize=15)

g.set_title('grafico massa-per.orb. distinto su metodo scoperta', fontsize=20)
g.set_xscale('log')
g.set_yscale('log')
g.legend(['Transit', 'Radial Velocity'], markerscale=5, fontsize=10)
'''

# istogramma pianeti

massTransit = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Transit']['pl_bmassj']
massRadVel = df_exoplanets.loc[df_exoplanets['discoverymethod'] == 'Radial Velocity']['pl_bmassj']

massTransit = np.log10(massTransit)
massRadVel = np.log10(massRadVel)

fig, g = plt.subplots(figsize=(12,9))

g.hist(massTransit, bins=50, color='gold', alpha=0.5)
g.hist(massRadVel, bins=50, color='darkturquoise', alpha=0.5)

g.set_xlabel('log10 Massa esopianeta', fontsize=15)
g.set_ylabel('Frequenza', fontsize=15)
g.set_title('Istogramma massa esopianeti', fontsize=20)
g.legend(['Transit', 'Radial Velocity'], markerscale=15, fontsize=15)

plt.savefig('Exoplanets_5')
plt.show()