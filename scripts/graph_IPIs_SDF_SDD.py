import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import PATTERNS, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s, Color
from imports.support.density_functions import generate_SDF, generate_SDD
from imports.training_data import generate_base_patterns, generate_training_data
import matplotlib.pyplot as plt

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True, num_of_bases=5)

# generamos sus perturbaciones
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5)

# los graficamos
print(f'Los graficamos...')
fig, x_axis = plt.subplots(nrows=3, ncols=3)
fig.set_figwidth(12)
fig.set_figheight(6)
for p, i, p_str in zip(PATTERNS, range(3), ['Scallop', 'Aceleración', 'Rasp']):
    # Ajustes de los ejes X de la columna
    x_axis[0,i].sharex(x_axis[1,i])
    x_axis[1,i].sharex(x_axis[2,i])
    x_axis[2,i].set_xlabel('Tiempo (s)')
    x_axis[0,i].xaxis.set_major_formatter(miliseconds_to_seconds)
    x_axis[1,i].xaxis.set_major_formatter(miliseconds_to_seconds)
    x_axis[2,i].xaxis.set_major_formatter(miliseconds_to_seconds)
    # ajustes del eje Y
    x_axis[0,i].yaxis.set_major_formatter(miliseconds_to_seconds)
    x_axis[1,i].yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
    x_axis[2,i].yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
    if p == 'scallop':
        x_axis[0,i].set_ylabel('Duracion del IPI (s)')
        x_axis[1,i].set_ylabel('Valor SDF (spikes / s)')
        x_axis[2,i].set_ylabel('Valor SDD (spikes / s²)')
    # sacamos los ipis
    spikes = vibrated_patterns_5ms[p][0][:-1]
    ipis = [spikes[0]] + [spikes[i+1] - spikes[i] for i in range(len(spikes)-1)]
    # grafica de ipis
    x_axis[0,i].plot(spikes, ipis, '--o')
    for s in spikes:
        x_axis[0,i].axvline(x=s, ymin=1, ymax=1.1, clip_on = False)
    x_axis[0,i].text(x=0, y=1.1, s=f'{i+1})', transform=x_axis[0,i].transAxes)
    x_axis[0,i].set_title(p_str, x=0.5, y=1.1)
    # grafica SDF
    spikes_SDF = generate_SDF(spikes_train=spikes)
    r = range(len(spikes_SDF))
    x_axis[1, i].plot(r, spikes_SDF)
    x_axis[1,i].text(x=0, y=1.1, s=f'{i+4})', transform=x_axis[1,i].transAxes)
    # grafica SDD
    spikes_SDD = generate_SDD(spikes_train=spikes)
    x_axis[2,i].plot(r, spikes_SDD)
    x_axis[2,i].text(x=0, y=1.1, s=f'{i+7})', transform=x_axis[2,i].transAxes)
# guardamos todo
fig.tight_layout()
fig.savefig(f'graficas/visualizacion-transformaciones-Carlson.jpg', dpi=300)
print(f'Archivo {Color.YELLOW}visualizacion-transformaciones-Carlson.jpg{Color.END} generado en graficas/')