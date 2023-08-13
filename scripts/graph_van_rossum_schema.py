#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import miliseconds_to_seconds, Color, PATTERNS
from imports.training_data import generate_base_patterns, generate_training_data
from elephant.kernels import ExponentialKernel
import quantities as pq
from numpy import linspace
import matplotlib.pyplot as plt

EXTRA_DISTANCE = 500
NUM_VALUES = 1000

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns()

# generamos las perturbaciones
print(f'Generando perturbaciones 5ms...')
training_data = generate_training_data(og_patterns=base_patterns, num_patterns=1, max_half_range=10)

# cogemos los dos primeros scallops
spikes1 = training_data['scallop'][0][:-1]
spikes2 = training_data['scallop'][1][:-1]
ranges_to_calculate = linspace(0, max(spikes1 + spikes2) + EXTRA_DISTANCE, NUM_VALUES)

# calculamos sus convoluciones con el kernel de van rossum, tau_r = 100ms
van_rossum_kernel = ExponentialKernel(sigma=100*pq.ms)

# calculamos los valores de cada grafica
print(f'Calculando vanRossum...')
vanRossum1 = {rc : 0/pq.ms for rc in ranges_to_calculate}
vanRossum2 = {rc : 0/pq.ms for rc in ranges_to_calculate}
i = 0
for r in ranges_to_calculate:    
    for sp in spikes1:
        vanRossum1[r] += van_rossum_kernel((r - sp)*pq.ms)
    for sp in spikes2:
        vanRossum2[r] += van_rossum_kernel((r - sp)*pq.ms)
    i += 1
    print(f'Procesado: {round(i*100/NUM_VALUES, 2)}%', end = '\r')

# graficamos
fig, axis = plt.subplots(nrows=3, gridspec_kw = {'wspace':0, 'hspace':0})
fig.set_figwidth(10)
fig.set_figheight(5)
axis[0].sharex(axis[1])
axis[1].sharex(axis[2])
# ponemos los IPIs en la zona superior
for s in spikes1:
    axis[0].axvline(x=s, ymin=1, ymax=1.1, clip_on = False)
for s in spikes2:
    axis[0].axvline(x=s, ymin=1, ymax=1.1, clip_on = False, color='red')
# primera secuencia
axis[0].plot(ranges_to_calculate, vanRossum1.values(), linewidth = 1)
axis[0].set_ylabel(r'$\hat{SPI}_{1} (ms^{-1})$')
# segunda secuencia
axis[1].plot(ranges_to_calculate, vanRossum2.values(), linewidth = 1, color='red')
axis[1].set_ylabel(r'$\hat{SPI}_{2} (ms^{-1})$')
axis[1].invert_yaxis()
# tercera secuencia
L2 = [(s1 - s2)**2 for s1, s2 in zip(vanRossum1.values(), vanRossum2.values())]
axis[2].fill_between(ranges_to_calculate, L2, step='pre', alpha=0.5)
axis[2].plot(ranges_to_calculate, L2, linewidth = 1)
axis[2].set_ylabel(r'$(\hat{SPI}_{1} - \hat{SPI}_{2})^{2}$')
axis[2].set_xlabel(f'Tiempo (s)')
# guardamos
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/grafica_vanRossum.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}grafica_vanRossum.jpg{Color.END} generado en scripts/auxiliar_img/')
