#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.density_functions import generate_SDF
from imports.support.utils import PATTERNS, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s, Color, error_print
from imports.training_data import generate_base_patterns, generate_training_data
from imports.insert_patterns import insert_patterns
import matplotlib.pyplot as plt
import gc
from bisect import bisect_left, bisect_right
from numpy import array, linspace, interp, diff


# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns()

# los perturbamos
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5)

# nos quedamos con los IPIs del primer scallop
spikes = vibrated_patterns_5ms['scallop'][0][:-1]
ipis = [spikes[0]] + [spikes[i+1] - spikes[i] for i in range(len(spikes)-1)]
T_max = 1000
num_points_interp = 20
normalized_spikes = array(spikes)*T_max/max(spikes)
normalized_ipis = [normalized_spikes[0]] + [normalized_spikes[i+1] - normalized_spikes[i] for i in range(len(normalized_spikes) - 1)]
# vamos a guardar sus interpolaciones derivadas
interpolation_range = linspace(start=0, stop=T_max, num=num_points_interp)
interpolation = interp(x=interpolation_range, xp=normalized_spikes, fp=normalized_ipis)
derivation = diff(interpolation)/diff(interpolation_range)

# graficamos
print(f'Graficado de los patrones...')
fig, x_axis = plt.subplots(ncols=4)
fig.set_figwidth(14)
fig.set_figheight(3)
fig.supxlabel('Tiempo (s)')
fig.supylabel('IPI (ms)')
x_axis[0].xaxis.set_major_formatter(miliseconds_to_seconds)
x_axis[1].xaxis.set_major_formatter(miliseconds_to_seconds)
x_axis[2].xaxis.set_major_formatter(miliseconds_to_seconds)
x_axis[3].xaxis.set_major_formatter(miliseconds_to_seconds)
# grafica original
x_axis[0].plot(spikes, ipis, '--o')
x_axis[0].set_title('a) SPI original')
# grafica normalizada
x_axis[1].plot(normalized_spikes, normalized_ipis, '--o')
x_axis[1].set_title('b) SPI longitud normalizada')
# grafica interpolada
x_axis[2].plot(interpolation_range, interpolation, '--o')
x_axis[2].set_title('c) SPI interpolada')
# grafica derivada
x_axis[3].plot(interpolation_range[1:], derivation, '--o')
x_axis[3].set_title('d) SPI derivada')
# guardamos
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/grafica_esquema_fitness.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}grafica_esquema_fitness.jpg{Color.END} generado en scripts/auxiliar_img/')
