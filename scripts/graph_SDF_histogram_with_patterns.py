#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import Color, spikes_per_ms_to_spikes_per_s, error_print
from imports.support.density_functions import generate_SDF
from imports.training_data import generate_base_patterns, generate_training_data
from imports.insert_patterns import insert_patterns
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

################################################################################
# PARTE DE SCRIPT
################################################################################

# sacamos la señal de fondo
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --data_name no detectado')

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True, num_of_bases=5)

# generamos sus perturbaciones
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5)

# sacamos la secuencia de patrones de fondo
print(f'Sacando la secuencia de fondo...')
background_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        background_spikes.append(float(line.strip()))

# los insertamos
print(f'Insertando patrones...')
new_spike_train, patterns_added = insert_patterns(
    patterns_to_insert=vibrated_patterns_5ms,
    background_sequence=background_spikes
)
print(f'Numero de patrones añadidos: {Color.YELLOW}{len(patterns_added)}{Color.END}')

# SDF
print(f'Generando SDF...')
SDF_data = generate_SDF(spikes_train=new_spike_train)

# graficamos su histograma
plt.hist(SDF_data, bins='auto')
plt.xlabel('Valores de la SDF (spikes / s)')
plt.ylabel('Cantidad de ocurrencias')
plt.xlim((0, 0.03))
plt.gca().xaxis.set_major_formatter(FuncFormatter(spikes_per_ms_to_spikes_per_s))
plt.savefig(os.path.join(os.getcwd(), f'graficas/{filename}-patterns_SDF_histogram.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}{filename}-patterns_SDF_histogram.jpg{Color.END} generado en graficas/')
