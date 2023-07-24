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

# sacamos el data name y el threshold
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns()

# los perturbamos
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

# graficamos cada uno de los patrones
L_SDF_data = len(SDF_data)
print(f'Graficado de los patrones...')
for p in PATTERNS:
    j=0
    for _, inserted_pattern in zip(range(5), [s[2:] for s in patterns_added if s[0] == p]):
        left_limit, right_limit = int(inserted_pattern[0]), int(inserted_pattern[-1])
        fig = plt.figure()
        r = range(min(right_limit+300, L_SDF_data - 1) - max([left_limit-300, 0]))
        plt.plot(
            r[:left_limit - max([left_limit-300, 0])],
            SDF_data[max([left_limit-300, 0]):left_limit],
            color='blue'
        )
        plt.plot(
            r[left_limit - max([left_limit-300, 0]) : right_limit - max([left_limit-300, 0])],
            SDF_data[left_limit:right_limit],
            color='red',
            label=p
        )
        plt.plot(
            r[right_limit - max([left_limit-300, 0]):],
            SDF_data[right_limit:min(right_limit+300, L_SDF_data - 1)],
            color='blue'
        )
        sp = new_spike_train[bisect_left(new_spike_train, max([left_limit-300, 0])) : bisect_right(new_spike_train, min(right_limit+300, L_SDF_data - 1))]
        sp = [s - max([left_limit-300, 0]) for s in sp]
        for s in sp:
            plt.axvline(x=s, ymin=1, ymax=1.1, clip_on = False)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('SDF (spikes / s)')
        plt.title(f'{p}', x=0.5, y=1.15)
        plt.legend()
        fig.gca().yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
        fig.gca().xaxis.set_major_formatter(miliseconds_to_seconds)
        fig.savefig(f'scripts/auxiliar_img/debug-{p}-{j}-inserted-in-{filename}.jpg', dpi=300) 
        print(f'Archivo {Color.YELLOW}debug-{p}-{j}-inserted-in-{filename}.jpg{Color.END} generado en scripts/auxiliar_img/')
        plt.close(fig=fig)
        gc.collect()
        j += 1

# guardamos el fichero
