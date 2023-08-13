#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import error_print, PATTERNS, Color, intervalIntersection, spikes_per_ms_to_spikes_per_s, miliseconds_to_seconds
from imports.support.bursts_finding import carlson_method
from imports.support.density_functions import generate_SDF
from imports.training_data import get_filters_for_detection, generate_base_patterns, generate_training_data, get_carlson_stats_of_perturbations
from imports.insert_patterns import insert_patterns
import matplotlib.pyplot as plt

# sacamos el data name y el threshold
filename = None
manual_threshold = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
    if sys.argv[param_pos] == '--threshold':
        manual_threshold = float(sys.argv[param_pos + 1])
if filename == None:
    error_print('Parametro --filename no detectado')
if manual_threshold == None:
    error_print('Parametro --threshold no detectado')

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns()

# los perturbamos y sacamos los thresholds para cada patron a detectar aqui
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5)
print(f'Sacando filtros...')
vibrated_patterns_5ms_stats = get_carlson_stats_of_perturbations(new_patterns=vibrated_patterns_5ms)
filters = {
    p : get_filters_for_detection(new_patterns_stats=vibrated_patterns_5ms_stats, p=p) for p in PATTERNS
}

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
num_patterns_added = len(patterns_added)
print(f'Numero de patrones añadidos: {Color.YELLOW}{num_patterns_added}{Color.END}')

# sacamos los tipos añadidos
num_scallops_added = len([p for p in patterns_added if p[0] == 'scallop'])
num_acc_added = len([p for p in patterns_added if p[0] == 'acceleration'])
num_rasps_added = len([p for p in patterns_added if p[0] == 'rasp'])

# SDF
print(f'Generando SDF...')
SDF_data = generate_SDF(spikes_train=new_spike_train)

# deteccion
patterns_detected = carlson_method(
    spikes=new_spike_train,
    SDF_data=SDF_data,
    burst_factor=manual_threshold,
    dict_of_thresholds=filters
)
patterns_detected = [q[0] for q in patterns_detected]

# recorremos detectandos falsos positivos y falsos negativos
inserted_patterns_detected = []
for data_inserted in patterns_added:
    # asociamos el patron insertado con el detectado con la interseccion mas grande
    intersection_info = []
    for range_detected in patterns_detected:
        inters = intervalIntersection(range_detected, [data_inserted[2], data_inserted[-1]])
        if inters is not None:
            # si hay interseccion, guardamos los rangos de los mismos
            intersection_info.append([range_detected, inters[1] - inters[0]])
    
    # si la longitud de las intersecciones NO es 0 es porque se ha detectado. Quitamos el insertado
    if len(intersection_info) != 0:
        max_intersecion = max(intersection_info, key=lambda x: x[1])
        inserted_patterns_detected.append((max_intersecion[0], data_inserted))

# para sacar el ruido, miramos qué detectados no se han asociado con uno insertado
already_removed = []
for r in inserted_patterns_detected:
    if r[0] not in already_removed:
        patterns_detected.remove(r[0])
        already_removed.append(r[0])
num_patterns_noise_r = len(patterns_detected)
falsos_positivos = patterns_detected

# graficamos los falsos positivos
fig, axis = plt.subplots(2, 2)
fig.set_figwidth(7)
fig.set_figheight(7)
fig.supxlabel('Tiempo (s)')
fig.supylabel('SDF (spikes / s)', rotation=90)
i = 1
for ax, (left_fp, right_fp), letter in zip([axis[0, 0], axis[0, 1], axis[1, 0], axis[1, 1]], falsos_positivos, ['a', 'b', 'c', 'd']):
    r = range(right_fp - left_fp + 1000)
    ax.plot(r[0:500], SDF_data[left_fp-500:left_fp], color='b')
    ax.plot(r[500:-500], SDF_data[left_fp:right_fp], color='r')
    ax.plot(r[-500:], SDF_data[right_fp:right_fp+500], color='b')
    ax.yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
    ax.xaxis.set_major_formatter(miliseconds_to_seconds)
    ax.set_title(f'{letter}) Ejemplo {i}')
    ax.axhline(y=manual_threshold, linewidth=1)
    i += 1
fig.savefig(f'graficas/{filename}_debug_ruido.jpg', dpi=300)
print(f'Archivo {Color.YELLOW}{filename}_debug_ruido.jpg{Color.END} generado en graficas/')
