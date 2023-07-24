import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import error_print, PATTERNS, intervalIntersection
from imports.support.bursts_finding import carlson_method
from imports.support.density_functions import generate_SDF
from imports.training_data import generate_training_data, get_smallest_maximum_from_perturbations, get_carlson_stats_of_perturbations, get_filters_for_detection
from imports.insert_patterns import insert_patterns
from numpy import mean

# sacamos la carpeta de la que probar los patrones y la señal de fondo
patterns_type = None
filename = None
perturbation_degree = None
N = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--pattern_folder':
        patterns_type = sys.argv[param_pos + 1]
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
    if sys.argv[param_pos] == '--perturbation_degree':
        perturbation_degree = float(sys.argv[param_pos + 1])
    if sys.argv[param_pos] == '--N':
        N = int(float(sys.argv[param_pos + 1]))
if patterns_type == None:
    error_print('Parametro --pattern_folder no detectado')
if filename == None:
    error_print('Parametro --filename no detectado')
if perturbation_degree == None:
    error_print('Parametro --perturbation_degree no detectado')
if N == None:
    error_print('Parametro --N no detectado')

# sacamos sus patrones base
base_patterns = {p : [] for p in PATTERNS}
for p in PATTERNS:
    with open(os.path.join(os.getcwd(), f'modelos de patrones {patterns_type}/{p}.dat'), 'r') as f:
        for line in f:
            base_patterns[p].append([float(s) for s in line.strip().split(' ')])

# sacamos la secuencia de patrones de fondo
background_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        background_spikes.append(float(line.strip()))

# sacamos los N primeros patrones base
base_patterns_N = {p : [base_patterns[p][i] for i in range(N+1)] for p in PATTERNS}

# los perturbamos con el ruido que toca
new_patterns_N = generate_training_data(
    og_patterns=base_patterns_N,
    max_half_range=perturbation_degree,
    fixed_seed=True
)

# sacamos el maximo mas pequeño de los patrones generados para sacar el threshold optimo
optimal_threshold = get_smallest_maximum_from_perturbations(new_patterns=new_patterns_N)*0.7

# sacamos los filtros para estos patrones generados
new_patterns_N_stats = get_carlson_stats_of_perturbations(new_patterns=new_patterns_N)
filters = {p : get_filters_for_detection(new_patterns_stats=new_patterns_N_stats, p=p) for p in PATTERNS}

# los insertamos y guardamos su localizacion en la señal
new_spike_train, patterns_added = insert_patterns(
    patterns_to_insert=new_patterns_N,
    background_sequence=background_spikes
)

# detectamos patrones
patterns_detected = carlson_method(
    spikes=new_spike_train,
    SDF_data=generate_SDF(spikes_train=new_spike_train),
    burst_factor=optimal_threshold,
    dict_of_thresholds=filters
)
# nos quedamos con los centros
patterns_detected = [s[0] for s in patterns_detected]

# sacamos correspondencia
num_patterns_detected_r = len(patterns_detected)
num_inserted_patterns_detected = 0
num_patterns_noise_r = 0
inserted_patterns_detected = []
intersections = []
falsos_negativos = 0
falsos_negativos_p = []
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
        intersections.append(max_intersecion[1]*100/(data_inserted[-1] - data_inserted[2]))
        num_inserted_patterns_detected += 1
        inserted_patterns_detected.append((max_intersecion[0], data_inserted))
    # si la longitud es 0 es porque es ruido
    else:
        falsos_negativos += 1
        falsos_negativos_p.append(data_inserted)
            
# para sacar el ruido, miramos qué detectados no se han asociado con uno insertado
already_removed = []
for r in inserted_patterns_detected:
    if r[0] not in already_removed:
        patterns_detected.remove(r[0])
        already_removed.append(r[0])
num_patterns_noise_r = len(patterns_detected)

# graficamos los falsos negativos
for fn in falsos_negativos_p:
    print(f'{fn[0]} {fn[1]}')

# sacamos por tipos
for _ in range(0, len(patterns_added) - num_patterns_detected_r):
    intersections.append(0)
num_detected_per_type = {p_: len([d for d in inserted_patterns_detected if d[1][0] == p_]) for p_ in PATTERNS}

print(f'{perturbation_degree} {round(optimal_threshold, 2)} {num_patterns_detected_r} {num_inserted_patterns_detected} {num_patterns_noise_r} {falsos_negativos} {mean(intersections)} {num_detected_per_type["scallop"]} {num_detected_per_type["acceleration"]} {num_detected_per_type["rasp"]}')