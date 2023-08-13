#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import error_print, PATTERNS, intervalIntersection, Color, miliseconds_to_seconds
from imports.support.bursts_finding import carlson_method
from imports.support.density_functions import generate_SDF
from imports.training_data import generate_training_data, get_smallest_maximum_from_perturbations, get_carlson_stats_of_perturbations, get_filters_for_detection, generate_base_patterns
from imports.insert_patterns import insert_patterns
from numpy import mean
import matplotlib.pyplot as plt

NOISES = [5, 7, 9, 11, 13, 15]

# sacamos el nombre del archivo matriz al que luego añadir los prefijos -surrogate y -shuffled
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(num_of_bases=1500)

# hacemos esto para cada archivo
for suffix in ['', '-surrogated', '-shuffled']:

    # sacamos la secuencia de patrones de fondo
    print(f'Sacando la secuencia de fondo sufijo {suffix}...')
    background_spikes = []
    with open(os.path.join(os.getcwd(), f'resultados/{filename}{suffix}_spikes.dat'), 'r') as f:
        for line in f:
            background_spikes.append(float(line.strip()))

    # perturbamos, insertamos y detectamos para diferentes numeros de patrones y ruido
    per_true_positives, per_false_positives, per_false_negatives = [], [], []
    per_scallops_detected, per_acc_detected, per_rasps_detected = [], [], []
    # recorremos las perturbaciones
    i = 0
    for perturbation_degree in NOISES:
        # los perturbamos con el ruido que toca
        print(f'Generando vibracion con vibracion {perturbation_degree}...')
        new_patterns_N = generate_training_data(
            og_patterns=base_patterns,
            max_half_range=perturbation_degree,
            num_patterns=1,
            fixed_seed=True
        )

        # sacamos el maximo mas pequeño de los patrones generados para sacar el threshold optimo
        print(f'Threshold ideal con vibracion {perturbation_degree}...')
        optimal_threshold = get_smallest_maximum_from_perturbations(new_patterns=new_patterns_N)*0.7

        # sacamos los filtros para estos patrones generados
        print(f'Sacando stats con vibracion {perturbation_degree}...')
        new_patterns_N_stats = get_carlson_stats_of_perturbations(new_patterns=new_patterns_N)
        print(f'Sacando filtros con vibracion {perturbation_degree}...')
        filters = {p : get_filters_for_detection(new_patterns_stats=new_patterns_N_stats, p=p) for p in PATTERNS}

        # los insertamos y guardamos su localizacion en la señal
        print(f'Insertando patrones con vibracion {perturbation_degree}...')
        new_spike_train, patterns_added = insert_patterns(
            patterns_to_insert=new_patterns_N,
            background_sequence=background_spikes
        )
        L_patterns_added = len(patterns_added)
        num_scallops_added = len([s for s in patterns_added if s[0] == "scallop"])
        num_acc_added = len([s for s in patterns_added if s[0] == "acceleration"])
        num_rasps_added = len([s for s in patterns_added if s[0] == "rasp"])
        print(f'Numero de patrones añadidos: {Color.YELLOW}{L_patterns_added}{Color.END}')
        print(f'Numero de scallops: {Color.YELLOW}{num_scallops_added}{Color.END}')
        print(f'Numero de aceleraciones: {Color.YELLOW}{num_acc_added}{Color.END}')
        print(f'Numero de rasps: {Color.YELLOW}{num_rasps_added}{Color.END}')

        # detectamos patrones
        print(f'Detectando patrones con vibracion {perturbation_degree}...')
        patterns_detected = carlson_method(
            spikes=new_spike_train,
            SDF_data=generate_SDF(spikes_train=new_spike_train),
            burst_factor=optimal_threshold,
            dict_of_thresholds=filters
        )
        # nos quedamos con los centros
        patterns_detected = [s[0] for s in patterns_detected]

        # sacamos correspondencia
        print(f'Detectando patrones insertados y detectados...')
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
        print(f'Sacando ruido...')
        already_removed = []
        for r in inserted_patterns_detected:
            if r[0] not in already_removed:
                patterns_detected.remove(r[0])
                already_removed.append(r[0])
        num_patterns_noise_r = len(patterns_detected)

        # guardamos los datos
        for _ in range(0, len(patterns_added) - num_patterns_detected_r):
            intersections.append(0)
        num_detected_per_type = {p_: len([d for d in inserted_patterns_detected if d[1][0] == p_]) for p_ in PATTERNS}
        
        # los mostramos para el debug
        print(f'{Color.CYAN}{perturbation_degree} {round(optimal_threshold*1000, 2)} {num_patterns_detected_r} {num_inserted_patterns_detected} {num_patterns_noise_r} {falsos_negativos} {mean(intersections)} {num_detected_per_type["scallop"]} {num_detected_per_type["acceleration"]} {num_detected_per_type["rasp"]}{Color.END}')

        per_true_positives.append(num_inserted_patterns_detected / L_patterns_added)
        per_false_positives.append(num_patterns_noise_r / L_patterns_added)
        per_false_negatives.append(falsos_negativos / L_patterns_added)
        per_scallops_detected.append(num_detected_per_type["scallop"] / num_scallops_added)
        per_acc_detected.append(num_detected_per_type["acceleration"] / num_acc_added)
        per_rasps_detected.append(num_detected_per_type["rasp"] / num_rasps_added)

        # mostramos cuanto llevamos
        i += 1
        print(f'Llevamos un {Color.YELLOW}{round(i*100/len(NOISES), 2)}{Color.END}% de perturbaciones evaluadas sufijo {suffix}')

    # graficamos
    fig, x_axis = plt.subplots(ncols=2)
    fig.set_figwidth(10)
    x_axis[0].sharey(x_axis[1])
    x_axis[0].plot(NOISES, per_true_positives, '--o', linewidth=0.5, label='TP', alpha=0.3)
    x_axis[0].plot(NOISES, per_false_positives, '--o', linewidth=0.5, label='FP', alpha=0.3)
    x_axis[0].plot(NOISES, per_false_negatives, '--o', linewidth=0.5, label='FN', alpha=0.3)
    x_axis[0].legend()
    x_axis[1].plot(NOISES, per_scallops_detected, '--o', linewidth=0.5, label='scallops', alpha=0.3)
    x_axis[1].plot(NOISES, per_acc_detected, '--o', linewidth=0.5, label='acc', alpha=0.3)
    x_axis[1].plot(NOISES, per_rasps_detected, '--o', linewidth=0.5, label='rasps', alpha=0.3)
    x_axis[1].legend()
    fig.supylabel('Porcentajes sobre cantidades insertadas')
    fig.supxlabel('vibración (ms)')
    fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/evaluacion-vibracion{suffix}.jpg'), dpi = 300)
    print(f'Archivo {Color.YELLOW}evaluacion-vibracion{suffix}.jpg{Color.END} generado en scripts/auxiliar_img/')
