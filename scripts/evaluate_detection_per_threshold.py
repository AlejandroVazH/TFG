import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import Color, PATTERNS, error_print, intervalIntersection
from imports.support.bursts_finding import carlson_method
from imports.support.density_functions import generate_SDF
from imports.training_data import get_filters_for_detection, generate_base_patterns, generate_training_data, get_carlson_stats_of_perturbations
from imports.insert_patterns import insert_patterns
import matplotlib.pyplot as plt
from numpy import mean, linspace

################################################################################
# FUNCIONES AUXILIARES Y MACROS
################################################################################

NUM_PARAMETERS_TO_EVALUATE = 70

################################################################################
# PARTE DE SCRIPT
################################################################################

# sacamos el archivo matriz desde el que añadir el surrogated y shuffled
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(num_of_bases=1500)

# rango de thresholds a medir
thresholds_range = linspace(start=0.015, stop=0.090, num=NUM_PARAMETERS_TO_EVALUATE)

# los perturbamos y sacamos los filtros para cada patron a detectar aqui
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5, num_patterns=1)
print(f'Sacando filtros...')
vibrated_patterns_5ms_stats = get_carlson_stats_of_perturbations(new_patterns=vibrated_patterns_5ms)
filters = {
    p : get_filters_for_detection(new_patterns_stats=vibrated_patterns_5ms_stats, p=p) for p in PATTERNS
}

# hacemos esto para cada archivo
for suffix in ['', '-surrogated', '-shuffled']:

    # sacamos la secuencia de patrones de fondo
    print(f'Sacando la secuencia de fondo sufijo {suffix}...')
    background_spikes = []
    with open(os.path.join(os.getcwd(), f'resultados/{filename}{suffix}_spikes.dat'), 'r') as f:
        for line in f:
            background_spikes.append(float(line.strip()))

    # los insertamos
    print(f'Insertando patrones sufijo {suffix}...')
    new_spike_train, patterns_added = insert_patterns(
        patterns_to_insert=vibrated_patterns_5ms,
        background_sequence=background_spikes
    )
    num_patterns_added = len(patterns_added)
    num_scallops_added = len([p for p in patterns_added if p[0] == 'scallop'])
    num_acc_added = len([p for p in patterns_added if p[0] == 'acceleration'])
    num_rasps_added = len([p for p in patterns_added if p[0] == 'rasp'])
    print(f'Numero de patrones añadidos: {Color.YELLOW}{num_patterns_added}{Color.END}')
    print(f'Numero de scallops añadidos: {Color.YELLOW}{num_scallops_added}{Color.END}')
    print(f'Numero de acc añadidos: {Color.YELLOW}{num_acc_added}{Color.END}')
    print(f'Numero de rasps añadidos: {Color.YELLOW}{num_rasps_added}{Color.END}')

    # SDF
    print(f'Generando SDF...')
    SDF_data = generate_SDF(spikes_train=new_spike_train)

    # recorremos los parametros
    num_patterns_detected = []
    num_patterns_detected_from_inserted = []
    num_patterns_noise = []
    num_falsos_negativos = []
    intersections_avg = []
    scallop_detected_from_inserted = []
    acc_detected_from_inserted = []
    rasp_detected_from_inserted = []
    print(f'Comenzamos la evaluacion de parametros sufijo {suffix}...')
    i = 0
    for r in thresholds_range:
        # hacemos el analisis POR TIPO DE PATRON y luego juntamos los resultados eliminando repeticiones
        patterns_detected = carlson_method(
            spikes=new_spike_train,
            SDF_data=SDF_data,
            burst_factor=r,
            dict_of_thresholds=filters
        )
        # nos quedamos con los centros
        patterns_detected = [q[0] for q in patterns_detected]

        # miramos la correspondencia
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
        
        # guardamos los datos
        num_patterns_detected.append(num_patterns_detected_r)
        num_patterns_detected_from_inserted.append(num_inserted_patterns_detected)
        num_patterns_noise.append(num_patterns_noise_r)
        num_falsos_negativos.append(falsos_negativos)
        for _ in range(0, num_patterns_added - num_patterns_detected_r):
            intersections.append(0)
        intersections_avg.append(mean(intersections))
        
        # sacamos los porcentajes por tipo de patron
        num_detected_per_type = {p_: len([d for d in inserted_patterns_detected if d[1][0] == p_]) for p_ in PATTERNS}
        scallop_detected_from_inserted.append(num_detected_per_type['scallop'])
        acc_detected_from_inserted.append(num_detected_per_type['acceleration'])
        rasp_detected_from_inserted.append(num_detected_per_type['rasp'])

        # indicamos cuanto llevamos
        i += 1
        print(f'Llevamos un {Color.YELLOW}{round(i*100/NUM_PARAMETERS_TO_EVALUATE, 2)}{Color.END}% de parametros evaluados', end='\r')

    # graficamos los resultados
    fig, x_axis = plt.subplots(ncols=2)
    fig.set_figwidth(10)
    # hacemos que compartan eje x las 
    x_axis[1].sharex(x_axis[1])
    # grafica de deteccion en general
    r = [round(r, 4) for r in thresholds_range[::10]]
    # grafica de porcentajes en general
    x_axis[0].set_xlabel('Valores del threshold')
    x_axis[0].set_xticks(r, labels=r, fontsize=10, rotation=90)
    x_axis[0].set_ylabel('Porcentajes de deteccion')
    x_axis[0].set_title(f'a) Porcentajes (general)', x=0.5, y=1.15)
    x_axis[0].plot(
        thresholds_range,
        [s*100/num_patterns_added for s in num_patterns_detected_from_inserted],
        label=f'%insertados y detectados\nrespecto a insertados',
        linewidth=1
    )
    x_axis[0].plot(
        thresholds_range,
        [(n*100/i if i != 0 else 0) for n, i in zip(num_patterns_noise, num_patterns_detected)], label=f'%detectados no insertados\nrespecto a detectados',
        linewidth=1
    )
    x_axis[0].legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=2,
        fontsize=8
    )
    # grafica de porcentajes por tipos
    x_axis[1].set_xlabel('Valores del threshold')
    x_axis[1].set_xticks(r, labels=r, fontsize=10, rotation=90)
    x_axis[1].set_title(f'b) Porcentajes (por tipos)', x=0.5, y=1.15)
    x_axis[1].plot(
        thresholds_range,
        [s*100/num_scallops_added for s in scallop_detected_from_inserted],
        label=f'porcentajes scallop',
        linewidth=1
    )
    x_axis[1].plot(
        thresholds_range,
        [s*100/num_acc_added for s in acc_detected_from_inserted],
        label=f'acceleration',
        linewidth=1
    )
    x_axis[1].plot(
        thresholds_range,
        [s*100/num_rasps_added for s in rasp_detected_from_inserted],
        label=f'rasp',
        linewidth=1
    )
    x_axis[1].legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=3
    )
    # guardamos
    fig.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/evaluacion-threshold{suffix}.jpg'), dpi=300)
    print(f'Archivo {Color.YELLOW}evaluacion-threshold{suffix}.jpg{Color.END} generado en scripts/auxiliar_img/')
