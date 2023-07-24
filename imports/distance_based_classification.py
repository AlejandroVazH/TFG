import os
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance, van_rossum_distance, SpikeTrain
from typing import Union
from bisect import bisect_left, bisect_right
from imports.support.utils import error_print, PATTERNS, Color, DISABLE_TIME_PRINTS
from imports.support.bursts_finding import burst_finding
import matplotlib.pyplot as plt

################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

def __graph_detected_patterns(classification: list[list, str], distance: str) -> None:
    fig, x_axis = plt.subplots()

    # ipis de los patrones detectados
    classification_per_type = {p : [classification[idx][0] for idx in range(len(classification)) if classification[idx][1] == p] for p in PATTERNS}
    classification_ipis_per_type ={p : [[spikes[0]] + [spikes[i+1] - spikes[i] for i in range(len(spikes) - 1)] for spikes in classification_per_type[p]] for p in PATTERNS}

    for p in PATTERNS:
        for p_class_idx in range(len(classification_per_type[p])):
            x_axis.plot(classification_per_type[p][p_class_idx], classification_ipis_per_type[p][p_class_idx], alpha=0.5)
        x_axis.set_xlabel(f'Tiempo (ms)')
        x_axis.set_ylabel(f'IPI (ms)')
        x_axis.set_title(f'Patrones {p} detectados en la secuencia')
        fig.set_figwidth(10)
        fig.set_figheight(8)
        fig.savefig(os.path.join(os.getcwd(), f'graficas/{p}s_detectados_por_{distance}.jpeg'))
        print(f'Archivo {Color.YELLOW}{p}s_detectados_por_{distance}.jpeg{Color.END} generado en graficas/')
        x_axis.cla()

    return

################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

#
# Clasificacion por distancia minima de Victor-Purpura o Van Rossum a los patrones de referencia
#
def distance_based_classification(
    # datos de este metodo
    filename: str,
    pattern_folder: str,
    save: bool = True,
    # distancia a utilizar, por defecto Victor Purpura (VP)
    distance: str = 'VP',
    # extraccion de spikes
    get_spikes_from_file: bool = False,
    spikes_train: Union[list, None] = None,
    # datos de los bursts
    get_bursts_times_ranges_from_file: bool = False,
    compute_bursts_times_ranges: bool = False,
    bursts_finding_mode: str = 'carlson',
    bursts_times_ranges: Union[list, None] = None,
    save_bursts_times_ranges: bool = False,
) -> list[list, str]:
    # sacamos el tren de spikes de la original.
    if get_spikes_from_file:
        if filename == None:
            error_print('Extraccion de los spikes por fichero activada pero no se ha pasado el parametro filename')
        spikes_train = []
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_spikes.dat"), 'r') as f:
            for line in f:
                spikes_train.append(float(line.strip()))
    elif type(spikes_train) == None:
        error_print('No se ha pasado argumento spikes_name con parametro get_spikes_from_file a False')

    # extraccion de los bursts
    # Si se ha efectuado con rebust_gaussian_surprise son literalmente los spikes de la señal
    if get_bursts_times_ranges_from_file:
        if filename == None:
            error_print('Extraccion de los maximos de los bursts por fichero activada pero no se ha pasado el parametro filename')
        bursts_times_ranges = []
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_bursts.dat"), 'r') as f:
            for line in f:
                l = line.strip().split(' ')
                bursts_times_ranges.append([float(l[0]), float(l[1])])
    elif compute_bursts_times_ranges:
        bursts_times_ranges = burst_finding(
            get_spikes_from_file=get_spikes_from_file,
            compute_SDF=True,
            filename=filename,
            bursts_finding_mode=bursts_finding_mode,
            save=save_bursts_times_ranges
        )
    elif type(bursts_times_ranges) == type(None):
        error_print('No se ha recibido bursts_times_ranges con el parametro get_bursts_times_ranges_from_file y compute_bursts_times_ranges sin especificar True')
    
    # sacamos los spikes de los bursts y los normalizamos llevandolos al 0
    data_bursts_spikes = []
    for left_limit, right_limit in bursts_times_ranges:
        data_bursts_spikes.append(spikes_train[bisect_left(spikes_train, left_limit):bisect_right(spikes_train, right_limit)])
    # normalizado: se toma como referencia el spike anterior, o el 0 si era el primero
    data_bursts_spikes_indexes = [[[idx for idx in range(len(spikes_train)) if spikes_train[idx] == b][0] for b in burst_d] for burst_d in data_bursts_spikes]
    data_bursts_spikes_normalized = []
    for burst_d_idx in data_bursts_spikes_indexes:
        if burst_d_idx[0] == 0:
            data_bursts_spikes_normalized.append([spikes_train[idx] for idx in burst_d_idx])
        else:
            data_bursts_spikes_normalized.append([spikes_train[idx] - spikes_train[burst_d_idx[0] - 1] for idx in burst_d_idx])

    # sacamos los spikes de referencia
    reference_spikes = {}
    for p in PATTERNS:
        reference_spikes[p] = []
        with open(os.path.join(os.getcwd(), f"simulaciones {pattern_folder}/{p}_spikes.dat"), 'r') as pf:
            for line in pf:
                reference_spikes[p].append([float(s) for s in line.strip().split(' ')])
    
    # para cada burst, calculamos su distancia con TODOS los patrones de referencia, y lo clasificamos en el de menor distancia
    classification = []
    classification_for_plotting = []
    progress_calculating_distances = 0
    L_data_bursts_spikes_normalized = len(data_bursts_spikes_normalized)
    if distance == 'VP':
        # lo hacemos por indices para sacar los originales
        for data_bs_idx in range(L_data_bursts_spikes_normalized):
            # clase necesaria para el modulo, t_stop puesto para asegurar que funciona
            data_bs_ST = SpikeTrain(data_bursts_spikes_normalized[data_bs_idx], units='ms', t_stop = max(data_bursts_spikes_normalized[data_bs_idx]))
            data_bs_distances = {
                p : min([
                        victor_purpura_distance(
                            [data_bs_ST, SpikeTrain(reference_p_pattern, units='ms', t_stop = max(reference_p_pattern))],
                            # los spikes ya estan ordenados
                            sort=False,
                            cost_factor=1.0/(10.0 * pq.ms)
                        )[0,1] for reference_p_pattern in reference_spikes[p]
                    ]) for p in PATTERNS
            }
            classification.append([data_bursts_spikes[data_bs_idx], min(data_bs_distances, key=data_bs_distances.get)])
            classification_for_plotting.append([data_bursts_spikes_normalized[data_bs_idx], min(data_bs_distances, key=data_bs_distances.get)])
            if not DISABLE_TIME_PRINTS:
                progress_calculating_distances += 1
                print(f'Llevamos un {Color.CYAN}{round(progress_calculating_distances * 100/L_data_bursts_spikes_normalized, 2)}{Color.END}% de calculo de la clasificacion por Victor Purpura', end='\r')
    elif distance == 'VR':
        # lo hacemos por indices para sacar los originales
        for data_bs_idx in range(L_data_bursts_spikes_normalized):
            # clase necesaria para el modulo, t_stop puesto para asegurar que funciona
            data_bs_ST = SpikeTrain(data_bursts_spikes_normalized[data_bs_idx], units='ms', t_stop = 10000)
            data_bs_distances = {
                p : min([
                    van_rossum_distance(
                        [data_bs_ST, SpikeTrain(reference_p_pattern, units='ms', t_stop = max(reference_p_pattern))],
                        # los spikes ya estan ordenados
                        sort=False,
                        time_constant=10.0*pq.ms
                    )[0,1] for reference_p_pattern in reference_spikes[p]
                ]) for p in PATTERNS
            }
            classification.append([data_bursts_spikes[data_bs_idx], min(data_bs_distances, key=data_bs_distances.get)])
            classification_for_plotting.append([data_bursts_spikes_normalized[data_bs_idx], min(data_bs_distances, key=data_bs_distances.get)])
            if not DISABLE_TIME_PRINTS:
                progress_calculating_distances += 1
                print(f'Llevamos un {Color.CYAN}{round(progress_calculating_distances * 100/L_data_bursts_spikes_normalized, 2)}{Color.END}% de calculo de la clasificacion por Van Rossum', end='\r')
    # guardamos la clasificacion
    if save:
        if type(filename) == None:
            error_print('Guardado de la clasificacion activada pero no se ha pasado el parametro filename')
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_classification_{distance}.dat"), 'w') as f:
            f.write(f'# clasificacion # spikes\n')
            for c in classification:
                f.write(f'{c[1]} {" ".join(str(s) for s in c[0])}\n')
    
    # dibujamos las graficas de los ipis de los patrones detectados
    __graph_detected_patterns(classification_for_plotting, distance=distance)

    return classification

