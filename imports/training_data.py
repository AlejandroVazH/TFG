import os
from random import uniform, seed, randint, getrandbits
from imports.support.utils import Color, PATTERNS, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s, error_print
from imports.support.carlson import carlson
from imports.support.density_functions import generate_SDF, generate_SDD
from imports.support.bursts_finding import carlson_method
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from numpy import mean, std, inf, linspace, sqrt
from sklearn.metrics import mean_squared_error
from bisect import bisect_left, bisect_right

################################################################################
# MACROS
################################################################################

# numero de patrones a generar a partir de uno base POR DEFECTO
NUM_PATTERNS = 100

# rango de valores en los que puede caer un scallop (ms)
SCALLOP_LOW_RANGE = [20, 500]

# duracion minima y maxima de los IPIs de la parte baja del scallop y del rasp (ms)
SCALLOP_LOW_IPI_RANGES = [9, 5]

# rango de numero de pulsos en la parte baja del scallop
SCALLOP_LOW_NUM_SPIKES = [1, 5]

# rango de recuperacion de los scallops
SCALLOP_RECOVER_TIMES = [400, 800]

# duracion de los IPIs de la parte baja del acceleration (ms)
ACC_LOW_IPI = 30

# rango de numero de pulsos para la parte de aceleracion
ACCELERATION_LOW_NUM_SPIKES = [10, 50]

# rangos para la subida y la bajada
ACCELERATION_RECOVER_TIME = [0, 150]

# rango de valores en los que puede caer un rasp (ms)
RASP_LOW_RANGE = [20, 300]

# rango de numero de pulsos para la parte de aceleracion del rasp
RASP_LOW_NUM_SPIKES = [8, 30]

# duracion de los IPIs de la BASELINE
BASELINE = 140

# seed para repetir experimentos
SEED = 15

# thresholds ideales respecto al maximo para cada patron
PATTERNS_THRESHOLD_MAX_FACTOR = {'scallop' : 0.6, 'acceleration' : 0.6, 'rasp' : 0.3}

################################################################################
# FUNCIONES PRIVADAS
################################################################################0

#
# Funcion para obtener la fase de descenso de los scallops y los rasps
#
def __get_scallop_down_and_middle() -> list[float]:
    T_b = uniform(SCALLOP_LOW_RANGE[0], SCALLOP_LOW_RANGE[1])
    # numero de ipis de 5ms que insertar
    num_5ms_ipis = randint(SCALLOP_LOW_NUM_SPIKES[0], SCALLOP_LOW_NUM_SPIKES[1])
    # generamos la lista de IPIs de la U
    a = 4/(SCALLOP_LOW_IPI_RANGES[0] - SCALLOP_LOW_IPI_RANGES[1])
    R = (SCALLOP_LOW_IPI_RANGES[0] - SCALLOP_LOW_IPI_RANGES[1])/2
    x_values = linspace(0, R, num=num_5ms_ipis//2 + 2).tolist()
    U_IPIs = [a*(x**2) + SCALLOP_LOW_IPI_RANGES[1] for x in x_values[::-1]]
    U_IPIs += [a*(x**2) + SCALLOP_LOW_IPI_RANGES[1] for x in x_values[1:]]
    # sacamos los valores de descenso hasta la U
    if T_b < 50:
        scallops_down = [(T_b - U_IPIs[0])]
    elif T_b < 250:
        scallops_down = [(T_b - U_IPIs[0])*x for x in [0.6, 0.3, 0.2]]
    elif T_b < 400:
        scallops_down = [(T_b - U_IPIs[0])*x for x in [0.4, 0.3, 0.2, 0.1]]
    else:
        scallops_down = [(T_b - U_IPIs[0])*x for x in [0.3, 0.25, 0.2, 0.15, 0.1]]
    
    return scallops_down + U_IPIs

#
# funcion para obtener la recuperacion de los scallops
#
def __get_scallop_recover() -> list[float]:
    recover_time = randint(SCALLOP_RECOVER_TIMES[0], SCALLOP_RECOVER_TIMES[1])
    recovery_function = lambda x : sqrt(x / recover_time) * BASELINE
    recovery_mean_time = (SCALLOP_RECOVER_TIMES[0] + SCALLOP_RECOVER_TIMES[1]) / 2
    if recover_time < recovery_mean_time:
        return [recovery_function(y) for y in [s*recover_time for s in [0.2, 0.4, 0.6, 0.8]]]
    else:
        return [recovery_function(y) for y in [s*recover_time for s in [0.2, 0.3, 0.4, 0.6, 0.8, 0.9]]]

#
# Funcion para obtener la subida y la bajada de la aceleración
#
def __get_acceleration_borders() -> list:
    # tiempo de suvida
    T_sub = uniform(ACCELERATION_RECOVER_TIME[0], ACCELERATION_RECOVER_TIME[1])
    if T_sub < 20:
        U_increase = []
    elif T_sub < 50:
        U_increase = [ACC_LOW_IPI + T_sub*x for x in [0.3, 0.7]]
    else:
        U_increase = [ACC_LOW_IPI + T_sub*x for x in [0.1, 0.3, 0.6]]
    # tiempo de bajada
    T_baj = uniform(ACCELERATION_RECOVER_TIME[0], ACCELERATION_RECOVER_TIME[1])
    # ecuacion de la curva de la subida y la bajada
    if T_baj < 20:
        U_decrease = []
    elif T_baj < 50:
        U_decrease = [ACC_LOW_IPI + T_baj*x for x in [0.7, 0.3]]
    else:
        U_decrease = [ACC_LOW_IPI + T_baj*x for x in [0.6, 0.3, 0.1]]
    
    return U_decrease, U_increase

#
# Funcion para la bajada de los rasps
#
def __get_rasp_down_and_middle() -> list:
    T_b = uniform(RASP_LOW_RANGE[0], RASP_LOW_RANGE[1])
    # numero de ipis de 5ms que insertar
    num_5ms_ipis = randint(SCALLOP_LOW_NUM_SPIKES[0], SCALLOP_LOW_NUM_SPIKES[1])
    # seleccionamos aleatoriamente si ponemos media U o una U
    full_U_or_not = bool(getrandbits(1))
    # generamos la lista de IPIs de la U
    a = 4/(SCALLOP_LOW_IPI_RANGES[0] - SCALLOP_LOW_IPI_RANGES[1])
    R = (SCALLOP_LOW_IPI_RANGES[0] - SCALLOP_LOW_IPI_RANGES[1])/2
    x_values = linspace(0, R, num=num_5ms_ipis//2 + 2).tolist()
    U_IPIs = [a*(x**2) + SCALLOP_LOW_IPI_RANGES[1] for x in x_values[::-1]]
    if full_U_or_not:
        U_IPIs += [a*(x**2) + SCALLOP_LOW_IPI_RANGES[1] for x in x_values[1:]]
    # sacamos los valores de descenso hasta la U
    if T_b < 50:
        rasps_down = [(T_b - U_IPIs[0])]
    elif T_b < 250:
        rasps_down = [(T_b - U_IPIs[0])*x for x in [0.6, 0.3, 0.2]]
    else:
        rasps_down = [(T_b - U_IPIs[0])*x for x in [0.4, 0.3, 0.2, 0.1]]
    
    return rasps_down + U_IPIs

#
# Funcion para la recuperacion de la fase de la aceleracion
#
def __get_rasp_increase() -> list:
    # tiempo de suvida
    T_sub = uniform(ACCELERATION_RECOVER_TIME[0], ACCELERATION_RECOVER_TIME[1])
    if T_sub < 20:
        U_increase = []
    elif T_sub < 50:
        U_increase = [ACC_LOW_IPI + T_sub*x for x in [0.3, 0.8]]
    elif T_sub < 100:
        U_increase = [ACC_LOW_IPI + T_sub*x for x in [0.1, 0.4, 1]]
    else:
        U_increase = [ACC_LOW_IPI + T_sub*x  for x in [0.1, 0.15, 0.3, 0.5, 0.7]]
    
    return U_increase


################################################################################
# FUNCIONES 
################################################################################0

#
# funcion para generar los patrones base
#
def generate_base_patterns(num_of_bases: int = 5, fixed_seed: bool = False) -> dict[str, list]:
    base_patterns = {p : [] for p in PATTERNS}

    # seed para reproducir los experimentos
    if fixed_seed:
        seed(SEED)

    for _ in range(num_of_bases):
        #
        # generacion de scallops
        #
        ipis = [BASELINE] * 3 + __get_scallop_down_and_middle() + __get_scallop_recover() + [BASELINE] * 3
        spikes, current_s = [], 0
        for ipi in ipis:
            current_s += ipi
            spikes.append(current_s)
        base_patterns['scallop'].append(spikes)

        #
        # generamos las aceleraciones
        #
        U_decrease, U_increase = __get_acceleration_borders()
        num_short_ipis = randint(ACCELERATION_LOW_NUM_SPIKES[0], ACCELERATION_LOW_NUM_SPIKES[1])
        ipis = [BASELINE] * 3 + U_decrease + [ACC_LOW_IPI]*num_short_ipis + U_increase + [BASELINE] * 3
        spikes, current_s = [], 0
        for ipi in ipis:
            current_s += ipi
            spikes.append(current_s)
        base_patterns['acceleration'].append(spikes)

        #
        # generamos los rasps
        #
        num_short_ipis = randint(RASP_LOW_NUM_SPIKES[0], RASP_LOW_NUM_SPIKES[1])
        ipis = [BASELINE] * 3 + __get_rasp_down_and_middle() + [ACC_LOW_IPI] * num_short_ipis + __get_rasp_increase() + [BASELINE] * 3
        spikes, current_s = [], 0
        for ipi in ipis:
            current_s += ipi
            spikes.append(current_s)
        base_patterns['rasp'].append(spikes)

    return base_patterns

#
# Funcion para generar las perturbaciones de los patrones base de un tipo concreto
#
# Solo requiere del nombre T de la carpeta que contiene los archivos modelos de patrones T/patrones.dat
#
def generate_training_data(
    og_patterns: dict[str, list],
    min_ipi_length: float = 5,
    max_half_range: float = 5,
    fixed_seed: bool = False,
    num_patterns: int = NUM_PATTERNS
) -> dict[str, list[list]]:
    new_patterns = {p : [] for p in PATTERNS}

    # input check
    if set([s for s in og_patterns.keys()]) != set(PATTERNS):
        error_print(f'Existen patrones no reconocidos entre los base: {set([s for s in og_patterns.keys()]).difference(set(PATTERNS))}')
    if min_ipi_length <= 0:
        error_print(f'parametro min_ipi_length {min_ipi_length} < 0')
    if max_half_range <= 0:
        error_print(f'parametro max_half_range {max_half_range} < 0')
    
    # perturbacion por rizado de la baseline
    if fixed_seed:
        seed(SEED)
    for p in PATTERNS:
        og_patterns_ipis = [[og_p[0]] + [og_p[i+1] - og_p[i] for i in range(len(og_p) - 1)] for og_p in og_patterns[p]]
        for base_p_patterns_idx in range(len(og_patterns[p])):
            spikes_range = []
            # hacemos todas las perturbaciones igual salvo la ultima ya que no tiene ipi derecho para comparar
            for spike_idx in range(len(og_patterns[p][base_p_patterns_idx]) - 1):
                half_ipi = min(max_half_range, min(
                    og_patterns_ipis[base_p_patterns_idx][spike_idx]/2,
                    og_patterns_ipis[base_p_patterns_idx][spike_idx + 1]/2
                ))
                spikes_range.append(
                    [og_patterns[p][base_p_patterns_idx][spike_idx] - half_ipi + min_ipi_length/2, og_patterns[p][base_p_patterns_idx][spike_idx] + half_ipi - min_ipi_length/2]
                    )
            # tocamos ahora el ultimo
            half_ipi = min(max_half_range, og_patterns_ipis[base_p_patterns_idx][-1]/2)
            spikes_range.append(
                [og_patterns[p][base_p_patterns_idx][-1] - half_ipi + min_ipi_length/2, og_patterns[p][base_p_patterns_idx][-1] + half_ipi - min_ipi_length/2]
                )
            # generamos secuencias aleatorias de spikes en esos rangos
            for i in range(0, num_patterns, 1):
                new_patterns[p].append([uniform(r[0], r[1]) for r in spikes_range] + [f'base={base_p_patterns_idx}'])

    # devolvemos los patrones generados
    return new_patterns

#
# Funcion para obtener el maximo mas pequeño de un conjunto de patrones generados (se usa para el threshold)
#
def get_smallest_maximum_from_perturbations(new_patterns: dict[str, list]) -> float:
    smallest_maximum = inf
    for p in PATTERNS:
        smallest_maximum = min(
            min([max(generate_SDF(spikes_train=generated_pattern_p[:-1])) for generated_pattern_p in new_patterns[p]]), smallest_maximum
        )
    return smallest_maximum

#
# funcion para sacar los spikes de la zona centro de un conjunto de spikes base
#
def get_center_of_base_patterns(base_patterns: dict[str, list]) -> dict[str, list]:
    centered_base_patterns = {p : [] for p in PATTERNS}
    for p in PATTERNS:
        for sp in base_patterns[p]:
            sp_SDF = generate_SDF(spikes_train=sp)

            burst_factor = max(sp_SDF)*PATTERNS_THRESHOLD_MAX_FACTOR[p]

            center = carlson_method(
                spikes=sp,
                SDF_data=sp_SDF,
                training_patterns=True,
                burst_factor=burst_factor
            )[0][0]

            centered_base_patterns[p].append(sp[bisect_left(sp, center[0]) : bisect_right(sp, center[1])])
    return centered_base_patterns


#
# Funcion para generar las estadisticas de Carlson de los patrones generados
#
# Solo requiere de la carpeta con los patrones generados. Por defecto se usa Carlson.
#
def get_carlson_stats_of_perturbations(new_patterns: dict[str, list], debug: bool = False) -> dict[str, list[dict[str, list]]]:
    if debug:
        # preparamos las graficas de los patrones para revisar el funcionamiento
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        pdf_debug_file = PdfPages(f'graficas/debug-generated-pattern-graphs.pdf')
    
    # esto sera lo que devolvamos
    new_patterns_stats = {p : [] for p in PATTERNS}

    # sacamos las estadisticas de cada patron
    for p in PATTERNS:
        for generated_pattern in new_patterns[p]:
            # sacamos las SDFs y las SDDs
            generated_pattern_SDF = generate_SDF(spikes_train=generated_pattern[:-1])
            generated_pattern_SDD = generate_SDD(spikes_train=generated_pattern[:-1])

            # parametro de los bursts para datos de entrenamiento
            burst_factor = max(generated_pattern_SDF)*PATTERNS_THRESHOLD_MAX_FACTOR[p]

            # sacamos los bursts (DEBE TENER 1)
            generated_pattern_center_and_burst = carlson_method(
                spikes=generated_pattern[:-1],
                SDF_data=generated_pattern_SDF,
                training_patterns=True,
                burst_factor=burst_factor
            )

            # tiene que haber UNO detectado
            if len(generated_pattern_center_and_burst) > 1:
                generated_pattern_center_and_burst = [max(generated_pattern_center_and_burst, key=lambda x: x[0][1] - x[0][0])]
            elif len(generated_pattern_center_and_burst) == 0:
                plt.plot(range(len(generated_pattern_SDF)), generated_pattern_SDF)
                plt.axhline(burst_factor)
                plt.savefig('graficas/log-generated-pattern-graphing.eps')
                error_print(f'El parametro generado {new_patterns[p].index(generated_pattern)} de los {p} ha detectado mas de un burst')
            
            # separamos el centro del burst
            generated_pattern_center = generated_pattern_center_and_burst[0][0]
            generated_pattern_burst = generated_pattern_center_and_burst[0][1]

            # sacamos su anchura y el número de spikes en la zona centro, y el porcentaje de burst frente a centro
            width = generated_pattern_center[1] - generated_pattern_center[0]
            center_spikes = bisect_left(generated_pattern[:-1], generated_pattern_center[1]) - bisect_right(generated_pattern[:-1], generated_pattern_center[0])
            burst_percenteage = (generated_pattern_burst[1] - generated_pattern_burst[0])/(generated_pattern_center[1] - generated_pattern_center[0])
            
            # extraccion de los datos de los bursts
            generated_pattern_stats = carlson(
                SDF_data=generated_pattern_SDF,
                SDD_data=generated_pattern_SDD,
                bursts_times_ranges=[generated_pattern_center]
            )
            generated_pattern_stats = generated_pattern_stats[0]

            # sacamos su altura lateral derecha e izquierda
            left_height = generated_pattern_stats['Pf'] - generated_pattern_SDF[generated_pattern_center[0]]
            right_height = generated_pattern_stats['Pf'] - generated_pattern_SDF[generated_pattern_center[1]]
            
            # los guardamos
            generated_pattern_stats.update({
                'base' : generated_pattern[-1],
                'limite_izquierdo_centro' : generated_pattern_center[0],
                'limite_derecho_centro' : generated_pattern_center[1],
                'limite_izquierdo_burst' : generated_pattern_burst[0],
                'limite_derecho_burst' : generated_pattern_burst[1],
                'anchura' : width,
                'spikes_centro' : center_spikes,
                'porcentaje_burst' : burst_percenteage,
                'altura_izquierda' : left_height,
                'altura_derecha' : right_height,
            })
            new_patterns_stats[p].append(generated_pattern_stats)
            center = generated_pattern_stats['centro']
            
            # si hay debugeo los graficamos
            if debug:
                fig = plt.figure()
                plt.plot(range(len(generated_pattern_SDF)), generated_pattern_SDF)
                for point_name, point_location, marker, color in zip(['St', 'Et'], [generated_pattern_stats['St'], generated_pattern_stats['Et']], ['o', 'o'], ['red', 'green']):
                    plt.plot(
                        point_location + center,
                        generated_pattern_SDF[point_location + center],
                        marker,
                        label=point_name,
                        color=color
                    )
                plt.axhline(burst_factor)
                plt.xlabel('Tiempo (s)')
                plt.ylabel('SDF (spikes / s)')
                plt.legend()
                fig.gca().yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
                fig.gca().xaxis.set_major_formatter(miliseconds_to_seconds)
                fig.savefig(pdf_debug_file, format='pdf') 
                plt.close(fig=fig)
            
            
    # cerramos el pdf en caso de debug
    if debug:
        pdf_debug_file.close()
        print(f'Archivo {Color.YELLOW}debug-generated-pattern-graphs{Color.END} generado en graficas/')
        
    return new_patterns_stats

#
# Funcion para sacar los filtros para la busqueda de patrones por patron
#
def get_filters_for_detection(new_patterns_stats: list[dict[str, list]], p: str, notify: bool = False) -> dict:
    # sacamos las distancias minimas para los minimos laterales el numero minimo de spikes dentro
    width_lower_threshold = [inf, None]
    width_higher_threshold = [-1, None]
    spikes_lower_threshold = [inf, None]
    spikes_higher_threshold = [-1, None]
    height_lower_threshold = [inf, None]
    height_higher_threshold = [-1, None]
    burst_percenteage_lower_threshold = [inf, None]
    
    # recorremos las estadisticas de los patrones que tenemos
    for new_pattern_p in new_patterns_stats[p]:
        # filtro de anchura
        if width_lower_threshold[0] > new_pattern_p['anchura']:
            width_lower_threshold = [new_pattern_p['anchura'], p + ' ' + new_pattern_p['base']]
        if width_higher_threshold[0] < new_pattern_p['anchura']:
            width_higher_threshold = [new_pattern_p['anchura'], p + ' ' + new_pattern_p['base']]
        
        # filtro de spikes
        if spikes_lower_threshold[0] > new_pattern_p['spikes_centro']:
            spikes_lower_threshold = [new_pattern_p['spikes_centro'], p + ' ' + new_pattern_p['base']]
        if spikes_higher_threshold[0] < new_pattern_p['spikes_centro']:
            spikes_higher_threshold = [new_pattern_p['spikes_centro'], p + ' ' + new_pattern_p['base']]
        
        # filtro de altura
        if height_lower_threshold[0] > min(new_pattern_p['altura_izquierda'], new_pattern_p['altura_derecha']):
            height_lower_threshold = [min(new_pattern_p['altura_izquierda'], new_pattern_p['altura_derecha']), p + ' ' + new_pattern_p['base']]
        if height_higher_threshold[0] < max(new_pattern_p['altura_izquierda'], new_pattern_p['altura_derecha']):
            height_higher_threshold = [max(new_pattern_p['altura_izquierda'], new_pattern_p['altura_derecha']), p + ' ' + new_pattern_p['base']]
        
        # filtro de bursts
        if burst_percenteage_lower_threshold[0] > new_pattern_p['porcentaje_burst']:
            burst_percenteage_lower_threshold = [new_pattern_p['porcentaje_burst'], p + ' ' + new_pattern_p['base']]
    
    # notificamos y devolvemos
    if notify:
        print(f'filtro de anchura minima entre minimos para {p}s: {round(width_lower_threshold[0], 5)}, patron: {width_lower_threshold[1]}')
        print(f'filtro de anchura maxima entre minimos para {p}s: {round(width_higher_threshold[0], 5)}, patron: {width_higher_threshold[1]}')
        print(f'filtro de spikes minimos entre minimos para {p}s: {round(spikes_lower_threshold[0], 2)}, patron: {spikes_lower_threshold[1]}')
        print(f'filtro de spikes maximos entre minimos para {p}s: {round(spikes_higher_threshold[0], 2)}, patron: {spikes_higher_threshold[1]}')
        print(f'filtro de altura minima entre minimos y maximo para {p}s: {round(height_lower_threshold[0], 5)}, patron: {height_lower_threshold[1]}')
        print(f'filtro de altura maxima entre minimos y maximo para {p}s: {round(height_higher_threshold[0], 5)}, patron: {height_higher_threshold[1]}')
        print(f'filtro de porcentaje de burst minimo para {p}s: {round(burst_percenteage_lower_threshold[0], 2)}, patron: {burst_percenteage_lower_threshold[1]}')

    return {
        'width_lower' : width_lower_threshold[0],
        'width_higher' : width_higher_threshold[0],
        'spikes_lower' : spikes_lower_threshold[0],
        'spikes_higher' : spikes_higher_threshold[0],
        'height_lower' : height_lower_threshold[0],
        'height_higher' : height_higher_threshold[0],
        'burst_percenteage_lower' : burst_percenteage_lower_threshold[0],
    }

#
# Funcion para graficar los patrones generados
#
# Solo requiere del directorio T con los patrones simulaciones T/patrones_spikes.dat generados
#
def generate_graphs(pattern_folder: str) -> None:
    # sacamos los patrones
    og_patterns = {'scallop' : [], 'acceleration' : [], 'rasp' : []}
    og_patterns_ipis = {'scallop' : [], 'acceleration' : [], 'rasp' : []}
    new_patterns = {'scallop' : [], 'acceleration' : [], 'rasp' : []}
    new_patterns_ipis = {'scallop' : [], 'acceleration' : [], 'rasp' : []}
    
    # sacamos los patrones y sus ipis
    for p in PATTERNS:
        with open(os.path.join(os.getcwd(), f'modelos de patrones {pattern_folder}/{p}.dat'), 'r') as pattern_file:
            line_idx=0
            for line in pattern_file:
                og_patterns[p].append([float(spike) for spike in line.strip().split(' ')])
                og_patterns_ipis[p].append([og_patterns[p][line_idx][0]] + [og_patterns[p][line_idx][i+1] - og_patterns[p][line_idx][i] for i in range(len(og_patterns[p][line_idx]) - 1)])
                line_idx+=1
        with open(os.path.join(os.getcwd(), f'simulaciones {pattern_folder}/{p}_spikes.dat'), 'r') as pattern_file:
            line_idx=0
            for line in pattern_file:
                l = line.strip().split(' ')
                new_patterns[p].append([float(s) for s in l[:-1]] + [l[-1]])
                new_patterns_ipis[p].append([new_patterns[p][line_idx][0]] + [new_patterns[p][line_idx][i+1] - new_patterns[p][line_idx][i] for i in range(len(new_patterns[p][line_idx]) - 2)])
                line_idx+=1

    fig, x_axis = plt.subplots()
    for pattern in PATTERNS:
        n_base_patterns = len(og_patterns_ipis[pattern])
        # tenemos que graficar las cosas respecto a los patrones base
        for i in range(n_base_patterns):
            # estadisticas
            new_patterns_mean_respect_to_base = mean([new_ipis for new_ipis in new_patterns_ipis[pattern][i*NUM_PATTERNS : (i+1)*NUM_PATTERNS - 1]], axis=0)
            new_patterns_std_respect_to_base = std([new_ipis for new_ipis in new_patterns_ipis[pattern][i*NUM_PATTERNS : (i+1)*NUM_PATTERNS - 1]], axis=0)
            new_patterns_ecm_respect_to_base = mean([mean_squared_error(og_patterns_ipis[pattern][i], new_ipis) for new_ipis in new_patterns_ipis[pattern][i*NUM_PATTERNS : (i+1)*NUM_PATTERNS - 1]], axis=0)

            # graficado de las estadisticas de estos ipis generados respecto a su base
            r = range(len(og_patterns_ipis[pattern][i]))
            x_axis.plot(r, og_patterns_ipis[pattern][i], color='k', label=f'{pattern} original')
            x_axis.plot(r, new_patterns_mean_respect_to_base, color='b', label=f'{pattern} medio generado')
            x_axis.plot(r, new_patterns_mean_respect_to_base + new_patterns_std_respect_to_base, color='r', label=f'{pattern} medio generado + desviacion tipica')
            x_axis.plot(r, new_patterns_mean_respect_to_base - new_patterns_std_respect_to_base, color='r',label=f'{pattern} medio generado - desviacion tipica')
            x_axis.plot(r, new_patterns_std_respect_to_base, color='green' ,label=f'desviacion tipica por punto')
            x_axis.plot([], [] ,label=f'Error cuadratico medio promedio: {new_patterns_ecm_respect_to_base}')
            # estetica
            xticks = [f'ipi {i+1}' for i in r]
            plt.xticks(r, xticks, fontsize=5)
            x_axis.set_title(f'{pattern}s')
            x_axis.set_xlabel('IPI')
            x_axis.set_ylabel('duracion del IPI (ms)')
            x_axis.legend(fontsize=5)
            # guardado en formato eps
            fig.savefig(os.path.join(os.getcwd(), f'simulaciones {pattern_folder}/{pattern}_simulated_base_{i}.eps'))
            print(f'Archivo {Color.YELLOW}{pattern}_simulated_base_{i}.eps{Color.END} generado en simulaciones {pattern_folder}/')
            x_axis.cla()

            # graficamos ahora los generados
            for generated_ipis_respect_to_base, generated_pattern_respect_to_base in zip(new_patterns_ipis[pattern][i*NUM_PATTERNS : (i+1)*NUM_PATTERNS - 1], new_patterns[pattern][i*NUM_PATTERNS : (i+1)*NUM_PATTERNS - 1]):
                x_axis.plot(generated_pattern_respect_to_base[:-1], generated_ipis_respect_to_base, alpha=0.5)
            # estetica
            x_axis.set_title(f'{pattern}s generados')
            x_axis.set_xlabel('Tiempo (s)')
            x_axis.set_ylabel('duracion del IPI (segundos / IPI)')
            fig.set_figwidth(10)
            fig.set_figheight(8)
            # ajuste de los valores del eje horizontal y vertical
            ms_to_s = FuncFormatter(miliseconds_to_seconds)
            plt.gca().yaxis.set_major_formatter(ms_to_s)
            plt.gca().xaxis.set_major_formatter(ms_to_s)
            # guardamos
            fig.savefig(os.path.join(os.getcwd(), f'simulaciones {pattern_folder}/{pattern}_simulated_examples_base_{i}.jpeg'), dpi=300)
            print(f'Archivo {Color.YELLOW}{pattern}_simulated_examples_base_{i}.jpeg{Color.END} generado en simulaciones {pattern_folder}/')
            x_axis.cla()

    return