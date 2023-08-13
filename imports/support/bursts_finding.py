################################################################################
# Módulo support
#
# Author: Alejandro Vázquez Huerta
# Descripción: Fichero con macros y objetos de soporte para el resto del fichero
################################################################################

import os
from imports.support.utils import Color, error_print, DISABLE_FACTOR_PRINTS, MINIUM_RANGE, PATTERNS
from imports.support.density_functions import generate_SDF
from numpy import log, percentile, median, array, sqrt
from scipy.stats import median_abs_deviation, norm
import matplotlib.pyplot as plt
from time import time
from typing import Union
from bisect import bisect_right, bisect_left
from numpy import inf

################################################################################
# MACROS
################################################################################

# factor por defectosobre el total para determinar que es un burst y que no
CARLSON_BURST_FACTOR_DEFAULT = 1.7

# factor utilizado para el threshold de los burst en robust_gaussian_surprise
ROBUST_GAUSSIAN_SURPRISE_BURST_THRESHOLD_FACTOR = 1.96

# threshold para el ruido en la gaussiana
THRESHOLD_FOR_NOISE_BURSTS = 0

# diccionario por defecto de thresholds
DEFAULT_DICT_OF_THRESHOLDS = {p : {
    'width_lower' : 0,
    'width_higher' : inf,
    'spikes_lower' : 0,
    'spikes_higher' : inf,
    'height_lower' : 0,
    'height_higher' : inf,
    'burst_percenteage_lower' : 0,
} for p in PATTERNS}

################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Metodo de Carlson para detectar bursts, requiere de la SDF
#
def carlson_method(
    spikes: list[float],
    SDF_data: list[float],
    burst_factor: Union[float, None] = None,
    carlson_burst_factor: float = CARLSON_BURST_FACTOR_DEFAULT,
    dict_of_thresholds: dict[str, dict] = DEFAULT_DICT_OF_THRESHOLDS,
    training_patterns: bool = False
) -> list:
    bursts_times_ranges = []
    bursts_max_values = []

    # calculamos el factor que usaremos para determinar cuando estamos en un burst
    if type(burst_factor) == type(None):
        # forma 1: por media y MAD, no apto para aceleraciones de entrenamiento
        median_SDF = median(SDF_data)
        MAD_SDF = median_abs_deviation(SDF_data)
        burst_factor = median_SDF + MAD_SDF*carlson_burst_factor
    
    # recorremos el eje OX hasta el ultimo punto
    x = 0
    last_spike = len(SDF_data)-1
    while x < last_spike:
        # si estamos en un burst, guardamos los rangos y el maximo
        if SDF_data[x] > burst_factor:
            max_value = SDF_data[x]
            beginning_time = x
            x += 1
            while x < last_spike and SDF_data[x] > burst_factor:
                max_value = max(max_value, SDF_data[x])
                x +=1
            bursts_times_ranges.append([beginning_time, x])
            bursts_max_values.append(max_value)
        x += 1
        
    # buscamos minimo local hacia la izquierda o tiempo 0. caso extremo: x = 0 (extremo izquierdo del dominio)
    SPIs_times_ranges = []
    for left_range, right_range in bursts_times_ranges:
        x = left_range
        if x > 0:
            x -= 1
            while x > 0:
                for y in range(max([0, x - MINIUM_RANGE]), min([len(SDF_data)-1, x + MINIUM_RANGE])):
                    if SDF_data[x] > SDF_data[y]:
                        break
                else:
                    break
                x -= 1
        left_min = x
        # para la derecha es igual, PERO ahora el caso extremo es x = len(SDF_data)-1 (extremo derecho del dominio)
        x, last_time = right_range, len(SDF_data)-1
        if x < last_time:
            x += 1
            while x < last_time:
                for y in range(max([0, x - MINIUM_RANGE]), min([len(SDF_data)-1, x + MINIUM_RANGE])):
                    if SDF_data[x] > SDF_data[y]:
                        break
                else:
                    break
                x += 1
        SPIs_times_ranges.append([left_min, x])
    
    # purgamos el ruido
    SPIs_full_data = [
        [
            [left_limit, right_limit],
            [burst_left_limit, burst_right_limit]
        ] for (left_limit, right_limit), (burst_left_limit, burst_right_limit) in zip(SPIs_times_ranges, bursts_times_ranges)
    ]
    if not training_patterns:
        # vamos a filtrar para cada patron de manera independiente y luego juntamos sin interseccion
        SPIs_full_data_final = []

        for p in dict_of_thresholds.keys():
            SPIs_full_data_copy = SPIs_full_data.copy()
        
            ## filtro de porcentaje minimo por encima del threshold
            #SPIs_full_data_copy = [
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] for
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] in SPIs_full_data_copy if (
            #        (burst_right_limit - burst_left_limit)/(right_limit - left_limit) >= dict_of_thresholds[p]#['burst_percenteage_lower']
            #    )
            #]

            ## filtro de anchura minima entre minimos laterales
            #SPIs_full_data_copy = [
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] for
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] in SPIs_full_data_copy if (right_limit - left_limit >= dict_of_thresholds[p]['width_lower'])
            #]
            # filtro de anchura maxima entre minimos laterales
            #SPIs_full_data_copy = [
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] for
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] in SPIs_full_data_copy if (right_limit - left_limit <= dict_of_thresholds[p]['width_higher'])
            #]

            # filtro de numero de spikes minimos en la zona entre minimos
            SPIs_full_data_copy = [
                [
                    [left_limit, right_limit],
                    [burst_left_limit, burst_right_limit]
                ] for
                [
                    [left_limit, right_limit],
                    [burst_left_limit, burst_right_limit]
                ] in SPIs_full_data_copy if (
                    bisect_left(spikes, right_limit) - bisect_right(spikes, left_limit) >= dict_of_thresholds[p]['spikes_lower']
                )
            ]
            # filtro de numero de spikes maximos en la zona entre minimos
            #SPIs_full_data_copy = [
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] for
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] in SPIs_full_data_copy if (
            #        bisect_left(spikes, right_limit) - bisect_right(spikes, left_limit) <= dict_of_thresholds[p]#['spikes_higher']
            #    )
            #]

            # filtro de altura minima entre el maximo y los minimos laterales
            SPIs_full_data_copy = [
                [
                    [left_limit, right_limit],
                    [burst_left_limit, burst_right_limit]
                ] for
                [
                    [left_limit, right_limit],
                    [burst_left_limit, burst_right_limit]
                ] in SPIs_full_data_copy if (
                    min(max(SDF_data[burst_left_limit:burst_right_limit]) - SDF_data[left_limit], max(SDF_data[burst_left_limit:burst_right_limit]) - SDF_data[right_limit]) >= dict_of_thresholds[p]['height_lower']
                )
            ]
            ## filtro de altura maxima entre el maximo y los minimos laterales
            #SPIs_full_data_copy = [
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] for
            #    [
            #        [left_limit, right_limit],
            #        [burst_left_limit, burst_right_limit]
            #    ] in SPIs_full_data_copy if (
            #        max(max(SDF_data[burst_left_limit:burst_right_limit]) - SDF_data[left_limit], max(SDF_data#[burst_left_limit:burst_right_limit]) - SDF_data[right_limit]) <= dict_of_thresholds[p]#['height_higher']
            #    )
            #]

            # juntamos con lo obtenido por los otros filtrados
            SPIs_full_data_copy = set([((q[0][0], q[0][1]), (q[1][0], q[1][1])) for q in SPIs_full_data_copy])
            # vamos uniendo los resultados obtenidos para eliminar repeticiones. Lo dejamos como lista
            SPIs_full_data_final = [s for s in set(SPIs_full_data_final).union(SPIs_full_data_copy)]

    else:
        SPIs_full_data_final = SPIs_full_data

    return SPIs_full_data_final

#
# Funcion privada para optimizar la normalizacion de los LIPIS
# La N es la longitud del tren de spikes len(spikes_train)
#
def optimized_LIPIs_normalization(ipis_times: list, N: int) -> list:
    log_ipis_times = log(ipis_times)
    tiempo_inicial = time()
    Q = int(max(20, 0.2*N/2))
    nlipis = []

    # primeros Q ipis: e-center, mad, c1_center y central_location son constantes
    NQ_i = log_ipis_times[0:2*Q] # numero de ipis: 2Q+1
    e_center = (percentile(NQ_i, q=0.05) + percentile(NQ_i, q=0.95))
    mad = median_abs_deviation(NQ_i)
    c1_center = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - e_center) <= 1.64*mad])
    central_location = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - c1_center) <= 1.64*mad])
    for i in range(0, Q, 1):
        nlipis.append(log_ipis_times[i] - central_location)
        print(f'Llevamos un {Color.CYAN}{round((i+1) * 100/N, 2)}{Color.END}% de NLIPIs', end='\r')
    
    # para los del medio se actualiza todo constantemente
    for i in range(Q, N-Q, 1):
        NQ_i = log_ipis_times[i-Q:i+Q]
        e_center = (percentile(NQ_i, q=0.05) + percentile(NQ_i, q=0.95))
        mad = median_abs_deviation(NQ_i)
        c1_center = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - e_center) <= 1.64*mad])
        central_location = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - c1_center) <= 1.64*mad])
        nlipis.append(log_ipis_times[i] - central_location)
        print(f'Llevamos un {Color.CYAN}{round((i+1) * 100/N, 2)}{Color.END}% de NLIPIs', end='\r')
    
    # para los del final, i >= N-Q, vuelve a ser todo constante
    NQ_i = log_ipis_times[(N-1) - (2*Q):N-1] # numero de ipis: 2Q+1
    e_center = (percentile(NQ_i, q=0.05) + percentile(NQ_i, q=0.95))
    mad = median_abs_deviation(NQ_i)
    c1_center = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - e_center) <= 1.64*mad])
    central_location = median([log_ipi for log_ipi in NQ_i if abs(log_ipi - c1_center) <= 1.64*mad])
    for i in range(N-Q, N, 1):
        nlipis.append(log_ipis_times[i] - central_location)
        print(f'Llevamos un {Color.CYAN}{round((i+1) * 100/N, 2)}{Color.END}% de NLIPIs', end='\r')
    
    print(f'tiempo de procesado de la normalizacion de los spikes: {Color.CYAN}{time() - tiempo_inicial}{Color.END}')

    return nlipis

#
# Robust Gaussian Surprise Method para detectar bursts, requiere del tren de spikes. Creditos de:
# Detection of Bursts and Pauses in Spike Trains, D. Ko1, C. J. Wilson2, C. J. Lobb2, and C. A. Paladini2,*
#
def robust_gaussian_surprise_method(ipis_times: list, filename: str, N: int, already_normalized: bool = False, RGS_factor: float = ROBUST_GAUSSIAN_SURPRISE_BURST_THRESHOLD_FACTOR) -> list:
    # ejecutamos el algoritmo para normalizar si se requiere
    if not already_normalized:
        nlipis = optimized_LIPIs_normalization(ipis_times=ipis_times, N=N)
    else:
        nlipis = ipis_times

    # Calculamos los factores necesarios para el algoritmo
    total_mad = median_abs_deviation(nlipis)
    burst_threshold = (- RGS_factor) * total_mad
    if not DISABLE_FACTOR_PRINTS:
        print(f'Burst threshold: {Color.CYAN}{round(burst_threshold, 2)}{Color.END}')
    central_nlipis = [nlipi for nlipi in nlipis if abs(nlipi) < RGS_factor * total_mad]
    center_median = median(central_nlipis)
    center_mad = median_abs_deviation(central_nlipis)

    # efectuamos el algoritmo para detectar los bursts
    t = time()
    burst_seed_set_idx = [idx for idx in range(len(nlipis)) if nlipis[idx] < burst_threshold] 
    nlipis = array(nlipis)
    burst_full_set_idx_with_pvalue = []
    l_burst_seed_set_idx = len(burst_seed_set_idx)
    i=0
    L_nlipis = len(nlipis)
    for idx in burst_seed_set_idx:
        # hay que expandir hacia la izquierda y la derecha
        l_idx, r_idx, q, current_burst_idx = idx-1, idx+1, 1, [idx]
        # nos quedaremos con los valores que bajen esto de valor
        string_pvalue = norm(loc=q*center_median, scale=sqrt(q)*(center_mad)).cdf(sum(nlipis[current_burst_idx]))
        while r_idx < N:
            # en el calculo se aumenta el 1 el numero de valores y se concatena el indice
            new_pvalue = norm(loc=(q+1)*center_median, scale=sqrt(q+1)*(center_mad)).cdf(sum(nlipis[current_burst_idx + [r_idx]]))
            if new_pvalue >= string_pvalue:
                break
            else:
                current_burst_idx.append(r_idx)
                string_pvalue = new_pvalue
                r_idx += 1
                q += 1
        while l_idx >= 0:
            # al expandirse a la izquierda, se concatena el indice al inicio
            new_pvalue = norm(loc=(q+1)*center_median, scale=sqrt(q+1)*(center_mad)).cdf(sum(nlipis[[l_idx] + current_burst_idx]))
            if new_pvalue >= string_pvalue:
                break
            else:
                # a la izquierda hay que concatenar al inicio
                current_burst_idx.insert(0, l_idx)
                string_pvalue = new_pvalue
                l_idx -= 1
                q += 1
        # una vez completado lo añadimos
        burst_full_set_idx_with_pvalue.append([current_burst_idx, string_pvalue])
        print(f'Llevamos un {Color.CYAN}{round((i+1) * 100/l_burst_seed_set_idx, 2)}{Color.END}% de bursts localizados', end='\r')
        i += 1
    print(f'tiempo de procesado de la localizacion de bursts: {Color.CYAN}{time() - t}{Color.END}')
    
    # ejecutamos el algoritmo para depurar las coincidencias de bursts
    # VERIFICADO QUE FUNCIONA
    t = time()
    i = 0
    while i < len(burst_full_set_idx_with_pvalue):
        while i+1 < len(burst_full_set_idx_with_pvalue):
            if len(set(burst_full_set_idx_with_pvalue[i][0]).intersection(burst_full_set_idx_with_pvalue[i+1][0])) == 0:
                break
            else:
                if burst_full_set_idx_with_pvalue[i][1] < burst_full_set_idx_with_pvalue[i+1][1]:
                    burst_full_set_idx_with_pvalue.remove(burst_full_set_idx_with_pvalue[i+1])
                else:
                    burst_full_set_idx_with_pvalue.remove(burst_full_set_idx_with_pvalue[i])
        i += 1
    print(f'tiempo de procesado de la eliminacion de bursts repetidos: {Color.CYAN}{time() - t}{Color.END}')

    # pueden aparecer patrones de longitud 2 al ser mini bursts aislados por la actividad del pez, esos los eliminamos
    burst_full_set_idx_with_pvalue = [b for b in burst_full_set_idx_with_pvalue if len(b[0]) > THRESHOLD_FOR_NOISE_BURSTS]
    
    # transformamos los conjuntos de indices en spikes y lo devolvemos
    return burst_full_set_idx_with_pvalue
