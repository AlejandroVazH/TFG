################################################################################
# Módulo density_functions
#
# Author: Alejandro Vázquez Huerta
# Descripción: Este módulo implementa las transformaciones por función de
#              densidad de pulsos de secuencias temporales de pulsos.
################################################################################

import os
import numpy as np
from time import time
from bisect import bisect_left, bisect_right
from scipy.signal import savgol_filter
from imports.support.utils import Color, DISABLE_TIME_PRINTS, error_print
from typing import Union

################################################################################
# CONSTANTES
################################################################################

# macro con la desviacion tipica para los kernels de la SDF
STANDARD_DEVIATION = 50 # Anchura del kernel 200ms => sd = ancho/4 = 50ms

# ventana de suavizado
WINDOW_LENGTH = 201

################################################################################
# FUNCIONES DE GENERACION DE SDF Y SDD
################################################################################

#
# Funcion para generar la SDF
# El parametro filename se debe cambiar si se quiere guardar la SDF (parametro save)
#
def generate_SDF(get_spikes_from_file: bool = False, spikes_train: Union[list[float], None] = None, filename: str = None, save: bool = False) -> list[float]:
    if get_spikes_from_file:
        if filename == None:
            error_print('Cálculo de la SDF por fichero activada pero no se ha pasado el parametro filename')
        spikes_train = []
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_spikes.dat"), 'r') as f:
            for line in f:
                spikes_train.append(float(line.strip()))
    # factores para la convolucion con el kernel gaussiano
    kernel_standarization_factor = 1 / (np.sqrt(2 * np.pi) * STANDARD_DEVIATION)
    kernel_SD_factor = 2 * np.power(STANDARD_DEVIATION, 2)

    # generamos los valores de la SDF con una precision del 99%
    SDF_data, last_value, x, t = [], int(spikes_train[-1]) + 1, 0, time()
    while x <= last_value:
        left_index_for_poles = bisect_right(spikes_train, x - 3*STANDARD_DEVIATION)
        right_index_for_poles = bisect_left(spikes_train, x + 3*STANDARD_DEVIATION)
        SDF_data.append(kernel_standarization_factor * sum([np.e**(-(x - pole)**2 / kernel_SD_factor) for pole in spikes_train[left_index_for_poles:right_index_for_poles]]))
        x += 1
    
    if not DISABLE_TIME_PRINTS:
        print(f'tiempo de procesado de la SDF: {Color.CYAN}{time() - t}{Color.END}')

    # filtramos el ruido para calcular los minimos
    t = time()
    SDF_data = savgol_filter(x=SDF_data, window_length=WINDOW_LENGTH, polyorder=4)
    if not DISABLE_TIME_PRINTS:
        print(f'tiempo de suavizado: {Color.CYAN}{time() - t}{Color.END}')

    # guardamos y devolvemos la SDF
    if save:
        if filename == None:
            error_print('guardado de la SDF activado pero no se ha pasado el parametro filename')
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_SDF.dat"), 'w') as f:
            for sdf in SDF_data:
                f.write(f'{sdf}\n')
        print(f'Archivo {Color.YELLOW}{filename}_SDF.dat{Color.END} generado en resultados/')

    return SDF_data

#
# Funcion para generar la SDD
# El parametro filename se debe cambiar si se quiere guardar la SDD (parametro save)
#
def generate_SDD(get_spikes_from_file: bool = False, spikes_train: Union[list[float], None] = None, filename: str = None, save: bool = False) -> list[float]:
    if get_spikes_from_file:
        if filename == None:
            error_print('Cálculo de la SDD por fichero activada pero no se ha pasado el parametro filename')
        spikes_train = []
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_spikes.dat"), 'r') as f:
            for line in f:
                spikes_train.append(float(line.strip()))
    # factores para la convolucion con el kernel gaussiano
    kernel_standarization_factor = 1 / (np.sqrt(2 * np.pi) * STANDARD_DEVIATION)
    kernel_SD_factor = 2 * np.power(STANDARD_DEVIATION, 2)

    # generamos los valores de la SDD con una precision del 99%
    SDD_data, last_value, x, t = [], int(spikes_train[-1]) + 1, 0, time()
    while x <= last_value:
        left_index_for_poles = bisect_right(spikes_train, x - 3*STANDARD_DEVIATION)
        right_index_for_poles = bisect_left(spikes_train, x + 3*STANDARD_DEVIATION)
        SDD_data.append(kernel_standarization_factor * sum([(pole_location - x) * np.e**(-(x - pole_location)**2 / kernel_SD_factor) for pole_location in spikes_train[left_index_for_poles:right_index_for_poles]]))
        x += 1
    if not DISABLE_TIME_PRINTS:
        print(f'tiempo de procesado de la SDD: {Color.CYAN}{time() - t}{Color.END}')

    # filtramos el ruido para calcular los minimos
    t = time()
    SDD_data = savgol_filter(x=SDD_data, window_length=WINDOW_LENGTH, polyorder=4)
    if not DISABLE_TIME_PRINTS:
        print(f'tiempo de suavizado: {Color.CYAN}{time() - t}{Color.END}')

    # guardamos y devolvemos la sdd
    if save:
        if filename == None:
            error_print('guardado de la SDF activado pero no se ha pasado el parametro filename')
        with open(os.path.join(os.getcwd(), f"resultados/{filename}_SDD.dat"), 'w') as f:
            for sdd in SDD_data:
                f.write(f'{sdd}\n')
        print(f'Archivo {Color.YELLOW}{filename}_SDD.dat{Color.END} generado en resultados/')

    return SDD_data
