################################################################################
# Módulo combine_spike_files
#
# Author: Alejandro Vázquez Huerta
# Descripción: Módulo con la metodología para concatenar secuencias temporales
#              de pulsos
################################################################################

import os
from typing import Union
from imports.support.utils import error_print, Color

################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

def combine_spikes_files(og_filenames: list[str], new_filename: Union[str, None] = None, save: bool = False) -> None:
    total_spikes = []
    
    # sacamos los ficheros que nos interesan
    last_spike = 0
    for f_name in [os.path.join(os.getcwd(), f'resultados/{f}_spikes.dat') for f in og_filenames]:
        # añadimos cada uno de los spikes del fichero sumandole el ultimo del fichero anterior 
        with open(f_name, 'r') as f_name_object:
            for line in f_name_object:
                current_spike = float(line.strip())
                total_spikes.append(current_spike + last_spike)
        # una vez acabado, actualizamos el last_spike
        last_spike += current_spike
    
    # lo guardamos
    if save:
        if type(new_filename) == None:
            error_print('guardado de los spikes combinados activado pero no se ha pasado el parametro new_filename')
        with open(os.path.join(os.getcwd(), f'resultados/{new_filename}_spikes.dat'), 'w') as new_file:
            for t in total_spikes:
                new_file.write(f'{t}\n')
        print(f'Archivo {Color.YELLOW}{new_filename}_spikes.dat{Color.END} generado en resultados/')
    
    return total_spikes
