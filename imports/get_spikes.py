################################################################################
# Módulo get_spikes
#
# Author: Alejandro Vázquez Huerta
# Descripción: Este módulo hace de puente entre este repositorio y el descrito
#              en el readme que implementa la metodología de extracción de
#              pulsos
################################################################################

import os
import subprocess
from imports.support.utils import Color

################################################################################
# MACROS
################################################################################

TOTAL_VOLTAGE_LOCATION = 4

################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Funcion para extraer los tiempos de suceso de los spikes
#
# Parametros:
#   path: path al fichero de datos
#
# Output:
#   filename: nombre del fichero de datos
#
def get_spike_times(path: str) -> None:
    filename = path.split('/')[-1].split('.')[0]

    subprocess.run(
        [
            f"{os.path.join(os.getcwd(),'bio-utils-master/scripts/spk-times/')}spk-times",
            "-i",
            path,
            "-o",
            os.path.join(
                os.getcwd(),
                f'resultados/{filename}_spikes.dat'
            ),
            "-c",
            str(TOTAL_VOLTAGE_LOCATION)
        ]
    )
    print(f'Archivo {Color.YELLOW}{filename}_spikes.dat{Color.END} generado en resultados/')
    return 
