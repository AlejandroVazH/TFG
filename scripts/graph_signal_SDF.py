#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import error_print, Color, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s
from imports.support.density_functions import generate_SDF
import matplotlib.pyplot as plt

# sacamos el nombre del fichero de pulsos a analizar
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# sacamos la secuencia de pulsos
print(f'Sacando la secuencia de pulsos...')
background_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        background_spikes.append(float(line.strip()))

# sacamos su SDF
sequence_SDF = generate_SDF(spikes_train=background_spikes)

# la graficamos
fig, x_axis = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(5)
x_axis.plot(sequence_SDF[:50000], linewidth=0.5)
x_axis.xaxis.set_major_formatter(miliseconds_to_seconds)
x_axis.yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
x_axis.set_ylabel('Valor SDF (spikes / s)')
x_axis.set_xlabel('Tiempo (s)')
fig.savefig(os.path.join(os.getcwd(), f'graficas/{filename}_seccion_SDF.jpg'), dpi = 300)
print(f'Archivo {Color.YELLOW}{filename}_seccion_SDF.jpg{Color.END} generado en graficas/')
