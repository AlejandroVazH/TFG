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
print(f'Sacando la SDF...')
sequence_SDF = generate_SDF(spikes_train=background_spikes)

# la graficamos
fig, x_axis = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(8)
x_axis.boxplot(
    sequence_SDF,
    vert=False,
    widths=[0.8],
    flierprops={'markersize': 2}
)
fig.tight_layout()
x_axis.xaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
x_axis.set_xlabel('Valor SDF (spikes / s)')
x_axis.set(ylabel=None)
plt.subplots_adjust(top=0.7, bottom=0.3)
fig.savefig(os.path.join(os.getcwd(), f'graficas/{filename}_boxplot.jpg'), dpi = 300)
print(f'Archivo {Color.YELLOW}{filename}_boxplot.jpg{Color.END} generado en graficas/')
