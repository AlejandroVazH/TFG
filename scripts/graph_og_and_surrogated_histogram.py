#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import miliseconds_to_seconds, Color, error_print
import matplotlib.pyplot as plt

# sacamos el data name y el threshold
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# sacamos la secuencia de patrones de fondo
print(f'Sacando la secuencia de fondo...')
og_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        og_spikes.append(float(line.strip()))
og_ipis = [og_spikes[0]] + [og_spikes[i+1] - og_spikes[i] for i in range(len(og_spikes) - 1)]

# sacamos la secuencia de patrones de fondo con surrogate
print(f'Sacando la secuencia de fondo con surrogate...')
surrogate_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}-surrogated_spikes.dat'), 'r') as f:
    for line in f:
        surrogate_spikes.append(float(line.strip()))
surrogate_ipis = [surrogate_spikes[0]] + [surrogate_spikes[i+1] - surrogate_spikes[i] for i in range(len(surrogate_spikes) - 1)]

# graficamos
print(f'Graficado de las senales...')
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(4)
# esquema de subplots
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(1, 2, 2)
# ejes de cada figura de IPIs con sus graficas
ax1.xaxis.set_major_formatter(miliseconds_to_seconds)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('IPI (ms)')
ax1.set_title(f'a) Señal original')
ax1.plot(og_spikes[:300], og_ipis[:300], linewidth=1)
ax2.xaxis.set_major_formatter(miliseconds_to_seconds)
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('IPI (ms)')
ax2.set_title(f'b) Señal con surrogate')
ax2.plot(surrogate_spikes[:300], surrogate_ipis[:300], linewidth=1)
# figura del histograma
ax3.hist(og_ipis, bins='auto', density=False, color='red', label='original')
ax3.hist(surrogate_ipis, bins='auto', density=False, label='surrogated', fill=None, histtype = 'step', color='blue')
ax3.set_xlabel('IPIs por duracion (s)')
ax3.set_ylabel('Numero de IPIs en el rango')
ax3.set_title(f'c) Histogramas comparativos')
ax3.xaxis.set_major_formatter(miliseconds_to_seconds)
ax3.set_xlim((0, 300))
ax3.legend()
# guardamos
fig.tight_layout()
fig.savefig(f'graficas/visualizacion-surrogate.jpg', dpi=300)
print(f'Archivo {Color.YELLOW}visualizacion-surrogate.jpg{Color.END} generado en graficas/')

plt.subplot
