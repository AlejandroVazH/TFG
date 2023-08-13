#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import miliseconds_to_seconds, Color, error_print
import matplotlib.pyplot as plt
from numpy import mean

NUM_SPIKES = 150

# sacamos el data name y el threshold
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if filename == None:
    error_print('Parametro --filename no detectado')

# sacamos la secuencia de patrones de fondo
print(f'Sacando la secuencia de fondo...')
background_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        background_spikes.append(float(line.strip()))

# sacamos lo ipis
background_ipis = [background_spikes[0]] + [background_spikes[i+1] - background_spikes[i] for i in range(len(background_spikes) - 1)]

# ipi promedio
ipi_mean = mean(background_ipis)

# graficamos
fig, axis = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(3)
axis.plot(background_spikes[1:NUM_SPIKES], background_ipis[1:NUM_SPIKES], '--o')
axis.axhline(ipi_mean, color = 'green')
axis.set_xlabel('tiempo (s)')
axis.set_ylabel('IPI (ms)')
axis.xaxis.set_major_formatter(miliseconds_to_seconds)
axis.set_title('Baseline')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/baseline_senal_real.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}baseline_senal_real.jpg{Color.END} generado en scripts/auxiliar_img/')
