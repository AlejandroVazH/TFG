#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import miliseconds_to_seconds, Color, PATTERNS
from imports.training_data import generate_base_patterns, generate_training_data
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True)

# generamos las perturbaciones
training_data = generate_training_data(og_patterns=base_patterns, fixed_seed=True)

# las graficamos
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
pdf_debug_file = PdfPages(f'scripts/auxiliar_img/patrones_base_perturbados_SPIs.pdf')
for p in PATTERNS:
    print(f'Generando {p}s...')
    for base_pattern in training_data[p]:
        fig, x_axis = plt.subplots()
        ipis = [base_pattern[0]] + [base_pattern[i+1] - base_pattern[i] for i in range(len(base_pattern)-2)]
        # grafica de ipis
        x_axis.plot(base_pattern[:-1], ipis, '--o')
        x_axis.set_xlabel('Tiempo (s)')
        x_axis.set_ylabel('Duración del IPI (s)')
        ms_to_s = FuncFormatter(miliseconds_to_seconds)
        fig.gca().yaxis.set_major_formatter(ms_to_s)
        fig.gca().xaxis.set_major_formatter(ms_to_s)
        fig.savefig(pdf_debug_file, format='pdf')
        x_axis.cla()
        plt.close(fig=fig)
print(f'Archivo {Color.YELLOW}patrones_base_perturbados_SPIs.pdf{Color.END} generado en scripts/auxiliar_img/')
pdf_debug_file.close()
