import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import Color, error_print, PATTERNS, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s
from imports.training_data import generate_base_patterns, generate_training_data
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True, num_of_bases=5)

# generamos sus perturbaciones
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5)
print(f'Generando perturbaciones 10ms...')
vibrated_patterns_10ms = generate_training_data(og_patterns=base_patterns, max_half_range=10)

# sacamos los ipis de un patron generado, y los graficamos todos juntos
for p in PATTERNS:
    for base_pattern_idx in range(len(base_patterns[p])):
        # grafica
        fig, x_axis = plt.subplots(ncols=3)
        fig.set_figwidth(12)
        fig.set_figheight(5)
        # SPIs
        bp = base_patterns[p][base_pattern_idx]
        ipis_bp = [bp[0]] + [bp[i+1] - bp[i] for i in range(len(bp)-1)]
        v5 = vibrated_patterns_5ms[p][100*base_pattern_idx][:-1]
        ipis_v5 = [v5[0]] + [v5[i+1] - v5[i] for i in range(len(v5)-1)]
        v10 = vibrated_patterns_10ms[p][100*base_pattern_idx][:-1]
        ipis_v10 = [v10[0]] + [v10[i+1] - v10[i] for i in range(len(v10)-1)]
        # ejes
        ms_to_s = FuncFormatter(miliseconds_to_seconds)
        # hacemos que compartan eje Y
        x_axis[0].sharey(x_axis[1])
        x_axis[1].sharey(x_axis[2])
        # graficas
        x_axis[0].plot(bp, ipis_bp, '--o')
        x_axis[0].set_ylabel('Duración del IPI (s)')
        x_axis[0].set_title(f'Patron {p} base')
        x_axis[0].yaxis.set_major_formatter(ms_to_s)
        x_axis[0].xaxis.set_major_formatter(ms_to_s)
        x_axis[1].plot(v5, ipis_v5, '--o')
        x_axis[1].set_xlabel('Tiempo (s)')
        x_axis[1].set_title(f'Patron {p} vibrado, 5ms')
        x_axis[1].yaxis.set_major_formatter(ms_to_s)
        x_axis[1].xaxis.set_major_formatter(ms_to_s)
        x_axis[2].plot(v10, ipis_v10, '--o')
        x_axis[2].set_title(f'Patron {p} vibrado, 10ms')
        x_axis[2].yaxis.set_major_formatter(ms_to_s)
        x_axis[2].xaxis.set_major_formatter(ms_to_s)
        # guardamos y cerramos
        fig.savefig(f'scripts/auxiliar_img/patron_{p}_base_perturbacion_{base_pattern_idx+1}.jpg', dpi=300)
        x_axis[0].cla()
        x_axis[1].cla()
        x_axis[2].cla()
        plt.close(fig=fig)
        print(f'Archivo {Color.YELLOW}patron_{p}_base_perturbacion_{base_pattern_idx+1}.jpg{Color.END} generado en scripts/auxiliar_img/')