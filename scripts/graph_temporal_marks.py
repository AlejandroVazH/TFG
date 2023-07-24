import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import PATTERNS, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s, Color
from imports.support.density_functions import generate_SDF, generate_SDD
from matplotlib.ticker import FuncFormatter
from imports.training_data import generate_base_patterns, generate_training_data, get_carlson_stats_of_perturbations
import matplotlib.pyplot as plt

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True, num_of_bases=5)

# generamos sus perturbaciones
print(f'Generando perturbaciones 10ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=10, num_patterns=1)

# sacamos sus stats
print(f'Generando stats 10ms...')
vibrated_patterns_5ms_stats = get_carlson_stats_of_perturbations(
    new_patterns=vibrated_patterns_5ms
)

# sacamos los ipis de un patron generado, sus estadisticas, y los graficamos todos juntos
fig, x_axis = plt.subplots(ncols=2)
fig.set_figwidth(10)
fig.set_figheight(3)
for p in ['acceleration']:
    # SDF y SDD
    spikes = vibrated_patterns_5ms[p][0]
    stats = vibrated_patterns_5ms_stats[p][0]
    center = stats['centro']
    spikes_SDF = generate_SDF(spikes_train=spikes[:-1])
    spikes_SDD = generate_SDD(spikes_train=spikes[:-1])
    # graficado estadisticas SDF
    r = range(len(spikes_SDF))
    x_axis[0].plot(r, spikes_SDF, color='k')
    x_axis[0].set_xlabel('Tiempo (s)')
    x_axis[0].set_ylabel('SDF (spikes / s)')
    x_axis[0].yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
    x_axis[0].xaxis.set_major_formatter(miliseconds_to_seconds)
    x_axis[0].set_title('a) St y Et')
    for point_name, point_location, marker, color in zip(['St', 'Et'], [int(stats['St']), int(stats['Et'])], ['o', 'o'], ['red', 'green']):
        x_axis[0].plot(point_location + center, spikes_SDF[point_location + center], marker, label=point_name, color=color)
    x_axis[0].legend()
    # graficado estadisticas SDD
    x_axis[1].plot(r, spikes_SDD, color='k')
    x_axis[1].set_xlabel('Tiempo (s)')
    x_axis[1].set_ylabel('SDD (spikes / s²)')
    x_axis[1].yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
    x_axis[1].xaxis.set_major_formatter(miliseconds_to_seconds)
    x_axis[1].set_title('a) R2t, R1t, F1t y F2t')
    for point_name, point_location, marker, color in zip(['R2t', 'R1t', 'F1t', 'F2t'], [int(stats['R2t']), int(stats['R1t']), int(stats['F1t']), int(stats['F2t'])], ['o', 'o', 'o', 'o'], ['red', 'green', 'blue', 'darkviolet']):
        x_axis[1].plot(point_location + center, spikes_SDD[point_location + center], marker, label=point_name, color=color)
    x_axis[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/Graficas_estadisticas_temporales.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}Graficas_estadisticas_temporales.jpg{Color.END} generado en scripts/auxiliar_img/')