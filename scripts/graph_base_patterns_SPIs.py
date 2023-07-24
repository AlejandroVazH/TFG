import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import Color, PATTERNS, miliseconds_to_seconds
from imports.training_data import generate_base_patterns, generate_training_data
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from string import ascii_lowercase

# numero de subplots
NUM_SUBPLOTS = 4

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(fixed_seed=True, num_of_bases=50)

# rizamos los patrones a 5ms
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms = generate_training_data(og_patterns=base_patterns, max_half_range=5, num_patterns=1)
print(f'Generando perturbaciones 10ms...')
vibrated_patterns_10ms = generate_training_data(og_patterns=base_patterns, max_half_range=10, num_patterns=1)
print(f'Generando perturbaciones 15ms...')
vibrated_patterns_15ms = generate_training_data(og_patterns=base_patterns, max_half_range=15, num_patterns=1)

# graficamos ahora todos juntos en una sola grafica
for p in PATTERNS:
    fig, x_axis = plt.subplots(nrows=NUM_SUBPLOTS)
    for base_pattern, vp_5, vp_10, vp_15 in zip(base_patterns[p], vibrated_patterns_5ms[p], vibrated_patterns_10ms[p], vibrated_patterns_15ms[p]):
        # sacamos los ipis
        vp_5, vp_10, vp_15 = vp_5[:-1], vp_10[:-1], vp_15[:-1]
        ipis_base = [base_pattern[0]] + [base_pattern[i+1] - base_pattern[i] for i in range(len(base_pattern)-1)]
        ipis_v5 = [vp_5[0]] + [vp_5[i+1] - vp_5[i] for i in range(len(vp_5)-1)]
        ipis_v10 = [vp_10[0]] + [vp_10[i+1] - vp_10[i] for i in range(len(vp_10)-1)]
        ipis_v15 = [vp_15[0]] + [vp_15[i+1] - vp_15[i] for i in range(len(vp_15)-1)]
        # grafica de ipis
        x_axis[0].plot(base_pattern, ipis_base, alpha=0.3, color='blue')
        x_axis[0].set_title(f'a) IPIs {p}s base')
        x_axis[0].xaxis.set_major_formatter(miliseconds_to_seconds)
        x_axis[1].plot(vp_5, ipis_v5, alpha=0.3, color='blue')
        x_axis[1].set_title(f'b) IPIs {p}s rizados 5ms')
        x_axis[1].xaxis.set_major_formatter(miliseconds_to_seconds)
        x_axis[2].plot(vp_10, ipis_v10, alpha=0.3, color='blue')
        x_axis[2].set_title(f'c) IPIs {p}s rizados 10ms')
        x_axis[2].xaxis.set_major_formatter(miliseconds_to_seconds)
        x_axis[3].plot(vp_15, ipis_v15, alpha=0.3, color='blue')
        x_axis[3].set_title(f'd) IPIs {p}s rizados 15ms')
        x_axis[3].xaxis.set_major_formatter(miliseconds_to_seconds)
        # ajuste de dimensiones
        if p == 'scallop':
            for i in range(NUM_SUBPLOTS):
                x_axis[i].set_xlim(left=420, right=1600)
                x_axis[i].set_ylim(bottom=0, top=200)
        elif p == 'acceleration':
            for i in range(NUM_SUBPLOTS):
                x_axis[i].set_xlim(left=420, right=2500)
        else:
            for i in range(NUM_SUBPLOTS):
                x_axis[i].set_xlim(left=420, right=2000)
    # labels globales
    fig.supylabel('Duración del IPI (ms)')
    fig.supxlabel('Tiempo (s)')
    # aumentamos el tamaño antes de guardar y guardamos
    fig.set_figheight(10)
    fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/patrones_base_SPIs_{p}_juntos.jpg'), dpi=300)
    for i in range(NUM_SUBPLOTS):
        x_axis[i].cla()
    plt.close(fig=fig)
    print(f'Archivo {Color.YELLOW}patrones_base_SPIs_{p}_juntos.jpg{Color.END} generado en scripts/auxiliar_img/')

# graficamos uno de cada a 5ms
fig, axis = plt.subplots(nrows=len(PATTERNS))
fig.set_figwidth(10)
fig.set_figheight(5)
i = 0
for p in PATTERNS:
    # ajuste de dimensiones
    if p == 'scallop':
        axis[i].set_xlim(left=420, right=1600)
        axis[i].set_ylim(bottom=0, top=200)
    elif p == 'acceleration':
        axis[i].set_xlim(left=420, right=2500)
    else:
        axis[i].set_xlim(left=420, right=2000)
    # graficado de los patrones
    for vp_5 in vibrated_patterns_5ms[p]:
        vp_5 = vp_5[:-1]
        ipis_v5 = [vp_5[0]] + [vp_5[j+1] - vp_5[j] for j in range(len(vp_5)-1)]
        axis[i].plot(vp_5, ipis_v5, alpha=0.3, color='blue')
    # estetica
    axis[i].set_title(f'{ascii_lowercase[i]}) IPIs {p}s rizados 5ms')
    axis[i].xaxis.set_major_formatter(miliseconds_to_seconds)
    i += 1
# guardamos
fig.supylabel('Duración del IPI (ms)')
fig.supxlabel('Tiempo (s)')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/patrones_base_para_tfg.jpg'), dpi=300)
print(f'Archivo {Color.YELLOW}patrones_base_para_tfg{Color.END} generado en scripts/auxiliar_img/')