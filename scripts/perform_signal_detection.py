#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import error_print, Color, miliseconds_to_seconds, spikes_per_ms_to_spikes_per_s, PATTERNS
from imports.support.density_functions import generate_SDF, generate_SDD
from imports.support.bursts_finding import carlson_method
from imports.support.carlson import carlson
from imports.training_data import get_filters_for_detection, get_carlson_stats_of_perturbations, generate_training_data, generate_base_patterns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from joblib import load
from bisect import bisect_left, bisect_right
from typing import Union

#################################
# MACROS
#################################

# rango a los laterales al graficar
RANGE_TO_MOVE_LATERALLY = 100

# Altura de la grafica
FIG_HEIGHT = 5

# anchura de la grafica
FIG_WIDTH = 5

#################################
# FUNCIONES PRIVADAS
#################################

def __graph_in_pdf(
    pdf_file: PdfPages,
    list_of_ranges: list,
    background_spikes: list,
    sequence_SDF: list,
    sup_title: Union[str, None] = None
) -> None:
    for left_fp, right_fp in list_of_ranges:
        fig, ax = plt.subplots(nrows=2)
        # macro
        fig.set_figheight(FIG_HEIGHT)
        fig.set_figwidth(FIG_WIDTH)
        if sup_title is not None:
            fig.suptitle(f'{sup_title}')
        r = range(right_fp - left_fp + RANGE_TO_MOVE_LATERALLY*2)
        # hacemos que compartan el eje OX
        ax[0].sharex(ax[1])
        fig.supxlabel('Tiempo (s)')
        # spikes e ipis
        spikes = background_spikes[bisect_left(background_spikes, left_fp-RANGE_TO_MOVE_LATERALLY) - 1 : bisect_right(background_spikes, right_fp + RANGE_TO_MOVE_LATERALLY)]
        ipis = [spikes[i+1] - spikes[i] for i in range(len(spikes) - 1)]
        spikes = [s - (left_fp-RANGE_TO_MOVE_LATERALLY) for s in spikes[1:]]
        # grafica de los IPIs
        ax[0].plot(spikes, ipis, '--o')
        ax[0].xaxis.set_major_formatter(miliseconds_to_seconds)
        ax[0].set_ylabel('IPI (ms)')
        for s in spikes:
            ax[0].axvline(x=s, ymin=1, ymax=1.1, clip_on = False)
        ax[0].set_title('a) Grafica de IPIs del patrón detectado', x=0.5, y=1.2)
        # grafica de la parte de la SDF
        ax[1].plot(
            r[0:RANGE_TO_MOVE_LATERALLY],
            sequence_SDF[left_fp-RANGE_TO_MOVE_LATERALLY:left_fp],
            color='b'
        )
        ax[1].plot(
            r[RANGE_TO_MOVE_LATERALLY:-RANGE_TO_MOVE_LATERALLY],
            sequence_SDF[left_fp:right_fp],
            color='r'
        )
        ax[1].plot(
            r[-RANGE_TO_MOVE_LATERALLY:],
            sequence_SDF[right_fp:right_fp+RANGE_TO_MOVE_LATERALLY],
            color='b'
        )
        ax[1].yaxis.set_major_formatter(spikes_per_ms_to_spikes_per_s)
        ax[1].xaxis.set_major_formatter(miliseconds_to_seconds)
        ax[1].set_ylabel('SDF (spikes / s)')
        ax[1].set_title('b) SDF del patrón detectado')
        # lo guardamos
        fig.savefig(pdf_file, format='pdf')
        ax[0].cla()
        ax[1].cla()
        plt.close(fig=fig)

#################################
# PARTE DE SCRIPT
#################################

# sacamos el nombre del fichero de pulsos a analizar
filename = None
threshold = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
    if sys.argv[param_pos] == '--threshold':
        threshold = float(sys.argv[param_pos + 1])
if filename == None:
    error_print('Parametro --filename no detectado')
if threshold == None:
    error_print('Parametro --threshold no detectado')

# sacamos la secuencia de pulsos
print(f'Sacando la secuencia de pulsos...')
background_spikes = []
with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'), 'r') as f:
    for line in f:
        background_spikes.append(float(line.strip()))

# sacamos su SDF
print(f'Sacando la SDF...')
sequence_SDF = generate_SDF(spikes_train=background_spikes)

# thresholld de pulsos
print(f'Sacando el threshold de pulsos')
filters = {p : get_filters_for_detection(
    new_patterns_stats=get_carlson_stats_of_perturbations(
        new_patterns=generate_training_data(
            og_patterns=generate_base_patterns(num_of_bases=100),
            num_patterns=1
        )
    ),
    p=p
) for p in PATTERNS}

# sacamos los patrones
print(f'Detectando patrones...')
patterns_detected = carlson_method(spikes=background_spikes, SDF_data=sequence_SDF, burst_factor=threshold, dict_of_thresholds=filters)
patterns_detected = [c[0] for c in patterns_detected]

# graficando los detectados
print(f'Los graficamos...')
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
pdf_file = PdfPages(f'graficas/{filename}_patrones_detectados_zona_centro{threshold}ms.pdf')
__graph_in_pdf(pdf_file, patterns_detected, background_spikes, sequence_SDF)
pdf_file.close()
print(f'Archivo {Color.YELLOW}{filename}_patrones_detectados_{threshold}ms.pdf{Color.END} generado en graficas/')

# sacamos la SDD
print(f'Sacando la SDD...')
sequence_SDD = generate_SDD(spikes_train=background_spikes)

# sacamos sus estadisticas
print(f'Generando estadisticas...')
detected_patterns_carlson_stats = carlson(
    SDF_data=sequence_SDF,
    SDD_data=sequence_SDD,
    bursts_times_ranges=patterns_detected
)

# cargamos los clasificadores LDA y QDA
print(f'Cargando estimadores LDA y QDA...')
lda = load(os.path.join(os.getcwd(), 'resultados/classificator_LDA.jl'))
qda = load(os.path.join(os.getcwd(), 'resultados/classificator_QDA.jl'))

# cargamos los datos en formato stats y labels y predecimos
print(f'Prediciendo por LDA y QDA...')
detected_patterns_carlson_stats_list_format = []
for d_of_data in detected_patterns_carlson_stats:
    detected_patterns_carlson_stats_list_format.append([d_of_data[k] for k in ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3']])
lda_y_pred = lda.predict(detected_patterns_carlson_stats_list_format)
qda_y_pred = qda.predict(detected_patterns_carlson_stats_list_format)

# los formateamos para el guardado
print(f'Formateado para impresion...')
detected_patterns_carlson_stats_dict_LDA_class_format = {
    p : [(left_fp, right_fp) for (left_fp, right_fp), lda_class in zip(patterns_detected, lda_y_pred) if lda_class == p] for p in PATTERNS
}
detected_patterns_carlson_stats_dict_QDA_class_format = {
    p : [(left_fp, right_fp) for (left_fp, right_fp), qda_class in zip(patterns_detected, qda_y_pred) if qda_class == p] for p in PATTERNS
}

# guardamos los resultados LDA
print(f'Graficamos las predicciones LDA...')
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
pdf_file_LDA = PdfPages(f'graficas/{filename}_patrones_detectados_{threshold}ms_LDA.pdf')
for p in PATTERNS:
    __graph_in_pdf(
        pdf_file_LDA,
        detected_patterns_carlson_stats_dict_LDA_class_format[p],
        background_spikes,
        sequence_SDF,
        f'{p}'
    )
pdf_file_LDA.close()
print(f'Archivo {Color.YELLOW}{filename}_patrones_detectados_{threshold}ms_LDA.pdf{Color.END} generado en graficas/')

# guardamos los resultados QDA
print(f'Graficamos las predicciones QDA...')
pdf_file_QDA = PdfPages(f'graficas/{filename}_patrones_detectados_{threshold}ms_QDA.pdf')
for p in PATTERNS:
    __graph_in_pdf(
        pdf_file_QDA,
        detected_patterns_carlson_stats_dict_QDA_class_format[p],
        background_spikes,
        sequence_SDF,
        f'{p}'
    )
pdf_file_QDA.close()
print(f'Archivo {Color.YELLOW}{filename}_patrones_detectados_{threshold}ms_QDA.pdf{Color.END} generado en graficas/')

# cargamos y predecimos ahora por fitness
print(f'Cargando estimador fitness...')
fm = load(os.path.join(os.getcwd(), 'resultados/classificator_FM.jl'))

# formateado para el fm
print(f'Formateando datos para el modo por fitness...')
detected_patterns_SPIs = [background_spikes[bisect_left(background_spikes, left_limit) : bisect_right(background_spikes, right_limit)] for left_limit, right_limit in patterns_detected]

# ahora si predecimos
print(f'Prediciendo por fitness...')
fm_y_pred = fm.predict(detected_patterns_SPIs)

# guardamos los resultados
print(f'Formateado para impresion...')
detected_patterns_carlson_stats_dict_FM_class_format = {
    p : [(left_fp, right_fp) for (left_fp, right_fp), fm_class in zip(patterns_detected, fm_y_pred) if fm_class == p] for p in PATTERNS
}

# guardamos los resultados FM
print(f'Graficamos las predicciones FM...')
pdf_file_FM = PdfPages(f'graficas/{filename}_patrones_detectados_{threshold}ms_FM.pdf')
for p in PATTERNS:
    __graph_in_pdf(
        pdf_file_FM,
        detected_patterns_carlson_stats_dict_FM_class_format[p],
        background_spikes,
        sequence_SDF,
        f'{p}'
    )
pdf_file_FM.close()
print(f'Archivo {Color.YELLOW}{filename}_patrones_detectados_{threshold}ms_FM.pdf{Color.END} generado en graficas/')
