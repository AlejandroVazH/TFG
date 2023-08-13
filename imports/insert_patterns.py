################################################################################
# Módulo insert_patterns
#
# Author: Alejandro Vázquez Huerta
# Descripción: Módulo con la funcionalidad para insertar patrones generados en
#              una secuencia de pulsos de fondo
################################################################################

import random
from imports.support.utils import PATTERNS

################################################################################
# MACROS
################################################################################

# Numero de spikes entre patron y patron
DISTANCE = 50

################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

#
# Funcion publica del modulo
#
def insert_patterns(patterns_to_insert: dict[str, list], background_sequence: list) -> list[list, list]:
    # convertimos los patrones a insertar en IPIs junto a la señal de fondo
    patterns_to_insert_ipis = {}
    for p in PATTERNS:
        patterns_to_insert_ipis[p] = []
        for p_to_insert in patterns_to_insert[p]:
            patterns_to_insert_ipis[p].append(
                [p_to_insert[0]] + [p_to_insert[i+1] - p_to_insert[i] for i in range(len(p_to_insert) - 2)] + [p_to_insert[-1]]
            )
    old_IPIs_list = [background_sequence[0]] + [background_sequence[i+1] - background_sequence[i] for i in range(len(background_sequence) - 1)]
    
    # algoritmo para añadirlos in situ
    patterns_added = []
    new_data_sequence = []
    pattern_counter = 1
    current_spike = 0
    # en vez de recorrer los antiguos spikes, recorremos los IPIs
    ipi_idx = 0
    L_old_IPIs_list = len(old_IPIs_list)
    while ipi_idx < L_old_IPIs_list:
        current_spike = old_IPIs_list[ipi_idx] + current_spike
        new_data_sequence.append(current_spike)

        # añadiremos de vez en cuando patrones a una distancia razonable
        if pattern_counter % DISTANCE == 0:
            # seleccion aleatoria del patron
            type = random.choice([x for x in patterns_to_insert.keys()])
            sequence_ipis = patterns_to_insert_ipis[type][random.randint(0, len(patterns_to_insert_ipis[type])-1)]

            # añadimos la secuencia sobre el ultimo spike añadido (es el primero de la secuencia)
            new_pattern_added = [type, sequence_ipis[-1]]
            for ipi_s in sequence_ipis[:-1]:
                current_spike = ipi_s + current_spike
                new_data_sequence.append(current_spike)
                new_pattern_added.append(current_spike)
            patterns_added.append(new_pattern_added)

            # buscamos concatenamos en una zona con un IPI menor de 110
            while ipi_idx + 1 < L_old_IPIs_list and old_IPIs_list[ipi_idx + 1] > 110:
                ipi_idx += 1
            
            # reiniciamos
            pattern_counter = 0
        
        # pasamos
        pattern_counter += 1
        ipi_idx += 1

    # extraemos los IPIs de los datos nuevos
    #new_IPIs_list = []
    #prev_spike = 0
    #for spike in new_data_sequence:
    #    new_IPIs_list.append(spike - prev_spike)
    #    prev_spike = spike
    # graficamos una sobre otra
    #fig, x_axis = plt.subplots()
    #x_axis.set_title("IPIs antiguos frente a nuevos")
    #plot1 = x_axis.hist(
    #    new_IPIs_list,
    #    bins = 'auto',
    #    density=True,
    #    label='datos con patrones',
    #    color = 'blue',
    #    fill=None,
    #    histtype = 'step'
    #)
    #plot2 = x_axis.hist(
    #    old_IPIs_list,
    #    bins = 'auto',
    #    density=True,
    #    label='datos sin patrones',
    #    color = 'red'#,
    #    #fill=None,
    #    #histtype = 'step'
    #)
    #x_axis.legend()
    #x_axis.set_title('Histograma comparativo entre ipis antes y después de los patrones')
    #x_axis.set_xlabel('IPIs (ms)')
    #x_axis.set_ylabel('frecuencia de los IPIs')
    #fig.savefig(os.path.join(os.getcwd(), f'graficas/{filename}-pattern-_ipis_contrast.eps'))
    #print(f'Archivo {Color.YELLOW}{filename}-pattern-{pattern_folder}_ipis_contrast.eps{Color.END} generado en graficas/')
    #plt.clf()

    return new_data_sequence, patterns_added
