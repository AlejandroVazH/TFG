import os
import numpy
import matplotlib.pyplot as plt
from imports.support.utils import Color, miliseconds_to_seconds
from random import choices, uniform
from random import shuffle

################################################################################
# MACROS
################################################################################

# numero de veces a concatenar los datos consigo mismos.
NUM_VUELTAS = 6

################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Surrogate de IPIs por CDF
#
def __surrogate(ipis_list: list[float], get_histogram: bool = False, filename: str = None) -> list[float]:
    surrogate_ipis = []

    # calculamos la CDF de los ipis
    og_histogram_density, og_bin_edges_density = numpy.histogram(ipis_list, bins='auto', density=True)
    # usamos esta funcion para seleccionar los bins en los que poner nuevos IPIs aleatoriamente
    generation_reference_bins_idx = choices(population=range(len(og_bin_edges_density[:-1])), weights=og_histogram_density, k=len(ipis_list))
    surrogate_ipis = [uniform(og_bin_edges_density[idx], og_bin_edges_density[idx+1]) for idx in generation_reference_bins_idx]
    
    # graficamos el histograma de contraste si se pide
    if get_histogram:
        fig, x_axis = plt.subplots()
        x_axis.hist(ipis_list, bins='auto', density=False, color='red', label='original')
        x_axis.hist(surrogate_ipis, bins='auto', density=False, label='surrogated', fill=None, histtype = 'step', color='blue')
        x_axis.set_xlabel('IPIs por duracion (s)')
        x_axis.set_ylabel('Numero de IPIs en el rango')
        fig.gca().xaxis.set_major_formatter(miliseconds_to_seconds)
        x_axis.set_xlim((0, 300))
        x_axis.legend()
        fig.savefig(os.path.join(os.getcwd(), f'graficas/{filename}-surrogated_histogram.eps'))
        x_axis.cla()
        print(f'Archivo {Color.YELLOW}{filename}-surrogated_histogram.eps{Color.END} generado en graficas/')

    return surrogate_ipis


################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

#
# Funcion publica del modulo
#
def data_analysis(filename: str, fake_mode: str, get_histogram: bool = True):
    fig, x_axis = plt.subplots()
    
    # obtenemos la lista de IPIs
    with open(os.path.join(os.getcwd(), f'resultados/{filename}_spikes.dat'),'r') as spk_times_file:
        # guardamos cuando hemos detectado el spike anterior
        previous_spike = 0
        ipis_list = []
        spikes_list = []

        for spk_line in spk_times_file:
            # leemos el spike actual y escribimos el ipi asociado
            line_spike_time = float(spk_line.strip())
            ipi = line_spike_time - float(previous_spike)
            # actualizamos el spike previo y guardamos el ipi
            previous_spike = line_spike_time
            ipis_list.append(ipi)
            spikes_list.append(line_spike_time)

    # generamos un histograma de conteo y lo graficamos si se pide
    if get_histogram:
        # grafica del conteo de ipis
        og_histogram, og_bin_edges = numpy.histogram(ipis_list, bins='auto', density=False)
        x_axis.bar(og_bin_edges[:-1], height=og_histogram, width=numpy.diff(og_bin_edges))
        x_axis.set_xlabel('IPIs por duracion (s)')
        x_axis.set_ylabel('numero de IPIs detectados')
        fig.gca().xaxis.set_major_formatter(miliseconds_to_seconds)
        x_axis.set_xlim((0, 300))
        fig.savefig(os.path.join(os.getcwd(), f'graficas/{filename}_ipis_histogram.eps'))
        x_axis.cla()
        print(f'Archivo {Color.YELLOW}{filename}_ipis_histogram.eps{Color.END} generado en graficas/')
    
    # efectuamos la replicacion por el metodo que se pida
    if fake_mode == 'concatenate':
        # concatenamos los datos consigo mismos varias veces para generar una secuencia muy larga y lo guardamos
        concatenated_data = []
        current_spike = 0
        for ipi in ipis_list * NUM_VUELTAS:
            current_spike += ipi
            concatenated_data.append(current_spike)

        # guardado de la concatenacion
        with open (os.path.join(os.getcwd(), f'resultados/{filename}-concatenated_spikes.dat'), 'w') as fake_spike_times:
            for d in concatenated_data:
                fake_spike_times.write(str(d) + '\n')
        print(f'Archivo {Color.YELLOW}{filename}-concatenated_spikes.dat{Color.END} generado en resultados/')

    elif fake_mode == 'surrogate':
        surrogated_ipis = __surrogate(ipis_list=ipis_list, get_histogram=get_histogram, filename=filename)
        surrogated_data = []
        current_spike = 0
        for ipi in surrogated_ipis:
            current_spike += ipi
            surrogated_data.append(current_spike)

        # guardado de la concatenacion
        with open (os.path.join(os.getcwd(), f'resultados/{filename}-surrogated_spikes.dat'), 'w') as fake_spike_times:
            for d in surrogated_data:
                fake_spike_times.write(str(d) + '\n')
        print(f'Archivo {Color.YELLOW}{filename}-surrogated_spikes.dat{Color.END} generado en resultados/')
    
    elif fake_mode == 'shuffle':
        # le metemos un shuffle a los datos y concatenamos
        shuffle(ipis_list)

        # guardado del shuffle
        with open (os.path.join(os.getcwd(), f'resultados/{filename}-shuffled_spikes.dat'), 'w') as fake_spike_times:
            current_spike = 0
            for ipi in ipis_list:
                current_spike += ipi
                fake_spike_times.write(str(current_spike) + '\n')
        print(f'Archivo {Color.YELLOW}{filename}-shuffled_spikes.dat{Color.END} generado en resultados/')
        

    return