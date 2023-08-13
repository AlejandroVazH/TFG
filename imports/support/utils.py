################################################################################
# Módulo utils
#
# Author: Alejandro Vázquez Huerta
# Descripción: Fichero con macros y objetos de soporte para el resto del fichero
################################################################################

import sys

################################################################################
# CONSTANTES
################################################################################

# clase para los colores
class Color:
    END = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    ORANGE = '\033[33m'
    CYAN = '\033[36m'
    YELLOW = '\033[93m'

# macro con la desviacion tipica para los kernels de la SDF
STANDARD_DEVIATION = 50 # Anchura del kernel 200ms => sd = ancho/4 = 50ms

# macro para deshabilitar los prints de tiempo
DISABLE_TIME_PRINTS = True

# macro para deshabilitar los prints de factores
DISABLE_FACTOR_PRINTS = True

# patrones que detectamos
PATTERNS = ['scallop', 'acceleration', 'rasp']

# metodos de fakear losd atos
FAKE_MODES = ['concatenate', 'surrogate']

# algoritmos de deteccion de patrones implementados
BURSTS_FINDING_MODE = ['carlson', 'robust_gaussian_surprise']

# Distancias definidas
DEFINED_DISTANCES = ['VP', 'VR']

# rango para detectar minimos y maximos de los bursts
MINIUM_RANGE = 5

################################################################################
# EL STRING DEL MENSAJE DE AYUDA
################################################################################

HELP_STRING = f"""
uso: python3 {sys.argv[0]}.py --<modo> <[parametros extra]>. Los modos con sus parametros, por orden recomendado de uso, son:

1.
--{Color.CYAN}CLEAN{Color.END}: Limpia los directorios de simulaciones, graficas y resultados.

2.
--{Color.CYAN}GET_SPIKES{Color.END}: Obtiene la ubicacion temporal de los spikes de una medicion de voltaje neuronal. El archivo de voltajes debe incluir cada medida de voltaje en una fila. El voltaje que se desea analizar debe estar en la cuarta (aunque esto es modificable para cada tipo de datos, la cuarta es por defecto la usada)

Parametros:
\t--{Color.CYAN}path{Color.END}: Path al fichero con los datos de voltajes.
Output:
\t{Color.CYAN}_spikes.dat{Color.END}: Archivo con los tiempos en los que se ha detectado un spike en las medidas de voltaje, uno por linea.

3.
--{Color.CYAN}ANALYZE_AND_FAKE{Color.END}: Analiza una secuencia de spikes y genera una secuencia falsa con igual distribucion. El archivo de spikes debe contener, en cada linea, el instante de deteccion de un spike, estando el archivo ordenado temporalmente.

Parametros:
\t--{Color.CYAN}data_name{Color.END}: Nombre del fichero de datos utilizado.
\tOutput:
\t{Color.CYAN}_ipis.dat{Color.END}: Archivo con la lista de IPIs detectados entre los spikes. Aparece un dato por linea.
\t{Color.CYAN}_ipis_histogram.dat{Color.END}: Datos correspondientes al numero de IPIs por rango de valores (datos del histograma). El formato de linea es: <numero de IPIs en ese rango> <extremo inferior del rango> <extremo superior>
\t{Color.CYAN}-fake_spikes.dat{Color.END}: Fichero con spikes de la secuencia de ipis simulada con igual distribucion que la de los datos. Aparece un instante por linea.
\t{Color.CYAN}-fake_ipis_histogram.dat{Color.END}: Datos del histograma de los IPIs falsos generados aleatoriamente con la misma CDF, con el mismo formato que _ipis_histogram.dat
\t{Color.CYAN}_ipis_histogram.eps{Color.END}: Histograma de los IPIs generados anteriormente.
\t{Color.CYAN}_ipis_histogram_with_PDF.eps{Color.END}: Histograma de los IPIs estandarizado junto a su PDF.
\t{Color.CYAN}_CDF.eps{Color.END}: Gráfica de la CDF asociada al histograma de IPIs.
\t{Color.CYAN}-fake_ipis_contrast.dat{Color.END}: Histograma de los IPIs junto a IPIs falsos generados aleatoriamente con la misma CDF.

4.
--{Color.CYAN}GENERATE_TRAINING_DATA{Color.END}: Genera datos de entrenamiento para el modelo de clasificación de bursts.

Parametros:
\t--{Color.CYAN}patterns_type{Color.END}: Tipo de patrones base a utilizar. Corresponde a los nombres que aparecen en las carpetas tipo modelos de patrones <Nombre>/.
\t--{Color.CYAN}bursts_max_finding_mode{Color.END}: Modo de detección de bursts. Actualmente se soportan carlson y robust_gaussian_surprise. Por defecto se usa carlson.
Parametros opcionales:
\t--{Color.CYAN}carlson_stats{Color.END}: Este parametro sin dato asociado se incluye cuando se quiere que se obtengan las estadisticas de carlson de los datos de entrenamiento.
Output:
\t{Color.CYAN}<patron>_spikes.dat{Color.END}: Spikes de los patrones generados. Hay un tren de spikes por linea.
\t{Color.CYAN}<patron>_bursts_data.dat{Color.END}: Datos de carlson de los spikes generados. Se corresponden por linea con los trenes de spikes del fichero anterior. Se incluye solo si se añade el parametro --carlson_stats
\t{Color.CYAN}<patron>_simulated_base_i.eps{Color.END}: Gráfica de los ipis de los patrones generados respecto a la base sobre la que se han generado. Muestra el patron medio y las desviaciones y el ECM.
\t{Color.CYAN}<patron>_simulated_examples_base_i.eps{Color.END}: Gráfica de los ipis de los patrones generados respecto a la base sobre la que se han generado.

5.
--{Color.CYAN}INSERT_PATTERNS{Color.END}: Añade los patrones de entrenamiento generados en el paso anterior sobre un tren de spikes original. Este metodo se usa principalmente para tests.

Parametros:
\t--{Color.CYAN}data_name{Color.END}: Nombre del fichero de datos utilizado.
\t--{Color.CYAN}patterns_type{Color.END}: Tipo de patrones base a utilizar. Corresponde a los nombres que aparecen en las carpetas tipo modelos de patrones <Nombre>/.
Output:
\t{Color.CYAN}-pattern_patterns_added.dat{Color.END}: Lista de patrones añadidos en la señal original. Se indica el tipo y sus spikes asociados.
\t{Color.CYAN}-pattern_spikes.dat{Color.END}: Tren de spikes original con los patrones introducidos
\t{Color.CYAN}-pattern_ipis_contrast.eps{Color.END}: Histograma de ipis comparativo de los datos antes y después de poner los patrones.

6.
--{Color.CYAN}BURSTS{Color.END}: Saca los bursts de un tren de spikes.

Parametros:
\t--{Color.CYAN}data_name{Color.END}: Nombre del fichero de datos utilizado.
\t--{Color.CYAN}bursts_finding_mode{Color.END}: Modo de detección de bursts. Actualmente se soportan carlson y robust_gaussian_surprise. Por defecto se usa carlson.
Parametros opcionales:
\t--{Color.CYAN}save_SDF{Color.END}: Se añade si se desea guardar la SDF computada del tren de spikes
\t--{Color.CYAN}get_SDF_from_file{Color.END}: Se añade si ya se ha computado y guardado la SDF, para extraerla de fichero.
Output:
\t{Color.CYAN}_bursts.dat{Color.END}: archivo con los spikes que componen los bursts, un burst por linea

7.
--{Color.CYAN}CARLSON{Color.END}: Analiza patrones en una secuencia de spikes en una red neuronal usando las tecnicas de Spike Density Function definidas por Bruce A. Carlson. Las medidas deben estar en un txt en el que haya una sola columna con el instante (ms) en el que se detecta un spike.

Parametros:
\t--{Color.CYAN}data_name{Color.END}: Nombre del fichero de datos utilizado.
\t--{Color.CYAN}bursts_finding_mode{Color.END}: Modo de detección de bursts. Actualmente se soportan carlson y robust_gaussian_surprise. Por defecto se usa carlson
Parametros opcionales:
\t--{Color.CYAN}save_SDF{Color.END}: Se añade si se desea guardar la SDF computada del tren de spikes
\t--{Color.CYAN}get_SDF_from_file{Color.END}: Se añade si ya se ha computado y guardado la SDF, para extraerla de fichero.
\t--{Color.CYAN}save_SDD{Color.END}: Se añade si se desea guardar la SDD computada del tren de spikes
\t--{Color.CYAN}get_SDD_from_file{Color.END}: Se añade si ya se ha computado y guardado la SDD, para extraerla de fichero.
\t--{Color.CYAN}save_bursts_max{Color.END}: Se añade si se desean guardar los maximos de los bursts del tren de spikes.
\t--{Color.CYAN}get_bursts_max_from_file{Color.END}: Se añade si ya se han obtenido los maximos de los bursts y guardado en un fichero, para extraerlos del fichero.
Output:
\t--{Color.CYAN}_bursts_data.dat{Color.END}: archivo con los valores de las estadisticas calculadas para cada burst.

8.
--{Color.CYAN}SUPERVISED_PCA_DFA_BURSTS_CLASSIFICATION{Color.END}: Este es el metodo mas importante. Requiere de haberse ejecutado antes el GENERATE_TRAINING_DATA. Detecta y clasifica por un algoritmo de clasificacion supervisada los bursts detectados en un tren de spikes.

Parametros:
\t--{Color.CYAN}data_name{Color.END}: Nombre del fichero de datos utilizado.
\t--{Color.CYAN}training_data{Color.END}: Tipo de patrones base a utilizar como entrenamiento. Corresponde a los nombres que aparecen en las carpetas tipo modelos de patrones <Nombre>/. Actualmente tenemos patrones de Carlson y de Angel.
Output:
\t{Color.CYAN}_classification.dat{Color.END}: Archivo en el que se guarda el centro de cada burst con su clasificacion correspondiente.
"""

################################################################################
# FUNCIONES AUXILIARES PARA LOS MODULOS PRINCIPALES
################################################################################

#
# Funcion para hacer print de errores rapido y salir
#
def error_print(error_message: str):
    print(f"{Color.RED}ERROR{Color.END}: {error_message}\n")
    sys.exit(1)

#
# Funcion para hacer print de warning
#
def warning_print(warning_message: str) -> None:
    print(f"{Color.ORANGE}WARNING{Color.END}: {warning_message}")
    return

#
# funcion para pasar de milisegundos a segundos en las graficas
#
def miliseconds_to_seconds(x, pos):
    return '{}'.format(round(x/1000.0, 3))

#
# funcion para pasar de spikes/ms a spikes/s en las graficas
#
def spikes_per_ms_to_spikes_per_s(x, pos):
    return '{}'.format(round(x*1000, 3))

#
# Interseccion de intervalos
#
def intervalIntersection(A, B) -> list[int]:
    ans = None
    # Let's check if A[i] intersects B[j].
    # lo - the startpoint of the intersection
    # hi - the endpoint of the intersection
    lo = max(A[0], B[0])
    hi = min(A[1], B[1])
    if lo <= hi:
        ans = [lo, hi]

    return ans
