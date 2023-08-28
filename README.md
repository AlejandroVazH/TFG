# Metodologías de detección, análisis y clasificación de patrones en secuencias temporales de pulsos

En este github se incluyen los códigos desarrollados por Alejandro V.H., propietario de este repositorio, para el desarrollo del TFG ''Estudio y análisis de metodologías para detectar secuencias temporales de actividad neuronal''. Los datos de los patrones de secuencias temporales de pulsos provienen de varios experimentos realizados por el Grupo de Neurocomputación  Biológica ([GNB](https://www.uam.es/uam/investigacion/grupos-de-investigacion/detalle/f1-182)) de la Universidad Autónoma de Madrid, que han sido suministrados por el mismo:

1- A. Lareo, C.G. Forlim, R.D. Pinto, P. Varona, and F.B. Rodriguez, Front. Neuroinform., ront. Neuroinform., 06 October 2016Volume 10 - 2016 | https://doi.org/10.3389/fninf.2016.00041.

2- A. Lareo, P. Varona, and F.B. Rodriguez, Front. Neuroinform., 28 June 2022 Volume 16 - 2022 | https://doi.org/10.3389/fninf.2022.912654.


## Instalación
Para utilizar este repositorio se necesita `Python 3.10.6` junto a los siguientes paquetes:
* [Sklearn](https://scikit-learn.org/stable/install.html)
* [Numpy](https://numpy.org/install/)
* [Scipy](https://scipy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Quantities](https://pypi.org/project/quantities/)
* [Elephant](https://elephant.readthedocs.io/en/v0.7.0/install.html)
* [Joblib](https://joblib.readthedocs.io/en/stable/)

A parte, se necesitará instalar [este](https://github.com/angellareo/bio-utils) repositorio ubicando su carpeta principal a la altura del directorio scripts.

## Estructura
El presente directorio consta de los siguientes elementos:
* [imports](imports/): Módulos que implementan metodologías de detección, análisis e implementación de patrones en secuencias temporales de pulsos, junto a otras para implementar y probar el simulador de patrones.
* [scripts](scripts/): Scripts para probar e utilizar los módulos anteriores. Cuenta con un subdirectorio para resultados auxiliares (.dat) y otro para gráficas auxiliares (.eps, .pdf y .jpg).
* [resultados](resultados/), junto a [gráficas](graficas/): Son los directorios en los que los módulos imprimen, si se solicita con las opciones correspondientes, los diferentes resultados obtenidos.

### Módulo Imports
Incluye los diferentes módulos que implementan las funcionalidades de detección, análisis y clasificación. Incluye estos módulos:
* [analyze](imports/analyze.py): Analiza y modifica una secuencia de datos. Permite obtener el histograma de IPIs de la secuencia y perturbarla por los diferentes métodos de surrogate definidos (actualmente el shuffle y el método del histograma de IPIs).
* [combine_spikes_files](imports/combine_spikes_files.py): Recibe una lista de ficheros con secuencias de pulsos y los combina concatenando las mismas.
* [get_spikes](imports/get_spikes.py): Extrae de un fichero de mediciones de voltaje la secuencia de pulsos. Utiliza [este](https://github.com/angellareo/bio-utils) git.
* [insert_patterns](imports/insert_patterns.py): Inserta un conjunto de patrones en una secuencia de pulsos. La secuencia se concatena con la baseline con un IPI menor a 110 unidades de tiempo.
* [pca_dfa](imports/pca_dfa.py): Aplica PCA a un conjunto de datos de estadísticas de Carlson etiquetados y grafica los resultados.
* [train_lda_model](imports/train_lda_model.py): Este módulo incluye las funciones para entrenar los diferentes clasificadores utilizados. Para LDA permite, además, proyectar datos en su plano de proyección para visualizar la transformación. Para los métodos de las distancias permite aplicar análisis por discriminante de distancia para hallar parámetros ideales.
* [training_data](imports/training_data.py): Incluye las funciones para generar y rizar patrones base, además de obtener filtros para detección y las estadísticas de Carlson de estos patrones. Este es el módulo que incluye toda la funcionalidad referida en el TFG como el ''simulador de patrones''.

A parte, se incluye un módulo `support` que aporta funcionalidades auxiliares sobre las que se apoyan los otros módulos. Se incluyen los siguientes submódulos:
* [bursts_finding](imports/support/bursts_finding.py): Aporta los métodos de detección de bursts y extracción de los patrones buscando los primeros mínimos laterales que aparezcan. Incluye funciones para implementar Robust Gaussian Surprise, aunque necesita de mucho más refinamiento para funcionar correctamente.
* [carlson](imports/support/carlson.py): Extrae las 20 estadísticas de Carlson de una lista de SPIs.
* [density_functions](imports/support/density_functions.py): Incluye una función para transformar una secuencia de pulsos en su SDF y otra para transformarla en su SDD.
* [utils](imports/support/utils.py): Aporta macros y funciones de apoyo para los otros módulos.

### Módulo scripts
Conjunto de scripts que utilizan los módulos anteriores para llevar a cabo el análisis de secuencias de pulsos y el desarrollo del simulador. Incluye estos ficheros:
* [analyze_generate_fake_data_script](scripts/analyze_generate_fake_data_script.py): Aplica una modalidad de surrogate implementada a un fichero de pulsos pasado con el parámetro `--filename`. Genera uno nuevo con el surrogate aplicado.
* [debug_base_patterns_carlson_stats](scripts/debug_base_patterns_carlson_stats.py): Permite visualizar las marcas temporales de Carlson sobre 100 patrones generadosde cada tipo.
* [debug_patterns_detection](scripts/debug_patterns_detection.py): Obtiene las SPIs detectadas en una secuencia de pulsos pasada con el parámetro `--filename` con un threshold concreto, pasado como argumento con el parámetro `--threshold`.
* [evaluate_detection_per_noise](scripts/evaluate_detection_per_noise.py): Evalúa la detección de patrones generados por el simulador sobre una señal de pulsos de fondo por el método de Carlson en función del rizado de los patrones. Tan solo recibe el fichero con los pulsos de fondo con el parámetro `--filename`, e imprime los patrones insertados y detectados junto a los detectados no insertados, además de los porcentajes de detección por tipo, utilizando el threshold ideal para cada conjunto de patrones, en función del rizado. Se puede configurar para utilizar nuevos tipos de surrogate junto con nuevos tipos de patrones.
* [evaluate_detection_per_noise](scripts/evaluate_detection_per_noise.py): Evalúa la detección de patrones generados por el simulador sobre una señal de pulsos de fondo por el método de Carlson en función del threshold del método. Tan solo recibe el fichero con los pulsos de fondo con el parámetro `--filename`, e imprime los patrones insertados y detectados junto a los detectados no insertados, además de los porcentajes de detección por tipo, utilizando el threshold ideal para cada conjunto de patrones, en función del threshold. Se puede configurar para utilizar nuevos tipos de surrogate junto con nuevos tipos de patrones.
* [graph_IPIs_SDF_SDD](scripts/graph_IPIs_SDF_SDD.py): Genera una gráfica en la que se muestra la transformación por SDF y SDD de cada tipo de patrón. Para añadir uno nuevo, requiere de añadir el nombre del patrón a la hora de graficarlo (línea 22).
* [graph_SDF_histogram_with_patterns](scripts/graph_SDF_histogram_with_patterns.py): Genera un histograma de IPIs de una secuencia de pulsos de fondo pasada por terminal con el parámetro `--filename` antes y después de insertar patrones.
* [graph_base_patterns_IPIs_and_vibrated](scripts/graph_base_patterns_IPIs_and_vibrated.py): Genera una gráfica de 5 patrones de cada tipo vibrados a 5 y 10 milisegundos.
* [graph_base_patterns_SPIs](scripts/graph_base_patterns_SPIs.py): Genera una gráfica en la que se muestran, a diferentes vibraciones, 50 patrones de cada tipo superpuestos. Se puede utilizar para verificar la variabilidad en la generación de patrones.
* [graph_baseline](scripts/graph_baseline.py): Grafica los IPIs de una sección de una secuencia de pulsos pasada por terminal con el parámetro `--filename`. Se puede utilizar para visualizar diferentes secciones de la señal.
* [graph_fitness_function_transformation](scripts/graph_fitness_function_transformation.py): Genera una gráfica que muestra la transformación que aplica la función de fitness a un scallop en diferentes imágenes consecutivas.
* [graph_inserted_patterns_example](scripts/graph_inserted_patterns_example.py): Grafica 5 patrones de cada tipo insertados en una señal de fondo pasada por terminal con el parámetro `--filename` y transformada en su SDF.
* [graph_og_and_surrogated_histogram](scripts/graph_og_and_surrogated_histogram.py): Genera un histograma de IPIs comparativo entre una señal de pulsos pasada por terminal con el parámetro `--filename` y esa misma señal pero transformada por surrogate.
* [graph_signal_SDF](scripts/graph_signal_SDF.py): Grafica los 50000 primeros valores de la SDF de una señal de pulsos pasada por terminal con el parámetro `--filename`.
* [graph_signal_boxplot](scripts/graph_signal_boxplot.py): Grafica la boxplot de la SDF de una señal de pulsos pasada por terminal con el parámetro `--filename`.
* [graph_temporal_marks](scripts/graph_temporal_marks.py): Grafica la ubicación temporal de los marcadores St, Et, R1t, R2t, F1t y F2t en un patrón aceleración generado.
* [graph_training_data_SPIs](scripts/graph_training_data_SPIs.py): Genera un pdf con patrones generados y rizados a 5ms para su observación individual.
* [graph_van_rossum_schema](scripts/graph_van_rossum_schema.py): Genera una gráfica en la que se muestra el funcionamiento de la distancia de van Rossum entre dos scallops distintos.
* [perform_LDA_projection](scripts/perform_LDA_projection.py): Genera la gráfica de la proyección de patrones test vibrados a 5ms por LDA entrenado a diferente número de parámetros de entrenamiento y con diferentes parámetros de regularización.
* [perform_base_patterns_PCA](scripts/perform_base_patterns_PCA.py): Aplica PCA a las estadísticas de Carlson de 1500 patrones de cada tipo generados y rizados a 5 y 10 milisegundos.
* [perform_distance_discriminant_analysis](scripts/perform_distance_discriminant_analysis.py): Efectúa el análisis por discriminante de distancia para los patrones considerados y los clasificadores por distancia definidos que requieran de hiperparámetro para localizar el hiperparámetro ideal. Actualmente se usa para Victor-Purpura y van Rossum.
* [perform_signal_detection](scripts/perform_signal_detection.py): Este es el script al que se debe llamar para aplicar detección y clasificación en una señal. Recibe el fichero con la secuencia de pulsos a analizar con el parámetro `--filename` y el threshold a utilizar en detección con el parámetro `--threshold`. Requiere de tener guardado, al menos, un clasificador si se quiere usar para clasificación (ver siguientes ficheros). Para utilizar uno nuevo se puede copiar y pegar el código para cualquier clasificador, simplemente cambiando sus nombres por los correspondientes, ya que todos se ejecutan bajo la misma estructura, salvo LDA y QDA que utilizan las estadísticas de Carlson.
* [train_LDAQDA_model_script](scripts/train_LDAQDA_model_script.py): Entrena clasificadores LDA y QDA y los guarda en formato .jl.
* [train_distances_model_script](scripts/train_distances_model_script.py): Entrena clasificadores de distancia y los guarda en formato .jl.
* [train_fitness_model_script](scripts/train_fitness_model_script.py): Entrena el clasificador fitness y lo guarda en formato .jl.

### Funcionamiento

El código de este github está enfocado a procesar datos de secuencias temporales de pulsos a posteriori, es decir, tras su extracción. Es a partir de estos datos desde los que se desarrolla el análisis de secuencias nuevas y métodos existentes.
* Para analizar una señal: Lo primero que se debe hacer es procesarla usando el módulo `get_spikes`, cuya guía se puede encontrar en el github original del mismo [aquí](https://github.com/angellareo/bio-utils). Una vez obtenida la secuencia de pulsos, si solo se desea detectar patrones se puede introducir en el script `perform_signal_detection` para procesarla directamente, aunque es recomendable utilizar el script `graph_signal_boxplot` para obtener un threshold aproximado para las SPIs de corta duración.
* Si, en el caso anterior, se conocen patrones que podrían aparecer en la señal, y se desea verificar su presencia, se deben introducir dichos patrones en el generador y entrenar los clasificadores que se deseen usar con los scripts `train_`.

Para introducir nuevos patrones, o modificar con los que se trabaja, se deben seguir estos pasos:
1. En el módulo `training_data`, en la función `generate_base_patterns`, se debe añadir la generación del patrón nuevo, añadiendo variabilidad en las partes que corresponda.
2. Hay que añadir el patrón en la variable `PATTERNS_THRESHOLD_MAX_FACTOR` de ese mismo módulo. Además, en el módulo `utils` hay que actualizar la variable `PATTERNS` con los patrones que utilicemos.
