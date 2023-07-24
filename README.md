# Metodologías de detección, análisis y clasificación de patrones en secuencias temporales de pulsos

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
* [training_data](imports/training_data.py): Incluye las funciones para generar y rizar patrones base, además de obtener filtros para detección y las estadísticas de Carlson de estos patrones.

## Ámbitos de uso
Se describen a continuación las diferentes modalidades de uso que se le pueden dar a este repositorio.

### Generar patrones de nuevos tipos
Las funciones que se utilizan para generar patrones y operar con ellos están en el módulo [training data](imports/training_data.py). Incluye una función para generar los patrones base, `generate_base_patterns`, y otra para los rizados de los patrones base, `generate_training_data`. La primera función genera patrones base utilizando características de su secuencia de IPIs, y la segunda genera rizados a magnitud dada por el parámetro `max_half_range`. No es recomendable alterar el parámetro `min_IPI_length` de la segunda función a menos que se sepa que es normal que aparezcan IPIs de menos de 5 unidades de tiempo. El resto de funciones se utilizan para extraer información de los mismos, o los filtrados en detección para detectar patrones simimlares a los generados.

Si se desea añadir un patrón, se deben seguir estos pasos:
1. Añadir en la macro `PATTERNS` de [utils](imports/support/utils.py) el nombre del patrón.
2. Añadir, en la macro `PATTERNS_THRESHOLD_MAX_FACTOR`, el porcentaje respecto al valor del máximo de la SDF del patrón respecto al cual se saca la zona central del mismo, entendida como la zona entre mínimos locales por encima de dicho threshold. Lo recomendable es que esté entre 0.6 y 0.3.
3. Añadir en `generate_base_patterns` el código que genere la secuencia de IPIs del nuevo patrón.

Para observar cómo quedan los nuevos patrones se pueden utilizar estos scripts de la carpeta del mismo nombre:
* [graph_base_patterns_IPIs_and_vibrated](scripts/graph_base_patterns_IPIs_and_vibrated.py): Grafica 5 ejemplos de patrones generados a rizados de 0, 5 y 10 unidades de tiempo.
* [graph_base_patterns_SPIs](scripts/graph_base_patterns_SPIs.py): Grafica 500 patrones de cada tipo juntos y superpuestos a diferentes vibraciones. Recomendable usarlo para verificar la variabilidad de los patrones generados.
* [graph_training_data_SPIs](scripts/graph_training_data_SPIs.py): Es como el anterior, pero en vez de graficarlos todos juntos, genera un pdf y muestra uno por página.
* [perform_base_patterns_PCA](scripts/perform_base_patterns_PCA.py): Permite aplicar PCA a los patrones generados y vibrados a 5ms y 10ms.

### Detectar y analizar patrones
Para esta explicación, se presupone que se han extraído los ficheros de pulsos utilizando [este](https://github.com/angellareo/bio-utils) repositorio descrito anteriormente. Los ficheros de pulsos deben consistir en una sola columna con el instante en el que se produce cada pulso en una fila, y estar en formato (.dat). Una vez hecho esto, se deben seguir estos pasos:

* Entrenar los clasificadores que se deseen usar. Los clasificadores se entrenan con los patrones registrados (ver uso anterior).
* Utilizar el fichero [perform_signal_detection](scripts/perform_signal_detection.py) cargando los clasificadores que se deseen. Se puede copiar y adaptar el uso de los clasificadores ahora mismo implementados simplemente cambiando dónde se carga y usa cada uno.
