# Metodologías de detección, análisis y clasificación de patrones en secuencias temporales de pulsos
* [Instalación](#Instalación)
* [Estructura](#Estructura)

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
* [imports](imnports/): Módulos que implementan metodologías de detección, análisis e implementación de patrones en secuencias temporales de pulsos, junto a otras para implementar y probar el simulador de patrones.
* [scripts](scripts/): Scripts para probar e utilizar los módulos anteriores. Cuenta con un subdirectorio para resultados auxiliares (.dat) y otro para gráficas auxiliares (.eps, .pdf y .jpg).
* [resultados](resultados/), junto a [gráficas](graficas/): Son los directorios en los que los módulos imprimen, si se solicita con las opciones correspondientes, los diferentes resultados obtenidos.

## Ámbitos de uso
Se describen a continuación las diferentes modalidades de uso que se le pueden dar a este repositorio.

### Utilización para generar patrones falsos
Las funciones que se utilizan para generar patrones y operar con ellos están en el módulo [training data](imports/training_data.py). Incluye una función para generar los patrones base, `generate_base_patterns`, y otra para los rizados de los patrones base, `generate_training_data`. La primera función genera patrones base utilizando características de su secuencia de IPIs, y la segunda genera rizados a magnitud dada por el parámetro `max_half_range`. No es recomendable alterar el parámetro `min_IPI_length` de la segunda función a menos que se sepa que es normal que aparezcan IPIs de menos de 5 unidades de tiempo. El resto de funciones se utilizan para extraer información de los mismos, o los filtrados en detección para detectar patrones simimlares a los generados.

Si se desea añadir un patrón, se deben seguir estos pasos:
1. Añadir en la macro `PATTERNS` de [utils](imports/support/utils.py) el nombre del patrón.
2. Añadir, en la macro `PATTERNS_THRESHOLD_MAX_FACTOR`, el porcentaje respecto al valor del máximo de la SDF del patrón respecto al cual se saca la zona central del mismo, entendida como la zona entre mínimos locales por encima de dicho threshold. Lo recomendable es que esté entre 0.6 y 0.3.
3. Añadir en `generate_base_patterns` el código que genere la secuencia de IPIs del nuevo patrón.

Para observar cómo quedan los nuevos patrones se pueden utilizar estos scripts de la carpeta del mismo nombre:
* [graph_base_patterns_IPIs_and_vibrated](scripts/graph_base_patterns_IPIs_and_vibrated.py): Grafica 5 ejemplos de patrones generados a rizados de 0, 5 y 10 unidades de tiempo.
* [graph_base_patterns_SPIs](scripts/graph_base_patterns_SPIs.py): Grafica 500 patrones de cada tipo juntos y superpuestos a diferentes vibraciones. Recomendable usarlo para verificar la variabilidad de los patrones generados.
* [graph_training_data_SPIs](scripts/graph_training_data_SPIs.py): Es como el anterior, pero en vez de graficarlos todos juntos, genera un pdf y muestra uno por página.

## Utilización para detectar patrones
Para esta explicación, se presupone que se han extraído los ficheros de pulsos utilizando el repositorio descrito anteriormente. Los ficheros de pulsos deben consistir en una sola columna con el instante en el que se produce cada pulso en una fila, y estar en formato (.dat). 
