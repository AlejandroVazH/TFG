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

## Utilización para generar patrones falsos


## Utilización para detectar patrones
Para esta explicación, se presupone que se han extraído los ficheros de pulsos utilizando el repositorio descrito anteriormente. Los ficheros de pulsos deben consistir en una sola columna con el instante en el que se produce cada pulso en una fila, y estar en formato (.dat). 
