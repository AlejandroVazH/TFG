#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.train_lda_model import train_fitness_model

# entrenamos los modelos
print(f'Generando el modelo...')
train_fitness_model()
