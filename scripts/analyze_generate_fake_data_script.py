#
# Script hecho por: Alejandro V.H.
#

import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.support.utils import  error_print
from imports.analyze import data_analysis

# vamos a necesitar estos parametros
mode = None
filename = None
for param_pos in range(0, len(sys.argv), 1):
    if sys.argv[param_pos] == '--mode':
        mode = sys.argv[param_pos + 1]
    if sys.argv[param_pos] == '--filename':
        filename = sys.argv[param_pos + 1]
if mode == None:
    error_print('Parametro --mode no detectado')
if filename == None:
    error_print('Parametro --data_name no detectado')

data_analysis(filename=filename, fake_mode=mode)
