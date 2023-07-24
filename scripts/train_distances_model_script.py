import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.train_lda_model import train_distances_model

train_distances_model(graph_database=True)