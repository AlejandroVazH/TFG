import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.train_lda_model import graph_LDA_projections

graph_LDA_projections(training_amounts_per_type=[5, 60], plot_labels=['5_patrones', '60_patrones'], reg_params=[0.5, 0])