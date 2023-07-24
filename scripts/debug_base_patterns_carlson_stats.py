import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.training_data import generate_training_data, get_carlson_stats_of_perturbations,  generate_base_patterns
from imports.pca_dfa import perform_PCA_patterns_analysis

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(num_of_bases=100)
print(f'Generando vibraciones...')
training_data = generate_training_data(og_patterns=base_patterns, num_patterns=1)
print(f'Genreando el fichero de debug de estadisticas...')
get_carlson_stats_of_perturbations(new_patterns=training_data, debug=True)