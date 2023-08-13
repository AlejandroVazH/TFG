################################################################################
# Módulo pca_dfa
#
# Author: Alejandro Vázquez Huerta
# Descripción: Originariamente este módulo implementaba toda la funcionalidad
#              de los clasificadores por función discriminante y PCA.
#              Actualmente solo implementa PCA.
################################################################################

import os
from numpy import where, array, ndarray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imports.support.utils import Color, PATTERNS

################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Funcion para graficar los datos test en las 2 primeras componentes principales
#
def __graph_pca_data(training_data_stats: ndarray, training_data_labels: ndarray, pca: PCA, num_coord: int, label: str = ''):
    # las separamos por tipo
    training_data_labels = array(training_data_labels)
    class_index = {p : where(training_data_labels == p)[0].tolist() for p in PATTERNS}
    pc_training_data_classes = {p: training_data_stats[class_index[p]] for p in PATTERNS}

    # grafica 3d
    fig = plt.figure()
    x_axis = fig.add_subplot(projection='3d')
    for p in PATTERNS:
        x_axis.scatter(xs=pc_training_data_classes[p][:,0], ys=pc_training_data_classes[p][:,1], zs=pc_training_data_classes[p][:,2], label=f'{p} tests', s=2.5)
    x_axis.set_xlabel(f'Factor {1}, {round(pca.explained_variance_ratio_[0]*100, 2)}% var')
    x_axis.set_ylabel(f'Factor {2}, {round(pca.explained_variance_ratio_[1]*100, 2)}% var')
    x_axis.set_zlabel(f'Factor {3}, {round(pca.explained_variance_ratio_[2]*100, 2)}% var')
    x_axis.set_title(f'Separacion de los burst tests por PCs')
    x_axis.legend()
    plt.show()
    
    # guardamos los pesos de las componentes en una tabla
    with open(os.path.join(os.getcwd(), f'scripts/auxiliar_results/pesos_estadisticas_PCA_{label}.dat'), 'w') as f:
        f.write('# estadistica ')
        for i in range(num_coord):
            f.write(f'# peso factor {i+1} ')
        f.write('\n')
        for stat, peso_en_factor in zip(
            ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3'],
            [[round(s, 4) for s in pca.components_[:3,i]] for i in range(pca.n_features_in_)]
        ):
            f.write(f'{stat} ')
            for peso in peso_en_factor:
                f.write(f'{peso} ')
            f.write('\n')
    print(f'Archivo {Color.YELLOW}pesos_estadisticas_PCA_{label}.dat{Color.END} generado en scripts/auxiliar_results/')

    return

################################################################################
# FUNCION PRINCIPAL
################################################################################

#
# Analiza una matriz de datos data con clases labels por PCA
#
def perform_PCA_patterns_analysis(data: list[list], labels: list[str], graph_label: str = '') -> None:
    # analisis por PCAs de los datos originales
    data_centered = StandardScaler().fit_transform(data)
    pca = PCA().fit(X=data_centered)
    data_pcs = pca.transform(data_centered)
    __graph_pca_data(
        training_data_stats=data_pcs,
        training_data_labels=labels,
        pca=pca,
        num_coord=3,
        label=graph_label
    )

    # analizamos juntos scallops y rasps
    #data_centered_without_acc = StandardScaler().fit_transform(data[where(labels != 'acceleration')[0].tolist()])
    #pca = PCA().fit(X=data_centered_without_acc)
    #data_pcs_without_acc = pca.transform(data_centered_without_acc)
    #__graph_pca_data(
    #    training_data_stats=data_pcs_without_acc,
    #    training_data_labels=labels[where(labels != 'acceleration')[0].tolist()],
    #    pca=pca,
    #    num_coord=3,
    #    label=f'datos_de_entrenamiento_{pattern_folder}'
    #)

    return
