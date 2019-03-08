import matplotlib
import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import matplotlib.pyplot as plt

#%matplotlib inline


class SeedClassifier():

    def __init__(self):
        # Crear dataframe a partir de .csv
        df = pd.read_csv("data/seeds.csv")

        # Mostrar número de filas y columnas del dataframe
        print("Filas: " + str(df.shape[0]) + " Columnass: " + str(df.shape[1]))

        # Mostrar las primeras 30 filas
        display(df.head(30))

        # Hallar número de valores únicos en cada columna
        display("Valores únicos por columnas: " + '\n' + str(df.nunique()))

        # Comprobar la existencia de valores nulos en el dataset
        display("Nulos por columnas: " + '\n' + str(df.isnull().any()))

        # Mostrar información general del dataframe
        display(df.info())

        # Descripción analítica básica del dataframe
        display(df.describe())

        # Mostrar matriz de correlación de variables
        # Pista: explore plt.matshow y corr() de un dataframe


        # Mostrar correlaciones como una función discreta entre las diferentes variables con una matriz
        # útil para apreciar relaciones lineales

        # Pista: explore pd.plotting.scatter_matrix

        # Crear un dataframe solo con la columna de la variable dependiente

        # Crear un dataframe con las variables independientes


        # Definir un RF con diferentes hiperparámetros vistos en lecciones anteriores (¡experimentar!)


        # Entrenar un RF con la totalidad del dataset


        ###### Snippet para imprimir resultados, df es la variable que refiere
        ###### al dataframe y clf al clasificador, cambiarlas si es necesario

        # feature_list = list(zip(df.columns.values, clf.feature_importances_))
        # zip empareja los elementos de la lista
        # sorted_by_importance = sorted(feature_list, key=lambda x: x[1], reverse=True)

        # for feat, value in sorted_by_importance:
        #    print(feat, value)

        # Crear un dataframe solo con la columna de la variable dependiente

        # Crear un dataframe con las variables independientes


        # Partir el test en cierta proporción (¡experimentar!)
        # X_train, X_test, y_train, y_test = # ?

        ###### Snippet para imprimir resultados, X_train es la variable que refiere
        ###### a la porcion de entrenamiento y X_test a la de test

        # print("El dataset de training tiene {} elementos.".format(X_train.shape[0]))
        # print("El dataset de testing tiene {} elementos.".format(X_test.shape[0]))

        # Definir un clasificador, recordar que se está haciendo una clasificación ternaria


        # Entrenar el clasificador con el dataset de train


        # Predecir valores para las variables independientes de test


        # Calcular la precisión
        # Pista: explorar sklearn.metrics.accuracy_score


if __name__ == '__main__':
    classifier = SeedClassifier()
