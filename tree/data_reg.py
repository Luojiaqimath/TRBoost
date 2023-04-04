import numpy as np
import pandas as pd
from scipy.io import arff


dataset_name_to_loader = {
    "concrete": pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),
    "energy":  pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ).iloc[:, :-1],
    "power": pd.read_excel("../dataset/power.xlsx"),
    "kin8nm": pd.DataFrame(arff.loadarff('../dataset/kin8nm.arff')[0]),
    "wine_quality":pd.read_csv('../dataset/wine_quality.csv'),
    "california":pd.read_csv('../dataset/california.csv')
}

def candidates():
    print(dataset_name_to_loader.keys())

def loader(name):
    data = dataset_name_to_loader[name]
    print('Data shape: ',data.shape)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    return X, y