import pandas as pd


dataset_name_to_loader = {    
    "sonar": pd.read_csv("../dataset/sonar.csv"),
    "german": pd.read_csv("../dataset/german.csv"),
    "spam": pd.read_csv("../dataset/spam.csv"),
    "adult": pd.read_csv('../dataset/adult.csv'),
    "credit": pd.read_csv('../dataset/credit-g.csv'),
    "electricity": pd.read_csv('../dataset/electricity.csv')
}

def candidates():
    print(dataset_name_to_loader.keys())

def loader(name):
    data = dataset_name_to_loader[name]
    print('Data shape: ',data.shape)
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
    return X, y