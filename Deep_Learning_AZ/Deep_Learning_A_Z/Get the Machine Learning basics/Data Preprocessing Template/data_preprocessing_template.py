# Data Preprocessing
# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
