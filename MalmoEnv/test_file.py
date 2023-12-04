import numpy as np

with open('./value_trained.npy', 'rb') as f:
    value = np.load(f)

for row in value:
    print(row, end=" ")


with open('./magnitude_trained.npy', 'rb') as f:
    magnitude = np.load(f)

for row in magnitude:
    print(row, end=" ")
