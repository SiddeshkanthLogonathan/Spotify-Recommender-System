import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

# Read data from .csv
data = pd.read_csv('./data/data.csv')
# print(data.columns)
# print(data.head(5))

# Drop irrelevant data
data.drop(['duration_ms', 'release_date', 'id'], axis=1, inplace=True)

# Normalize columns
data['popularity'] = data['popularity']/100
data['tempo'] = (data['tempo'] - 50)/100
data['loudness'] = (data['loudness'] + 60)/60

# Visualization
# print(data)
# plt.scatter(x=data['danceability'],
#             y=data['popularity'])
# plt.show()

# Convert to tensor
t = data.to_numpy()
print(t.shape)
