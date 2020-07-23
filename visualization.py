import random
import plotly.express as plt
import pandas as pd
import numpy as np
from scipy.spatial import distance

## This is used to simulate our 3d Dataset that we will use to visualize. It is unnecessary once we have our data
def generate_dummy_values(low, high):
    coordinates = []
    for i in range(100):
        coordinates.append(random.randint(low, high))
    return coordinates

class DataVisualizer:
    COLOR_OF_CHOSEN_POINT = 'chosen song'
    COLOR_OF_K_NEIGHBOURS = 'k neighbours'

    def __init__(self, dataframe, K, index_of_chosen_point):
        self.df = dataframe
        self.compute_k_nearest_neighbours(K=K, index_of_chosen_point=index_of_chosen_point)

    def compute_k_nearest_neighbours(self, K, index_of_chosen_point):
        points = np.transpose([self.df['dim1'].to_numpy(), self.df['dim2'].to_numpy(), self.df['dim3'].to_numpy()]) # df converted to array

        D = distance.squareform(distance.pdist(points))  # Gives us a matrix of distances from each point to all others
        closest = np.argsort(D, axis=1)  # Sort it based on closest to furthest for each point

        neighbours_indices = self.k_nearest_neighbours_indices(closest, K, index_of_chosen_point) # indexes of k nearest neighbours

        self.update_point_color(df, index_of_chosen_point, self.COLOR_OF_CHOSEN_POINT)
        self.update_point_color(df, neighbours_indices, self.COLOR_OF_K_NEIGHBOURS)

    def get_k_neighbours(self, df, indices):
        return self.df.loc[indices]

    def k_nearest_neighbours_indices(self, matrix, k, x):
        return matrix[x, 1:k + 1]

    def update_point_color(self, df, chosen_point, value):
        self.df.iloc[chosen_point, -1] = value

    def visualize(self):
        cols = self.df.columns
        plt.scatter_3d(self.df, x=cols[0], y=cols[1], z=cols[2], color=cols[3]).show()

## This will be replaced with our processed DataFrame ===
df = pd.DataFrame()
df['dim1'] = generate_dummy_values(1, 100)
df['dim2'] = generate_dummy_values(5, 60)
df['dim3'] = generate_dummy_values(10, 50)
df['color'] = ['song'] * 100
## ======================================================

DataVisualizer(df, K=5, index_of_chosen_point=6).visualize()