import random
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from model_testing import get_encodings

## This is used to simulate our 3d Dataset that we will use to visualize. It is unnecessary once we have our data
def generate_dummy_values(low, high):
    coordinates = []
    for i in range(10000):
        coordinates.append(random.randint(low, high))
    return coordinates

class DataVisualizer:

    def __init__(self, dataframe, K, index_of_chosen_point):
        self.df = dataframe
        self.knn = KNN(dataframe=self.df, K=K)
        self.indices_to_plot = self.knn.compute_k_nearest_neighbours(index_of_chosen_point=index_of_chosen_point)



    def visualize(self):
        cols = self.df.columns
        fig = go.Figure(data=[go.Scatter3d(
            x=self.df[cols[0]][self.indices_to_plot],
            y=self.df[cols[1]][self.indices_to_plot],
            z=self.df[cols[2]][self.indices_to_plot],
            mode='markers',
            marker=dict(
                size=4,
                color=self.df[cols[3]][self.indices_to_plot],
            )
        )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
        # plt.scatter_3d(self.df, x=cols[0], y=cols[1], z=cols[2], color=cols[3], size_max=5).show()

    def get_recommended_songs(self):
        return self.df.iloc[self.knn.neighbours_indices] # returns the dataset containing the recommended songs


class KNN:
    COLOR_OF_CHOSEN_POINT = 'red'
    COLOR_OF_K_NEIGHBOURS = 'mediumpurple'

    def __init__(self):
        with torch.no_grad():
            self.points = get_encodings().numpy()

        self.df = pd.read_csv('data/data.csv')


    def compute_k_nearest_neighbours(self, index_of_chosen_point, k):
        chosen_point = self.points[index_of_chosen_point]

        dist_array = []
        for point in self.points:
            dist = np.linalg.norm(chosen_point - point) # computes the Euclidean distance between chosen point and x
            dist_array.append(dist)

        closest = np.argsort(dist_array)  # Sort it based on closest to furthest for each point
        neighbours_indices = self.k_nearest_neighbours_indices(closest, k)  # indexes of k nearest neighbours

        #self.update_point_color(index_of_chosen_point, self.COLOR_OF_CHOSEN_POINT)
        #self.update_point_color(self.neighbours_indices, self.COLOR_OF_K_NEIGHBOURS)

        indices = [index_of_chosen_point] + neighbours_indices
        knn_df = self.df.loc[indices]
        encodings_of_neighbours = self.points[indices]

        knn_df['encoding_x_column'] = encodings_of_neighbours[:, 0]
        knn_df['encoding_y_column'] = encodings_of_neighbours[:, 1]
        knn_df['encoding_z_column'] = encodings_of_neighbours[:, 2]

        knn_df['color_column'] = 0
        knn_df['symbol_column'] = 0

        return knn_df


    def get_closest_indices(self, array, number):
        return array[0: number]

    def k_nearest_neighbours_indices(self, array, k):
        return array[1:k + 1]

    def update_point_color(self, chosen_point, value):
        self.df.iloc[chosen_point, -1] = value

def main():
    knn = KNN()
    df = knn.compute_k_nearest_neighbours(69, 10)
    a = 10

if __name__ == "__main__":
    main()