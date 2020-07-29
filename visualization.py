import random
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from model_testing import get_encodings
from data_loading import SpotifyRecommenderDataset

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

    def __init__(self, dataset: SpotifyRecommenderDataset()):
        self.dataset = dataset
        self.points = dataset.df[['encoding_x', 'encoding_y', 'encoding_z']].values

    def knn_query(self, song_id: str, k: int):
        boolean_index = self.dataset.df['id'] == song_id
        index_of_chosen_point = self.dataset.df.index[boolean_index].item()
        knn_indices = self.compute_k_nearest_neighbours(index_of_chosen_point, k)

        chosen_song_df = self.dataset.df.iloc[index_of_chosen_point, :]
        chosen_song_df['color'] = 1
        chosen_song_df['symbol'] = 1
        knn_df = self.dataset.df.iloc[knn_indices, :]
        knn_df['color'] = 0
        knn_df['symbol'] = 0

        return chosen_song_df, knn_df

    def compute_k_nearest_neighbours(self, index_of_chosen_point, k):
        chosen_point = self.points[index_of_chosen_point]

        dist_array = []
        for point in self.points:
            dist = np.linalg.norm(chosen_point - point) # computes the Euclidean distance between chosen point and x
            dist_array.append(dist)

        closest = np.argsort(dist_array)  # Sort it based on closest to furthest for each point

        return closest[1: k+1]

def main():
    dataset = SpotifyRecommenderDataset()
    knn = KNN(dataset)
    song_id = "0wwPcA6wtMf6HUMpIRdeP7"
    song_df, knn_df = knn.knn_query(song_id, 42)
    print(knn_df)
    print(song_df)
    a = 10

if __name__ == "__main__":
    main()