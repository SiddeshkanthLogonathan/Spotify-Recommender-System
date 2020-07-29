import random
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

    def __init__(self, dataframe, K):
        self.df = dataframe
        self.K = K
        self.neighbours_indices = []


    def compute_k_nearest_neighbours(self, index_of_chosen_point):
        points = np.transpose([self.df['enc1'].to_numpy(), self.df['enc2'].to_numpy(),
                               self.df['enc3'].to_numpy()])  # df converted to array
        chosen_point = points[index_of_chosen_point]

        dist_array = []
        for x in points:
            dist = np.linalg.norm(chosen_point - x) # computes the Eucledian distance between chosen point and x
            dist_array.append(dist)

        closest = np.argsort(dist_array)  # Sort it based on closest to furthest for each point
        self.neighbours_indices = self.k_nearest_neighbours_indices(closest, self.K)  # indexes of k nearest neighbours

        self.update_point_color(index_of_chosen_point, self.COLOR_OF_CHOSEN_POINT)
        self.update_point_color(self.neighbours_indices, self.COLOR_OF_K_NEIGHBOURS)

        return self.get_closest_indices(closest, 3000)

    def get_closest_indices(self, array, number):
        return array[0: number]

    def k_nearest_neighbours_indices(self, array, k):
        return array[1:k + 1]

    def update_point_color(self, chosen_point, value):
        self.df.iloc[chosen_point, -1] = value


## This will be replaced with our processed DataFrame ===
df = pd.read_csv('data/encodings.csv')
#df['dim1'] = generate_dummy_values(1, 100)
#df['dim2'] = generate_dummy_values(5, 60)
#df['dim3'] = generate_dummy_values(10, 50)

#df['dim1'] = df['enc1']
#df['dim2'] = df['enc2']
#df['dim3'] = df['enc3']

#df['color'] = ['lightskyblue'] * 10000
## ======================================================

data_v = DataVisualizer(df, K=10, index_of_chosen_point=6)
data_v.visualize()
# print(data_v.get_recommended_songs())