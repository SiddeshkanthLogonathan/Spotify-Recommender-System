import numpy as np
from sklearn.decomposition import PCA


class KNN:
    COLOR_OF_CHOSEN_POINT = 'red'
    COLOR_OF_K_NEIGHBOURS = 'mediumpurple'

    def __init__(self, dataset):
        self.dataset = dataset
        self.points = dataset.encodings_tensor

    def knn_query(self, song_id: str, k: int = 20):
        boolean_index = self.dataset.df['id'] == song_id
        index_of_chosen_point = self.dataset.df.index[boolean_index].item()
        knn_indices = self._compute_k_nearest_neighbours(index_of_chosen_point, k)

        chosen_point_low_dim, knn_points_low_dim = self._get_3d_encodings(index_of_chosen_point, knn_indices)

        chosen_song_df = self.dataset.df.iloc[index_of_chosen_point, :]
        chosen_song_df['type'] = 'input song'
        chosen_song_df['encoding_x'] = chosen_point_low_dim[0]
        chosen_song_df['encoding_y'] = chosen_point_low_dim[1]
        chosen_song_df['encoding_z'] = chosen_point_low_dim[2]

        knn_df = self.dataset.df.iloc[knn_indices, :]
        knn_df['type'] = 'recommended song'
        knn_df['encoding_x'] = knn_points_low_dim[:, 0]
        knn_df['encoding_y'] = knn_points_low_dim[:, 1]
        knn_df['encoding_z'] = knn_points_low_dim[:, 2]

        return chosen_song_df, knn_df

    def _get_3d_encodings(self, index_of_chosen_point, knn_indices):
        if self.points.shape[1] == 3:
            return self.points[index_of_chosen_point], self.points[knn_indices]

        all_indices = [index_of_chosen_point] + list(knn_indices)
        points = self.points[all_indices]
        pca = PCA(n_components=3)
        dim_reduced_points = pca.fit_transform(points)

        chosen_point_low_dim = dim_reduced_points[0]
        knn_points_low_dim = dim_reduced_points[1:]

        return chosen_point_low_dim, knn_points_low_dim

    def _compute_k_nearest_neighbours(self, index_of_chosen_point, k):
        chosen_point = self.points[index_of_chosen_point]

        dist_array = []
        for point in self.points:
            dist = np.linalg.norm(chosen_point - point) # computes the Euclidean distance between chosen point and x
            dist_array.append(dist)

        closest = np.argsort(dist_array)  # Sort it based on closest to furthest for each point

        return closest[1: k+1]
