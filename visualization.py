import random
import plotly.express as plt
import pandas as pd

## This is used to simulate our 3d Dataset that we will use to visualize. It is unnecessary once we have our data
def generate_dummy_values(low, high):
    coordinates = []
    for i in range(100):
        coordinates.append(random.randint(low, high))
    return coordinates

class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def visualize(self):
        cols = self.df.columns
        plt.scatter_3d(self.df, x=cols[0], y=cols[1], z=cols[2]).show()

## This will be replaced with our processed DataFrame ===
df = pd.DataFrame()
df['dim1'] = generate_dummy_values(1, 100)
df['dim2'] = generate_dummy_values(5, 60)
df['dim3'] = generate_dummy_values(10, 50)
## ======================================================

DataVisualizer(df).visualize()