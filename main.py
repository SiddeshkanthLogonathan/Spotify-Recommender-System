from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader, GenreOccurenceDataset
from architecture import Autoencoder, GenreEmbedder
from train import train_autoencoder, train_genre_embedder
from model_testing import test
from webapp import WebApp
import torch
from visualization import GeneralPurposeVisualizer
import pandas as pd
import plotly.express as px

"""
dataset = SpotifyRecommenderDataset()
dataloader = SpotifyRecommenderDataLoader(dataset, batch_size = 256, shuffle=True)
model = Autoencoder()
model = train_autoencoder(model, dataloader)
torch.save(model, 'model.pth')

loss = test(model)
print("Loss: ", loss)

dataloader = SpotifyRecommenderDataLoader(dataset, batch_size=len(dataset))
batch = next(iter(dataloader))

with torch.no_grad():
    encodings = model.encode(batch.training_label)
dataset.add_encoding_columns(encodings)
#GeneralPurposeVisualizer.visualize_encodings(encodings)

webapp = WebApp()
webapp.run()
"""

genre_embedder = GenreEmbedder()
genre_occurrence_dataset = GenreOccurenceDataset()
train_dataloader = torch.utils.data.DataLoader(genre_occurrence_dataset, batch_size = 256, shuffle=True)

genre_embedder = train_genre_embedder(genre_embedder, train_dataloader)

eval_input = torch.arange(0, len(genre_occurrence_dataset.distinct_genres), dtype=torch.long)
with torch.no_grad():
    embeddings = genre_embedder(eval_input)

df = pd.DataFrame({
    'genre_name': genre_occurrence_dataset.distinct_genres,
    'x': embeddings[:, 0],
    'y': embeddings[:, 1],
    'z': embeddings[:, 2]
})

fig = px.scatter_3d(df, x='x', y='y', z='z', color='genre_name')
fig.show()

a = 10