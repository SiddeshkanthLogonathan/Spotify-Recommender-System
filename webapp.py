import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

from data_loading import SpotifyRecommenderDataset

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import torch
import numpy as np
import datetime
from visualization import KNN
from data_loading import SpotifyRecommenderDataset

dataset = SpotifyRecommenderDataset()
dataset_df = dataset.df

dataset_df['duration (minutes)'] = dataset_df['duration_ms'] / 60000
new_artist_column = [", ".join(artist_list) for artist_list in dataset_df['artists']]
dataset_df['artists'] = new_artist_column
new_genres_column = [", ".join(gemre_list) for gemre_list in dataset_df['genres']]
dataset_df['genres'] = new_genres_column
dataset_df = dataset_df[['id', 'name', 'artists', 'genres', 'duration (minutes)']]

knn = KNN()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.H1("Spotify Recommender Webapp", style={'text-align': 'center'}),

    html.H3("Enter a song id"),
    dcc.Input(id='song_id_input', value="7GhIk7Il098yCjg4BQjzvb"),
    html.Br(),

    dcc.Graph(id='3d_scatter_plot', figure={}),

    dash_table.DataTable(
        columns=[{'name': column_name,
                  'id': column_name}
                 for column_name in dataset_df],
        data=dataset_df.to_dict('records'),
        filter_action='native',
        style_table={
            'height': 400,
        },
        style_data={
            'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        }
    )
])
@app.callback(
    # [Output(component_id='output_container', component_property='children'),
    Output(component_id='3d_scatter_plot', component_property='figure'),
    [Input(component_id='song_id_input', component_property='value')]
)
def update_graph(song_id):

    chosen_song_df, knn_df = knn.knn_query(song_id)



    fig = px.scatter_3d(knn_df, x='encoding_x', y='encoding_y', z='encoding_z', hover_name="name",
                        hover_data=knn_df.columns, color='color', symbol='symbol')

    return fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
