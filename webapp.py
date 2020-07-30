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

class WebApp:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self):
        self.dataset = SpotifyRecommenderDataset()
        self.dataset_df = self.dataset.df

        self.dataset_df['duration (minutes)'] = self.dataset_df['duration_ms'] / 60000
        new_artist_column = [", ".join(artist_list) for artist_list in self.dataset_df['artists']]
        self.dataset_df['artists'] = new_artist_column
        new_genres_column = [", ".join(genre_list) for genre_list in self.dataset_df['genres']]
        self.dataset_df['genres'] = new_genres_column
        self.dataset_df = self.dataset_df[['id', 'name', 'artists', 'genres', 'duration (minutes)']]

        self.knn = KNN()

        self.app = dash.Dash(__name__, external_stylesheets=self.external_stylesheets)
        self._setup_dash_app()

    def _setup_dash_app(self):
        self.app.layout = html.Div([
            html.H1("Spotify Recommender Webapp", style={'text-align': 'center'}),

            html.H3("Enter a song id"),
            dcc.Input(id='song_id_input', value="7GhIk7Il098yCjg4BQjzvb"),
            html.Br(),

            dcc.Graph(id='3d_scatter_plot', figure={}),

            dash_table.DataTable(
                columns=[{'name': column_name,
                          'id': column_name}
                         for column_name in self.dataset_df],
                data=self.dataset_df.to_dict('records'),
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
        @self.app.callback(
            # [Output(component_id='output_container', component_property='children'),
            Output(component_id='3d_scatter_plot', component_property='figure'),
            [Input(component_id='song_id_input', component_property='value')]
        )
        def update_graph(song_id):
            chosen_song_df, knn_df = self.knn.knn_query(song_id)

            chosen_song_and_knn_df = pd.concat([pd.DataFrame(chosen_song_df).transpose(), knn_df])
            print(pd.DataFrame(chosen_song_df).transpose())
            print(knn_df)

            fig = px.scatter_3d(chosen_song_and_knn_df, x='encoding_x', y='encoding_y', z='encoding_z', hover_name='name',
                                hover_data=chosen_song_and_knn_df.columns, color='color', symbol='symbol')

            return fig

    def run(self, debug=False):
        self.app.run_server(debug=debug)