import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

from data_loading import SpotifyRecommenderDataset
import visualization
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import torch
import numpy as np
from visualization import KNN
from data_loading import SpotifyRecommenderDataset

class WebApp:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self, dataset: SpotifyRecommenderDataset, knn: visualization.KNN ):
        self.dataset = dataset
        self.dataset_df = self.dataset.df
        self.dataset_df['duration (minutes)'] = self.dataset_df['duration_ms'] / 60000
        new_artist_column = [", ".join(artist_list) for artist_list in self.dataset_df['artists']]
        self.dataset_df['artists'] = new_artist_column
        new_genres_column = [", ".join(genre_list) for genre_list in self.dataset_df['genres']]
        self.dataset_df['genres'] = new_genres_column
        self.dataset_df = self.dataset_df[['id', 'name', 'artists', 'genres', 'popularity']]

        self.knn = knn

        self.app = dash.Dash(__name__, external_stylesheets=self.external_stylesheets)
        self._setup_dash_app()

    def _setup_dash_app(self):
        self.app.layout = html.Div([
            html.H1("Spotify Recommender Webapp", style={'text-align': 'center'}),

            html.Div([
                html.H3("How many recommendations"),
                dcc.Input(id='value_for_k', value="20"),
                html.Br()
            ], style={'display': 'inline-block', 'margin-bottom': '2em'}),

            html.Div([
                html.Div([
                    dcc.Graph(id='3d_scatter_plot', figure={},
                              style={
                                  'height': '600px',
                              })
                ], style={
                        'marginRight': '1em',
                        'width': '100%',
                        'min-width': 400,
                    }),

                html.Div([
                    html.H3("Recommended songs"),
                    dash_table.DataTable(
                        id='recommendations_table',
                        columns=[{'name': column_name,
                                  'id': column_name}
                                 for column_name in self.dataset_df.loc[:, ['name', 'artists']]],
                        data=self.dataset_df.to_dict('records'),
                        filter_action='native',
                        style_table={
                            'max-height': 530,
                            'overflow-y': 'scroll',
                        },
                        style_data={
                            'width': '100px', 'minWidth': '100px', 'maxWidth': '100px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                        }
                    )
                ], style={
                        'display': 'flex',
                        'flex-flow': 'column nowrap',
                        'width': '100%',
                        'min-width': 400,
                        'max-width': 600,
                        'padding': '1em',
                        'box-shadow': '0px 2px 4px rgba(0,0,0,0.18)',
                        'border-radius': 6,
                        'margin-bottom': '2em',
                    }),

            ], style={'display': 'flex', 'flex-flow': 'row nowrap', 'justify-content': 'space-between'}),

            dash_table.DataTable(
                id='songs_table',
                columns=[{'name': column_name,
                          'id': column_name}
                         for column_name in self.dataset_df.loc[:, ['name', 'artists', 'genres', 'popularity']]],
                data=self.dataset_df.to_dict('records'),
                filter_action='native',
                row_selectable='single',
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
            [Output(component_id='3d_scatter_plot', component_property='figure'),
            Output(component_id='recommendations_table', component_property='data')],
            [Input(component_id='songs_table', component_property='selected_rows'),
             Input(component_id='value_for_k', component_property='value')]
        )
        def update_graph(song_id, k):
            if song_id is None:
                song_id = [84645]

            song_name = self.dataset_df.loc[song_id[0], 'name']
            song_id = self.dataset_df.loc[song_id[0], 'id']

            chosen_song_df, knn_df = self.knn.knn_query(song_id, int(k))
            chosen_song_and_knn_df = pd.concat([knn_df, pd.DataFrame(chosen_song_df).transpose()])
            song_recommendations_df = knn_df[['name', 'artists']]
            print(song_recommendations_df)
            fig = px.scatter_3d(chosen_song_and_knn_df, x='encoding_x', y='encoding_y', z='encoding_z', hover_name='name',
                                hover_data=chosen_song_and_knn_df.loc[:, ['artists']], color='type')
            fig.update_layout(showlegend=False, title=song_name,
                              font=dict(
                                    family="Courier New, monospace",
                                    size=18
                                )
                              )

            return fig, song_recommendations_df.to_dict('records')

    def run(self, debug=False):
        self.app.run_server(debug=debug)