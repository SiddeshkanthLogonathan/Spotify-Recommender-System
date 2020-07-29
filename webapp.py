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

raw_dataset = pd.read_csv('data/data.csv')
raw_dataset = raw_dataset[['name', 'artists', 'release_date', 'duration_ms', 'popularity']]

knn = KNN()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.H1("Spotify Recommender Webapp", style={'text-align': 'center'}),

    html.H3("Enter a song name"),
    dcc.Input(id='song_name_input', value="Fantasiestücke, Op. 111: Più tosto lento"),
    html.H3("Enter list of artists"),
    dcc.Input(id='artists_input', value="['Robert Schumann', 'Vladimir Horowitz']"),
    html.Br(),

    dcc.Graph(id='3d_scatter_plot', figure={}),

    dash_table.DataTable(
        columns=[{'name': column_name,
                  'id': column_name,
                  'type': ('text' if column_name in ['name', 'artists'] else 'numeric')}
                 for column_name in raw_dataset.columns],
        data=raw_dataset.to_dict('records'),
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


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    # [Output(component_id='output_container', component_property='children'),
    Output(component_id='3d_scatter_plot', component_property='figure'),
    [Input(component_id='song_name_input', component_property='value'),
     Input(component_id='artists_input', component_property='value')]
)
def update_graph(song_name, artists_string):
    print(song_name)
    print(artists_string)

    df = knn.compute_k_nearest_neighbours(4269, 10)

    fig = px.scatter_3d(df, x='encoding_x_column', y='encoding_y_column', z='encoding_z_column', hover_name="name",
                        hover_data=df.columns, color='color_column', symbol='symbol_column')

    return fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
