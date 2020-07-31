# Spotify-Recommender-System
Using Deep Learning to create a Spotify Recommender System

# Installation

Just cd into the root of this repository and type
```
pip install -r requirements.txt
```

This installs [pytorch](https://pytorch.org/) along with some other libraries for data processing and visualization.

# How to run the app

In the main directory execute 

```
python main.py
```

This starts the web application for visualization. This application is accessible through the URL shown in the terminal output.

However if you want to train the model, you can use the `--train` option and spefify the number of epochs with the `--num-epochs n` option.
You can turn on verbose mode by adding `--verbose` and you can create a new dataset file by adding `--clean`.

# Data

We worked with the Spotify Song Dataset from kaggle, which can be downloaded under [this link](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks).
A copy of those files is saved in the `data/` directory.