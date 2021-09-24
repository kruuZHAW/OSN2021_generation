# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 / 2021

@author: kruu

Generate plots for the dimension reduction method 
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from traffic.core import Traffic, Flight
import pyproj
import warnings
warnings.filterwarnings("ignore")

# Data processed
X_along_lines = pd.read_pickle("Data/distributions_along_lines.pkl")
perpendiculars = pd.read_parquet("Data/Normals_sampling.parquet")

#traffic objects
t_clean = Traffic.from_file("Data/Go_Arounds_clean.parquet")
t_resampled = Traffic.from_file("Data/GA_resampled.parquet")

def plot_projection(t_resampled, Projs):
    features = ["x", "y"]
    X = np.stack(list(f.data[features].values.ravel() for f in t_resampled))
    x, y, z = (
    np.array(X)[:, ::2].flatten(),
    np.array(X)[:, 1::2].flatten(),
    np.array(Projs).flatten(),
    )
    cmap = sns.color_palette("viridis", as_cmap=True)
    f, ax = plt.subplots(figsize=(15, 10))
    points = ax.scatter(x, y, c=z, s=10, cmap=cmap)
    f.colorbar(points)

    plot_name = 'Plots/projections.png'
    plt.savefig(plot_name)
    plt.close()
    return None

def plot_traj_reduced(t_clean, t_resampled, perpendiculars, i_ref = 97):
    swiss = pyproj.Proj(init="EPSG:21781")

    x_centroid, y_centroid = (
    t_clean[i_ref].compute_xy(swiss).data.x.values,
    t_clean[i_ref].compute_xy(swiss).data.y.values,
    )

    sel = np.random.randint(len(t_resampled))
    x = t_clean[sel].compute_xy(swiss).data.x.values
    y = t_clean[sel].compute_xy(swiss).data.y.values
    x_sampled = t_resampled[sel].data.x.values
    y_sampled = t_resampled[sel].data.y.values

    plt.plot(x_centroid.T, y_centroid.T)
    plt.plot(x_sampled, y_sampled, marker=".")
    plt.plot(x, y)
    plt.legend(["Reference trajectory", "Sampled trajectory", "True trajectory"])

    for i in range(len(perpendiculars)):
        y1 = perpendiculars.m[i] * (perpendiculars.x[i] - 4000) + perpendiculars.p[i]
        y2 = perpendiculars.m[i] * (perpendiculars.x[i] + 4000) + perpendiculars.p[i]

        plt.plot(
            [(perpendiculars.x[i] - 4000), (perpendiculars.x[i] + 4000)], [y1, y2], color="black", linestyle="--",
        )
    plt.xlim(x_centroid.min() - 12000, x_centroid.max() + 12000)
    plt.ylim(y_centroid.min() - 12000, y_centroid.max() + 12000)

    plot_name = 'Plots/resampling.png'
    plt.savefig(plot_name)
    plt.close()
    return None 


plot_projection(t_resampled, X_along_lines)
plot_traj_reduced(t_clean, t_resampled, perpendiculars, i_ref = 97)

