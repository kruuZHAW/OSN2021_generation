# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31

@author: kruu

Plot script for go-arounds generation trajectories visualisation
"""

import os
import pandas as pd
import numpy as np
import scipy as  sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist

#Generated data
X_raw = pd.read_pickle("Data/X_raw.pkl")
Real = pd.read_pickle("Data/distributions_along_lines.pkl")
Vines_samp = pd.read_pickle("Data/generated_vines_and_sampling.pkl")
Mvn_samp = pd.read_pickle("Data/generated_MVN_and_sampling.pkl")
Gm_samp = pd.read_pickle("Data/generated_GM_and_sampling.pkl")
Mvn_Wsamp = pd.read_pickle("Data/generated_MVN_without_sampling.pkl")
Gm_Wsamp = pd.read_pickle("Data/generated_GM_without_sampling.pkl")
Vines_Wsamp = pd.read_pickle("Data/generated_vines_without_sampling.pkl")


#Perpendiculars data
perpendiculars = pd.read_parquet("Data/Normals_sampling.parquet")
perpendiculars["angle"] = np.arctan(perpendiculars.m)

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the mean of the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def energy_distance(x, y):
    """Compute energy distance between the distributions of 2 samples of different length
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    a = cdist(x, y, "euclidean").mean()
    b = cdist(x, x, "euclidean").mean()
    c = cdist(y, y, "euclidean").mean()
    e = (n1 * n2 / (n1 + n2)) * (2 * a - b - c)
    return e

# Signed curvature
def curvature(x, y):
    x_t = np.gradient(x)
    y_t = np.gradient(y)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    return (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** (3 / 2)

def reconstruct_trajs(df, sampled = True):
    n_gen = len(df)

    if sampled : 
        gen_x = np.array(df) * np.array([np.cos(perpendiculars.angle.values),] * n_gen) + np.array(
        [perpendiculars.x.values,] * n_gen
        )

        calc_y = perpendiculars.m.values * gen_x + perpendiculars.p.values
    
    else : 
        gen_x = np.array(df.iloc[:,::2])
        calc_y = np.array(df.iloc[:,1::2])

    return gen_x, calc_y

def fit_spline(df, name_png, sampled = True, n_display = 400):
    rand = np.random.choice(range(len(df)), n_display, replace=False)
    gen_x, calc_y = reconstruct_trajs(df.loc[rand], sampled = sampled)
    if sampled:
        n_features = df.shape[1]
    else :
        n_features = df.shape[1]//2

    x = list(range(0, n_features))
    spline_x = []
    spline_y = []

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Spline fitting")

    for i in range(n_display):
        col = colors[i % len(colors)]
        y = np.vstack((gen_x[i], calc_y[i])).T
        cs = CubicSpline(x, y)
        xs = np.linspace(0, 29, 1000)
        spline_x.append(cs(xs)[:, 0])
        spline_y.append(cs(xs)[:, 1])

        if i % 40 == 0:
            ax.plot(y[:, 0], y[:, 1], "o", ms = 2, label="data", color=col)
            ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label="spline", color=col)
        else : 
            ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label="spline", color="grey", alpha = 0.2)
            
    plot_name = 'Plots/splines_' + name_png + '.png'
    plt.savefig(plot_name)
    plt.close()
    return spline_x, spline_y

def mean_turns(spline_x, spline_y, threshold = 0.00015):
    nb_turns = []
    prop_turns = []
    for i in range(len(spline_x)):
        indexes1 = np.where(curvature(spline_x[i], spline_y[i]) > threshold)
        indexes2 = np.where(curvature(spline_x[i], spline_y[i]) < -threshold)
        nb_turns.append(
            np.sum(np.diff(indexes1[0], 1) != 1)
            + 1
            + np.sum(np.diff(indexes2[0], 1) != 1)
            + 1
        )
        prop_turns.append(len(indexes1[0])*0.001 + len(indexes2[0])*0.001)



    return np.array(nb_turns).mean(), np.array( prop_turns).mean()

def plot_raw_trajs(df, name_png, sampled = True, n_display = 400):
    rand = np.random.choice(range(len(df)), n_display, replace=False)
    gen_x, calc_y = reconstruct_trajs(df.loc[rand], sampled = sampled)

    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.plot(gen_x.T)
    plt.title("Generated x")

    plt.subplot(222)
    plt.plot(calc_y.T)
    plt.title("Calculated y")

    plt.subplot(223)
    plt.title("Generated trajectory")
    plt.plot(
        gen_x.T, calc_y.T,
    )

    plt.subplot(224)
    plt.title("Generated points")
    plt.scatter(
        gen_x.T, calc_y.T,
    )
    plt.scatter(perpendiculars.x.values, perpendiculars.y.values, color="r")

    plot_name = 'Plots/Raw_trajectories_' + name_png + '.png'
    plt.savefig(plot_name)
    plt.close()
    return None

#Plot turns of 1 trajectory
def plot_turns(spline_x, spline_y, name_png, i, threshold = 0.00015):

    plt.figure(figsize=(15, 10))

    plt.subplot(221)
    plt.plot(curvature(spline_x[i], spline_y[i]))
    indexes1 = np.where(curvature(spline_x[i], spline_y[i]) > threshold)
    indexes2 = np.where(curvature(spline_x[i], spline_y[i]) < -threshold)
    plt.hlines(threshold, 0, 1000, color="r")
    plt.hlines(-threshold, 0, 1000, color="r")
    plt.title("Signed Curvature of the trajectoy")

    plt.subplot(222)
    plt.plot(spline_x[i], spline_y[i])
    plt.scatter(spline_x[i][indexes1], spline_y[i][indexes1], color="r", label = "Positive Curvature")
    plt.scatter(spline_x[i][indexes2], spline_y[i][indexes2], color="g", label = "Negative Curvature")
    plt.legend()
    plt.title("High Curvature parts of one trajectory")

    plt.subplot(223)
    curv_var = np.gradient(curvature(spline_x[i], spline_y[i]))
    plt.plot(curv_var)
    plt.title("Curvature variation")

    plot_name = 'Plots/Turns_' + name_png + '.png'
    plt.savefig(plot_name)
    plt.close()
    return None

#Plot generation
display = 400
threshold = 0.00010

plot_raw_trajs(Real, "Real", sampled = True, n_display = display)
spline_x, spline_y = fit_spline(Real, "Real", sampled = True, n_display = display)
plot_turns(spline_x, spline_y, "Real", 10, threshold = threshold)
print("Real :", mean_turns(spline_x, spline_y, threshold = threshold), "\n")


plot_raw_trajs(Vines_samp, "Vines_samp", sampled = True, n_display = display)
spline_x, spline_y = fit_spline(Vines_samp, "Vines_samp", sampled = True, n_display = display)
plot_turns(spline_x, spline_y, "Vines_samp", 10, threshold = threshold)
print("Vines and Samp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean:", np.mean(mahalanobis(x=Vines_samp, data=Real)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Vines_samp), np.array(Real), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Vines_samp), np.array(Real)), "\n")

plot_raw_trajs(Gm_samp, "Gm_samp", sampled = True, n_display = display)
spline_x, spline_y = fit_spline(Gm_samp, "Gm_samp", sampled = True, n_display = display)
plot_turns(spline_x, spline_y, "Gm_samp", 10, threshold = threshold)
print("GM and Samp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean :", np.mean(mahalanobis(x=Gm_samp, data=Real)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Gm_samp), np.array(Real), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Gm_samp), np.array(Real)), "\n")

plot_raw_trajs(Mvn_samp, "Mvn_samp", sampled = True, n_display = display)
spline_x, spline_y = fit_spline(Mvn_samp, "Mvn_samp", sampled = True, n_display = display)
plot_turns(spline_x, spline_y, "Mvn_samp", 10, threshold = threshold)
print("MVN and Samp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean :", np.mean(mahalanobis(x=Mvn_samp, data=Real)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Mvn_samp), np.array(Real), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Mvn_samp), np.array(Real)), "\n")

plot_raw_trajs(Vines_Wsamp, "Vines_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Vines_Wsamp, "Vines_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Vines_Wsamp", 10, threshold = threshold)
print("Vines without Samp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean :", np.mean(mahalanobis(x=Vines_Wsamp, data=X_raw)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Vines_Wsamp), np.array(X_raw), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Vines_Wsamp), np.array(X_raw)), "\n")

plot_raw_trajs(Gm_Wsamp, "Gm_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Gm_Wsamp, "Gm_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Gm_Wsamp", 10, threshold = threshold)
print("Gm_Wsamp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean:", np.mean(mahalanobis(x=Gm_Wsamp, data=X_raw)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Gm_Wsamp), np.array(X_raw), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Gm_Wsamp), np.array(X_raw)), "\n")

plot_raw_trajs(Mvn_Wsamp, "Mvn_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Mvn_Wsamp, "Mvn_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Mvn_Wsamp", 10, threshold = threshold)
print("Mvn_Wsamp :", mean_turns(spline_x, spline_y, threshold = threshold))
print("Mahalanobis distance to mean :", np.mean(mahalanobis(x=Mvn_Wsamp, data=X_raw)))
print("Mean Mahalanobis distance to every real trajectories:", cdist(np.array(Mvn_Wsamp), np.array(X_raw), "mahalanobis").mean())
print("Energy distance : ", energy_distance(np.array(Mvn_Wsamp), np.array(X_raw)), "\n")