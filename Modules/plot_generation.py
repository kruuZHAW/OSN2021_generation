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
import matplotlib as mpl
import seaborn as sns
from matplotlib.collections import LineCollection
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

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Spline fitting")

    for i in range(n_display):
        y = np.vstack((gen_x[i], calc_y[i])).T
        cs = CubicSpline(x, y)
        xs = np.linspace(0, 29, 1000)
        spline_x.append(cs(xs)[:, 0])
        spline_y.append(cs(xs)[:, 1])

        if i % 40 == 0:
            ax.plot(y[:, 0], y[:, 1], "o", ms = 2, label="data", color=[1,0.65,0])
            ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label="spline", color=[1,0.65,0])
        else : 
            ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label="spline", color= [0.80,0.80,0.80], alpha = 0.2)
            
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

    return np.array(nb_turns).mean()#, np.array( prop_turns).mean()

def plot_raw_trajs(df, name_png, sampled = True, n_display = 400):
    rand = np.random.choice(range(len(df)), n_display, replace=False)
    gen_x, calc_y = reconstruct_trajs(df.loc[rand], sampled = sampled)

    alphas = np.ones(n_display)*0.1
    idx = np.random.choice(n_display, 5, replace=False)
    alphas[idx] = 1
    rgba_colors = np.zeros((n_display,4))
    rgba_colors[:,0] = 0.8
    rgba_colors[:,1] = 0.8
    rgba_colors[:,2] = 0.8
    rgba_colors[:,3] = alphas
    rgba_colors[idx,:3] = [1,0.65,0]
    xaxis = np.array([np.arange(gen_x.shape[1]),] * gen_x.shape[0])
    rgba_scatter = np.ravel([rgba_colors]*gen_x.shape[1], order="F").reshape((4, n_display*gen_x.shape[1])).T

    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    #plt.plot(gen_x.T)
    segs = np.stack((xaxis, gen_x), axis=2)
    lc = LineCollection(segs, colors=rgba_colors)
    plt.gca().add_collection(lc)
    plt.autoscale()
    plt.title("Generated x")

    plt.subplot(222)
    #plt.plot(calc_y.T)
    segs = np.stack((xaxis, calc_y), axis=2)
    lc = LineCollection(segs, colors=rgba_colors)
    plt.gca().add_collection(lc)
    plt.autoscale()
    plt.title("Calculated y")

    plt.subplot(223)
    plt.title("Generated trajectory")
    segs = np.stack((gen_x, calc_y), axis=2)
    lc = LineCollection(segs, colors=rgba_colors)
    plt.gca().add_collection(lc)
    plt.autoscale()
    #plt.plot(
    #    gen_x.T, calc_y.T,
    #)

    plt.subplot(224)
    plt.title("Generated points")
    plt.scatter(
        gen_x, calc_y, color =  rgba_scatter)
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
i = np.random.randint(display)
threshold = 0.00010
metrics = {}

plot_raw_trajs(Real, "Real", sampled = True, n_display = display)
x,y = reconstruct_trajs(Real)
spline_x, spline_y = fit_spline(Real, "Real", sampled = True, n_display = display)
plot_turns(x, y, "Real", i, threshold = threshold)
metrics["Real"] = [mean_turns(spline_x, spline_y, threshold = threshold), 0, 0, 0]
print("Real :", metrics["Real"][0], "\n")


plot_raw_trajs(Vines_samp, "Vines_samp", sampled = True, n_display = display)
x,y = reconstruct_trajs(Vines_samp)
spline_x, spline_y = fit_spline(Vines_samp, "Vines_samp", sampled = True, n_display = display)
plot_turns(x, y, "Vines_samp", i, threshold = threshold)
metrics["Vines_samp"] = [mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Vines_samp, data=Real)), cdist(np.array(Vines_samp), np.array(Real), "mahalanobis").mean(), energy_distance(np.array(Vines_samp), np.array(Real))]
print("Vines and Samp :",metrics["Vines_samp"][0])
print("Mahalanobis distance to mean:",metrics["Vines_samp"][1])
print("Mean Mahalanobis distance to every real trajectories:", metrics["Vines_samp"][2])
print("Energy distance : ", metrics["Vines_samp"][3], "\n")

plot_raw_trajs(Gm_samp, "Gm_samp", sampled = True, n_display = display)
x,y = reconstruct_trajs(Gm_samp)
spline_x, spline_y = fit_spline(Gm_samp, "Gm_samp", sampled = True, n_display = display)
plot_turns(x, y, "Gm_samp", i, threshold = threshold)
metrics["Gm_samp"] = [mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Gm_samp, data=Real)), cdist(np.array(Gm_samp), np.array(Real), "mahalanobis").mean(), energy_distance(np.array(Gm_samp), np.array(Real))]
print("GM and Samp :",  metrics["Gm_samp"][0])
print("Mahalanobis distance to mean :", metrics["Gm_samp"][1])
print("Mean Mahalanobis distance to every real trajectories:",metrics["Gm_samp"][2])
print("Energy distance : ", metrics["Gm_samp"][3], "\n")

plot_raw_trajs(Mvn_samp, "Mvn_samp", sampled = True, n_display = display)
x,y = reconstruct_trajs(Mvn_samp)
spline_x, spline_y = fit_spline(Mvn_samp, "Mvn_samp", sampled = True, n_display = display)
plot_turns(x, y, "Mvn_samp", i, threshold = threshold)
metrics["Mvn_samp"] = [ mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Mvn_samp, data=Real)), cdist(np.array(Mvn_samp), np.array(Real), "mahalanobis").mean(), energy_distance(np.array(Mvn_samp), np.array(Real))]
print("MVN and Samp :", metrics["Mvn_samp"][0])
print("Mahalanobis distance to mean :", metrics["Mvn_samp"][1])
print("Mean Mahalanobis distance to every real trajectories:", metrics["Mvn_samp"][2])
print("Energy distance : ", metrics["Mvn_samp"][3], "\n")

plot_raw_trajs(Vines_Wsamp, "Vines_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Vines_Wsamp, "Vines_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Vines_Wsamp", i, threshold = threshold)
metrics["Vines_Wsamp"] = [mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Vines_Wsamp, data=X_raw)), cdist(np.array(Vines_Wsamp), np.array(X_raw), "mahalanobis").mean(), energy_distance(np.array(Vines_Wsamp), np.array(X_raw))]
print("Vines without Samp :", metrics["Vines_Wsamp"][0])
print("Mahalanobis distance to mean :", metrics["Vines_Wsamp"][1])
print("Mean Mahalanobis distance to every real trajectories:", metrics["Vines_Wsamp"][2])
print("Energy distance : ", metrics["Vines_Wsamp"][3], "\n")

plot_raw_trajs(Gm_Wsamp, "Gm_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Gm_Wsamp, "Gm_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Gm_Wsamp", i, threshold = threshold)
metrics["Gm_Wsamp"] = [mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Gm_Wsamp, data=X_raw)), cdist(np.array(Gm_Wsamp), np.array(X_raw), "mahalanobis").mean(), energy_distance(np.array(Gm_Wsamp), np.array(X_raw))]
print("Gm_Wsamp :", metrics["Gm_Wsamp"][0])
print("Mahalanobis distance to mean:",metrics["Gm_Wsamp"][1] )
print("Mean Mahalanobis distance to every real trajectories:", metrics["Gm_Wsamp"][2])
print("Energy distance : ",metrics["Gm_Wsamp"][3] , "\n")

plot_raw_trajs(Mvn_Wsamp, "Mvn_Wsamp", sampled = False, n_display = display)
spline_x, spline_y = fit_spline(Mvn_Wsamp, "Mvn_Wsamp", sampled = False, n_display = display)
plot_turns(spline_x, spline_y, "Mvn_Wsamp", i, threshold = threshold)
metrics["Mvn_Wsamp"] = [mean_turns(spline_x, spline_y, threshold = threshold), np.mean(mahalanobis(x=Mvn_Wsamp, data=X_raw)), cdist(np.array(Mvn_Wsamp), np.array(X_raw), "mahalanobis").mean(), energy_distance(np.array(Mvn_Wsamp), np.array(X_raw))]
print("Mvn_Wsamp :", metrics["Mvn_Wsamp"][0])
print("Mahalanobis distance to mean :", metrics["Mvn_Wsamp"][1])
print("Mean Mahalanobis distance to every real trajectories:", metrics["Mvn_Wsamp"][2])
print("Energy distance : ", metrics["Mvn_Wsamp"][3], "\n")
