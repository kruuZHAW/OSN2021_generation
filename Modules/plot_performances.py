# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 / 2021

@author: kruu

Evaluate goodness of fit for different generation methods 
"""

import os
import pandas as pd
import numpy as np
import scipy as  sp
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm.autonotebook import tqdm
from sklearn.mixture import GaussianMixture
import openturns as ot
import pyvinecopulib as pv

from traffic.core import Traffic

#Trajectories projected
X_along_lines = pd.read_pickle("Data/distributions_along_lines.pkl")

#trajectories not resampled
X_not_sampled = pd.read_pickle("Data/X_raw.pkl")

# traffic resampled
t_resampled = Traffic.from_file("Data/GA_resampled.parquet")
features = ["x", "y"]
X_sampled = pd.DataFrame(np.stack(list(f.data[features].values.ravel() for f in t_resampled)))

# Normals for sampling
perpendiculars = pd.read_parquet("Data/Normals_sampling.parquet")
perpendiculars["angle"] = np.arctan(perpendiculars.m)

#re-run the generation or not
simulation = True

def inverse_sampling(df):
    n_gen = df.shape[0]
    n_feat = df.shape[1]
    res = np.zeros((n_gen, n_feat*2))
    res[:,0::2] = np.array(df) * np.array([np.cos(perpendiculars.angle.values),] * n_gen) + np.array(
    [perpendiculars.x.values,] * n_gen
    )
    res[:,1::2] = perpendiculars.m.values * res[:,0::2] + perpendiculars.p.values
    return res

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

def plot_metrics(metrics, names = ["Energy", "Mahala_to_mean", "Mean_mahala"] ):

    #Normalization
    a = pd.DataFrame(metrics)
    b = a.div(a.max(axis=1), axis=0)
    norm = b.to_dict(orient="list")

    plt.figure(figsize=(12, 8))

    # set width of bars
    barWidth = 1 / (len(metrics) + 1)

    # Set position of bar on X axis
    r1 = np.arange(len(names))

    #colors
    col = ["lightsalmon", "coral", "lightgreen", "darkseagreen", "lightskyblue", "steelblue"]

    # Make the plot
    for i, key in enumerate(norm) : 
        plt.bar(r1 + barWidth*i, norm[key], width=barWidth, edgecolor="white", color = col[i], label=key)

    # Add xticks on the middle of the group bars
    plt.xlabel("Metric", fontweight="bold")
    plt.xticks(
        [r + 0.5-barWidth for r in range(len(names))], names
    )

    # Create legend & Show graphic
    plt.legend()
    plt.title("Metrics")

    plot_name = 'Plots/metrics.png'
    plt.savefig(plot_name)
    plt.close()
    return None

def MVN_perf(data, data_eval, n_sim, n_sample, reduced = False):
    """
    Evaluate performances metrics for MVN generation over n_sim generations
    of n_sample samples
    """
    energies = []
    mahala_to_mean = []
    mean_mahala = []

    for _ in tqdm(range(n_sim), leave=False):

        x_sim = np.random.multivariate_normal(np.mean(data), np.cov(data.T), n_sample)

        if reduced :
            x_sim = inverse_sampling(x_sim)     
            
        energies.append(energy_distance(x_sim, np.array(data_eval)))
        mean_mahala.append(cdist(x_sim, np.array(data_eval), "mahalanobis").mean())
        mahala_to_mean.append(np.mean(mahalanobis(x=pd.DataFrame(x_sim), data=data_eval)))
    
    return [np.mean(energies), np.mean(mahala_to_mean), np.mean(mean_mahala)]

def GM_perf(data,  data_eval, n_components, n_sim, n_sample, reduced = False):
    """
    Evaluate performances metrics for Gaussian Mixture generation over n_sim generations
    of n_sample samples
    """
    energies = []
    mahala_to_mean = []
    mean_mahala = []
    gm = GaussianMixture(n_components=n_components).fit(data)

    for _ in tqdm(range(n_sim), leave=False):
        x_sim = gm.sample(n_sample)[0]

        if reduced : 
            x_sim = inverse_sampling(x_sim)

        energies.append(energy_distance(x_sim, np.array(data_eval)))
        mean_mahala.append(cdist(x_sim, np.array(data_eval), "mahalanobis").mean())
        mahala_to_mean.append(np.mean(mahalanobis(x=pd.DataFrame(x_sim), data=data_eval)))
    
    return [np.mean(energies), np.mean(mahala_to_mean), np.mean(mean_mahala)]

def Vine_perf(data, data_eval, mat, n_sim, n_sample, reduced = False):
    """
    Evaluate performances metrics for MVN generation over n_sim generations
    of n_sample samples
    """
    energies = []
    mahala_to_mean = []
    mean_mahala = []

    #We give the D-vine strucutre (Suitable for our trajectory case)
    n_feat = data.shape[1]
    u = pv.to_pseudo_obs(data)  # CDR empirique (c'est mieux)

    if mat == True : 
        mat_vine = np.zeros((n_feat, n_feat))
        mat_vine[:(n_feat-1), 0] = list(range(2, n_feat+1))

        for i in range(1, n_feat):
            mat_vine[:(n_feat-1), i] = mat_vine[:(n_feat-1), i - 1] + 1

        mat_vine[mat_vine > n_feat] = 0

        for i in range(0, n_feat):
            mat_vine[i, (n_feat-1) - i] = n_feat - i
        
        print("Fitting copula ...")
        cop = pv.Vinecop(data=u, matrix=mat_vine)

    else:
        print("Fitting copula ...")
        cop = pv.Vinecop(data=u)

    #Vine copula fitting
    

    for _ in tqdm(range(n_sim), leave =False):
        u_sim = cop.simulate(n_sample)
        x_sim = np.asarray(
            [np.quantile(data.iloc[:, i], u_sim[:, i]) for i in range(0, data.shape[1])]
        ).T

        if reduced :
            x_sim = inverse_sampling(x_sim)

        energies.append(energy_distance(x_sim, np.array(data_eval)))
        mean_mahala.append(cdist(x_sim, np.array(data_eval), "mahalanobis").mean())
        mahala_to_mean.append(np.mean(mahalanobis(x=pd.DataFrame(x_sim), data=data_eval)))
    
    return [np.mean(energies), np.mean(mahala_to_mean), np.mean(mean_mahala)]

if simulation : 
    n_sim = 100
    n_gen = 1000
    metrics = {}

    metrics["MVN_raw"] = MVN_perf(X_not_sampled, X_not_sampled, n_sim, n_gen)
    print("Performances MVN without Sampling : ", metrics["MVN_raw"])
    metrics["MVN_reduced"] = MVN_perf(X_along_lines, X_sampled, n_sim, n_gen, reduced = True)
    print("Performances MVN with Sampling : ", metrics["MVN_reduced"], "\n")

    metrics["GM_raw"] = GM_perf(X_not_sampled, X_not_sampled, 40, n_sim, n_gen)
    print("Performances GM without Sampling : ", metrics["GM_raw"])
    metrics["GM_reduced"] = GM_perf(X_along_lines, X_sampled, 20, n_sim, n_gen, reduced = True)
    print("Performances GM with Sampling : ", metrics["GM_reduced"], "\n")

    metrics["Vines_raw"] = Vine_perf(X_not_sampled, X_not_sampled, True, n_sim, n_gen)
    print("Performances Vines without Sampling : ", metrics["Vines_raw"])
    metrics["Vines_reduced"] = Vine_perf(X_along_lines, X_sampled, True, n_sim, n_gen, reduced = True)
    print("Performances Vines with Sampling : ", metrics["Vines_reduced"], "\n")

    json.dump( metrics, open( "Data/metrics.json", 'w' ) )
    plot_metrics(metrics)

else :
    metrics = json.load( open( "Data/metrics.json" ) )
    plot_metrics(metrics)