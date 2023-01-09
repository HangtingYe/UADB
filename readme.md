Official code and data repository of [**UADB**: Unsupervised Anomaly Detection Booster].
__Please star, watch, and fork UADB for the active updates!__

## What is UADB?
UADB is a booster for unsupervised anomaly detection (UAD) on tabular tasks.
Note that UADB is not a universal winner on all taular tasks, however, it is a model-agnostic framework that can generally enhance any UAD on all types of tabular datasets in a unified way.

## How to train?
Prepare (create Results first)
* ```mkdir Results```

Select tabular data and source UAD needed to be enhanced
* modify config.py

Run UADB
* ```python main.py```

## Mainstream Unsupervised Anomaly Detection Models.
Isolation Forest (IForest) \cite{liu2008isolation} that isolates observations by randomly selecting a feature and a splitting point.
Histogram-based outlier detection (HBOS) \cite{goldstein2012histogram} assumes the feature independence and calculates the degree of outlyingness by building histograms; 
Local Outlier Factor (LOF) \cite{breunig2000lof} measures the local deviation of the density of a sample with respect to its neighbors;
K-Nearest Neighbors (KNN) \cite{ramaswamy2000efficient} views an instance's distance to its kth nearest neighbor as the outlying score;
Principal Component Analysis (PCA) \cite{shyu2003novel} is a linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. In anomaly detection, it projects the data to the lower dimensional space and then reconstruct it, thus the reconstruction errors are viewed as the anomaly scores;
One-class SVM (OCSVM) \cite{scholkopf1999support} maximizes the margin between the abnormal and the normal samples, and uses the hyperplane that determines the margin for decision;
Clustering Based Local Outlier Factor (CBLOF) \cite{he2003discovering} classifies the samples into small clusters and large clusters  and then using the distance among clusters as anomaly scores;
Connectivity-Based Outlier Factor (COF) \cite{tang2002enhancing} uses the ratio of average chaining distance of data point and the average of average chaining distance of k nearest neighbor of the data point, as the outlier score for observations;
Subspace Outlier Detection (SOD) \cite{kriegel2009outlier} detects outlier in varying subspaces of a high dimensional feature space;
Empirical-Cumulative-distribution-based Outlier Detection (ECOD) \cite{li2022ecod} is a parameter-free, highly interpretable outlier detection algorithm based on empirical CDF functions;
Gaussian Mixture Models (GMM) \cite{reynolds2009gaussian} fit k Gaussians to the data. Then for each data point, calculate the probabilities of belonging to each of the clusters, where the lower probabilities indicate higher anomaly scores;
Lightweight on-line detector of anomalies (LODA) \cite{pevny2016loda} is an ensemble detector and is particularly useful in domains where a large number of samples need to be processed in real-time or in domains where the data stream is subject to concept drift and the detector needs to be updated online;
Copula Based Outlier Detector (COPOD) \cite{li2020copod} is a parameter-free, highly interpretable outlier detection algorithm based on empirical copula models;
Deep Support Vector Data Description (DeepSVDD) \cite{ruff2018deep} trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data, the distance of the transformed embedding to the hypersphere's center is used to calculate the anomaly score.


## Surprising effects on source UAD's decision boundaries.
![image](figures/decision_boundary.png)


