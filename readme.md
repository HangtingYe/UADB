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
Isolation Forest (IForest) [paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) that isolates observations by randomly selecting a feature and a splitting point;

Histogram-based outlier detection (HBOS) [paper](https://www.goldiges.de/publications/HBOS-KI-2012.pdf) assumes the feature independence and calculates the degree of outlyingness by building histograms; 

Local Outlier Factor (LOF) [paper](https://dl.acm.org/doi/pdf/10.1145/342009.335388) measures the local deviation of the density of a sample with respect to its neighbors;

K-Nearest Neighbors (KNN) [paper](https://dl.acm.org/doi/pdf/10.1145/342009.335437) views an instance's distance to its kth nearest neighbor as the outlying score;

Principal Component Analysis (PCA) [paper](https://apps.dtic.mil/sti/pdfs/ADA465712.pdf) is a linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. In anomaly detection, it projects the data to the lower dimensional space and then reconstruct it, thus the reconstruction errors are viewed as the anomaly scores;

One-class SVM (OCSVM) [paper](https://proceedings.neurips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf) maximizes the margin between the abnormal and the normal samples, and uses the hyperplane that determines the margin for decision;

Clustering Based Local Outlier Factor (CBLOF) [paper](http://www.diag.uniroma1.it/~sassano/STAGE/Outliers.pdf) classifies the samples into small clusters and large clusters  and then using the distance among clusters as anomaly scores;

Connectivity-Based Outlier Factor (COF) [paper](https://link.springer.com/chapter/10.1007/3-540-47887-6_53) uses the ratio of average chaining distance of data point and the average of average chaining distance of k nearest neighbor of the data point, as the outlier score for observations;

Subspace Outlier Detection (SOD) [paper](https://www.dbs.ifi.lmu.de/~zimek/publications/PAKDD2009/pakdd09-SOD.pdf) detects outlier in varying subspaces of a high dimensional feature space;

Empirical-Cumulative-distribution-based Outlier Detection (ECOD) [paper](https://arxiv.org/pdf/2201.00382.pdf) is a parameter-free, highly interpretable outlier detection algorithm based on empirical CDF functions;

Gaussian Mixture Models (GMM) [paper](http://www.leap.ee.iisc.ac.in/sriram/teaching/MLSP_19/refs/GMM_Tutorial_Reynolds.pdf) fit k Gaussians to the data. Then for each data point, calculate the probabilities of belonging to each of the clusters, where the lower probabilities indicate higher anomaly scores;

Lightweight on-line detector of anomalies (LODA) [paper](https://link.springer.com/article/10.1007/s10994-015-5521-0) is an ensemble detector and is particularly useful in domains where a large number of samples need to be processed in real-time or in domains where the data stream is subject to concept drift and the detector needs to be updated online;

Copula Based Outlier Detector (COPOD) [paper](https://arxiv.org/pdf/2009.09463.pdf) is a parameter-free, highly interpretable outlier detection algorithm based on empirical copula models;

Deep Support Vector Data Description (DeepSVDD) [paper](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data, the distance of the transformed embedding to the hypersphere's center is used to calculate the anomaly score.

For all source UAD models, we use their default parameters in their original papers (which have been fine-tuned to achieve the best performance).
Please refer to [PyOD](https://pyod.readthedocs.io/en/latest/pyod.models.html) for more information.

## Runtime of iterative training with 10 iterations on 84 tabular datasets.
| Dataset            | time(seconds) |
| ------------------ | ------------- |
| 1_abalone          | 45.97332      |
| 2_ALOI             | 62.09925      |
| 3_annthyroid       | 55.46907      |
| 4_Arrhythmia       | 39.06063      |
| 5_breastw          | 36.12518      |
| 6_cardio           | 33.07165      |
| 7_Cardiotocography | 33.20133      |
| 9_concrete         | 31.79388      |
| 10_cover           | 64.64015      |
| 11_fault           | 39.39834      |
| 12_glass           | 37.71293      |
| 13_HeartDisease    | 37.38352      |
| 14_Hepatitis       | 38.01874      |
| 15_http            | 61.8857       |
| 16_imgseg          | 40.50281      |
| 17_InternetAds     | 45.73133      |
| 18_Ionosphere      | 37.77196      |
| 19_landsat         | 53.47792      |
| 20_letter          | 40.62032      |
| 21_Lymphography    | 38.29522      |
| 23_mammography     | 64.00619      |
| 24_mnist           | 59.77169      |
| 25_musk            | 45.249        |
| 26_optdigits       | 48.52583      |
| 27_PageBlocks      | 48.49409      |
| 28_Parkinson       | 37.96288      |
| 29_pendigits       | 54.03328      |
| 30_Pima            | 38.34605      |
| 31_satellite       | 53.66807      |
| 32_satimage-2      | 50.83709      |
| 33_shuttle         | 61.62736      |
| 34_skin            | 63.52546      |
| 35_smtp            | 61.39091      |
| 36_SpamBase        | 45.18679      |
| 37_speech          | 47.31526      |
| 38_Stamps          | 37.86243      |
| 39_thyroid         | 46.09605      |
| 40_vertebral       | 36.9513       |
| 41_vowels          | 38.88513      |
| 42_Waveform        | 45.00404      |
| 43_WBC             | 38.99096      |
| 44_WDBC            | 35.97541      |
| 45_Wilt            | 46.98276      |
| 46_wine            | 37.96173      |
| 47_WPBC            | 37.3959       |
| 48_yeast           | 39.1033       |
| 49_CIFAR10_0       | 52.26063      |
| 49_CIFAR10_1       | 52.14188      |
| 49_CIFAR10_2       | 45.87855      |
| 49_CIFAR10_3       | 46.25881      |
| 49_CIFAR10_4       | 46.14659      |
| 49_CIFAR10_5       | 52.32802      |
| 49_CIFAR10_6       | 51.16338      |
| 49_CIFAR10_7       | 53.35524      |
| 49_CIFAR10_8       | 53.4575       |
| 49_CIFAR10_9       | 50.95251      |
| 50_FashionMNIST_0  | 54.4895       |
| 50_FashionMNIST_1  | 53.37795      |
| 50_FashionMNIST_2  | 45.93535      |
| 50_FashionMNIST_3  | 47.02001      |
| 50_FashionMNIST_4  | 45.6286       |
| 50_FashionMNIST_5  | 46.80384      |
| 50_FashionMNIST_6  | 43.90277      |
| 50_FashionMNIST_7  | 45.52983      |
| 50_FashionMNIST_8  | 49.52233      |
| 50_FashionMNIST_9  | 49.5379       |
| 51_SVHN_0          | 53.37439      |
| 51_SVHN_1          | 55.47349      |
| 51_SVHN_2          | 63.5158       |
| 51_SVHN_3          | 56.23896      |
| 51_SVHN_4          | 56.96233      |
| 51_SVHN_5          | 51.96182      |
| 51_SVHN_6          | 51.20783      |
| 51_SVHN_7          | 64.19061      |
| 51_SVHN_8          | 52.99753      |
| 51_SVHN_9          | 63.43625      |
| 52_agnews_0        | 58.6838       |
| 52_agnews_1        | 56.56641      |
| 52_agnews_2        | 56.18832      |
| 52_agnews_3        | 63.5042       |
| 53_amazon          | 59.95886      |
| 54_imdb            | 64.06704      |
| 55_yelp            | 58.62         |

## Surprising effects on source UAD's decision boundaries.
![image](figures/decision_boundary.png)
