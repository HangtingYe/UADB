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


## Surprising effects on source UAD's decision boundaries.
![image](figures/decision_boundary.png)


