## Simple Spectral Graph Convolutional

### Overview
This repo contains an example implementation of the Simple Spectral Graph Convolutional (S^2GC) model.
This code is based on SGC. We will update the code in Text Classification and Node Clustering latter. 

SGC removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a linear model. However, it also cannot beat GCNs in most benchmarks. S^2GC is a new baseline method for GCNs and downstream applications.

S^2GC achieves competitive performance while saving much training time, basically the computational cost is very close to SSGC.

Dataset | Metric |
:------:|:------:|
Cora    | Acc: 83.0 %
Citeseer| Acc: 73.6 %
Pubmed  | Acc: 80.6 %
Reddit  | F1:  95.3 %

This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed) and social network (Reddit).


### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).

### Usage

```
$ python citation_cora.py
$ python citation_citeseer.py 
$ python citation_pubmed.py 
```

