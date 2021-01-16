## Simple Spectral Graph Convolutional

### Overview
This repo contains an example implementation of the Simple Spectral Graph Convolutional
(S^2GC) model This code is based on SGC

SGC removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a linear model. 
For an illustration, ![](https://github.com/Tiiiger/SimpleGraphConvolution/blob/master/model.jpg "SGC")

SGC achieves competitive performance while saving much training time. For reference, on a GTX 1080 Ti,

Dataset | Metric | Training Time 
:------:|:------:|:-----------:|
Cora    | Acc: 81.0 %     | 0.13s
Citeseer| Acc: 71.9 %     | 0.14s
Pubmed  | Acc: 78.9 %     | 0.29s
Reddit  | F1:  94.9 %     | 2.7s

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

