import scipy.io as sio
import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist

def dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]

        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))


    return intra_dist


if __name__ == '__main__':

    dataset = 'pubmed'
    data = sio.loadmat('{}.mat'.format(dataset))
    feature = data['fea']
    if sp.issparse(feature):
        feature = feature.todense()

    adj = data['W']
    gnd = data['gnd']
    gnd = gnd.T
    gnd = gnd - 1
    gnd = gnd[0, :]
    k = len(np.unique(gnd))
    adj = sp.coo_matrix(adj)
    intra_list = []
    intra_list.append(10000)


    acc_list = []
    nmi_list = []
    f1_list = []
    stdacc_list = []
    stdnmi_list = []
    stdf1_list = []
    max_iter = 60
    rep = 10
    t = time.time()
    #adj_normalized = preprocess_adj(adj)
    #adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2
    adj_normalized = preprocess_adj(adj,loop=False)
    # adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized + adj_normalized.dot(adj_normalized)) / 3
    total_dist = []

    tt = 0
    alpha = 0.05 #0.05 for pubmed, 0.1 for citeseer
    feature_ori = feature.astype('float64')
    #emb = feature.astype('float64')
    #emb = feature.astype('float64')
    emb = np.zeros_like(feature).astype('float64')
    oneV = np.ones(feature.shape[0],1)
    den = np.zeros_like(oneV)
    while 1:
        tt = tt + 1
        power = tt
        intraD = np.zeros(rep)


        ac = np.zeros(rep)
        nm = np.zeros(rep)
        f1 = np.zeros(rep)



        feature = adj_normalized.dot(feature)
        
        emb += feature
        emb_norm = emb/tt
        u, s, v = sp.linalg.svds(emb_norm, k=16, which='LM')
        u = normalize(emb_norm.dot(v.T))




        for i in range(rep):
            kmeans = KMeans(n_clusters=k).fit(u)
            predict_labels = kmeans.predict(u)
            intraD[i] = square_dist(predict_labels, u)
            #intraD[i] = dist(predict_labels, feature)
            cm = clustering_metrics(gnd, predict_labels)
            ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

        intramean = np.mean(intraD)
        acc_means = np.mean(ac)
        acc_stds = np.std(ac)
        nmi_means = np.mean(nm)
        nmi_stds = np.std(nm)
        f1_means = np.mean(f1)
        f1_stds = np.std(f1)

        intra_list.append(intramean)
        acc_list.append(acc_means)
        stdacc_list.append(acc_stds)
        nmi_list.append(nmi_means)
        stdnmi_list.append(nmi_stds)
        f1_list.append(f1_means)
        stdf1_list.append(f1_stds)
        print('power: {}'.format(power),
              'intra_dist: {}'.format(intramean),
              'acc_mean: {}'.format(acc_means),
              'acc_std: {}'.format(acc_stds),
              'nmi_mean: {}'.format(nmi_means),
              'nmi_std: {}'.format(nmi_stds),
              'f1_mean: {}'.format(f1_means),
              'f1_std: {}'.format(f1_stds))

        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:
            print('bestpower: {}'.format(tt - 1))
            t = time.time() - t
            print(t)
            break





