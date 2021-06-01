import argparse

import torch
import torch.nn.functional as F

from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from typing import Optional
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from logger import Logger
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing



class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, dropout: float = 0.05, **kwargs):
        super(SGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        #self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            x_set = []
            alpha = 0.05
            output = alpha * x
            #temp_edge_index, edge_weight = dropout_adj(edge_index, 0.5)
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                # x_set.append(x)
                output = output + (1. / self.K) * x
            # x = torch.stack(x_set,2)
            # alpha = 0.05
            # x = (1-alpha)*torch.mean(x,2).squeeze() + alpha*x_ori
            x = output
            if self.cached:
                self._cached_x = x
        else:
            x = cache

        return x#self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_layers, dropout):
        super(SGC, self).__init__()
        self.conv1 = SGConv(
            in_channels, hidden, K=num_layers, cached=True)
        self.lin = torch.nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        #x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv1(x, edge_index)#F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        # model = GCN(data.num_features, args.hidden_channels,
        #             dataset.num_classes, args.num_layers,
        #             args.dropout).to(device)
        model = SGC(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout).to(device)
    features = model(data.x, data.adj_t)
    torch.save(features,'embedding.pt')
    # evaluator = Evaluator(name='ogbn-arxiv')
    # logger = Logger(args.runs, args)
    #
    # for run in range(args.runs):
    #     model.reset_parameters()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #     #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,  momentum=0.9)
    #     for epoch in range(1, 1 + args.epochs):
    #         loss = train(model, data, train_idx, optimizer)
    #         result = test(model, data, split_idx, evaluator)
    #         logger.add_result(run, result)
    #
    #         if epoch % args.log_steps == 0:
    #             train_acc, valid_acc, test_acc = result
    #             print(f'Run: {run + 1:02d}, '
    #                   f'Epoch: {epoch:02d}, '
    #                   f'Loss: {loss:.4f}, '
    #                   f'Train: {100 * train_acc:.2f}%, '
    #                   f'Valid: {100 * valid_acc:.2f}% '
    #                   f'Test: {100 * test_acc:.2f}%')
    #
    #     logger.print_statistics(run)
    # logger.print_statistics()


if __name__ == "__main__":
    main()
