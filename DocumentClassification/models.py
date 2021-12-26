import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()

        #self.W = nn.Linear(nfeat, 100, bias=bias)
        self.W = nn.Linear(nfeat, nclass, bias=True)
        #self.W1 = nn.Linear(100,nclass, bias=False)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        #x = F.relu(x)
        out = self.W(x)
        #x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)) + (1e-8 * torch.eye(x.shape[1])).unsqueeze(0)
        # x = self.W1(x)
        # x = self.W1(x.transpose(dim0=1,dim1=2))
        #out = torch.diagonal(x, dim1=1,dim2 =2)
        #out = torch.diag(x,dim=1)
        return out
