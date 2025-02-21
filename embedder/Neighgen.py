import torch.nn.functional as F
import torch.nn as nn
import torch

class neighGen(nn.Module):
    def __init__(self, feat_shape, args):
        super().__init__()
        self.args = args
        self.gen = Gen(latent_dim=feat_shape,
                          dropout=0.5, num_pred=1, feat_shape=feat_shape)

    def forward(self, feat):
        gen_feat = self.gen(feat)
        return gen_feat
        
class Gen(nn.Module):
    def __init__(self,latent_dim, dropout, num_pred, feat_shape):
        super(Gen, self).__init__()
        self.num_pred=num_pred
        self.feat_shape=feat_shape

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc_flat = nn.Linear(512, self.feat_shape)

        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x
