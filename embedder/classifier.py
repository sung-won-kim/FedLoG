from torch import nn
import torch
class clsf_module(nn.Module):
    def __init__(self, raw_dim, hid_dim):
        super(clsf_module, self).__init__()
        linear_xavier = nn.Linear(hid_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(linear_xavier.weight, gain=0.001)
        self.msg_mlp = nn.Sequential(
            nn.Linear(raw_dim + 1, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU()
        )

        self.trans_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            linear_xavier
        )

        self.posi_mlp = nn.Sequential(
            nn.Linear(raw_dim + hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, raw_dim)
        )
    
    def msg_model(self, neighbor, sqr_dist):
        out = torch.cat([neighbor, sqr_dist], dim=1)
        out = self.msg_mlp(out)
        return out # directed task specific graph
    
    def coord_model(self, qry_embeds, proto_embeds, edge_index, coord_diff, msg):
        x = torch.cat([proto_embeds, qry_embeds])
        row, col = edge_index
        trans = coord_diff * self.trans_mlp(msg)
        trans = diff_mean(trans, row, num_nodes=x.size(0))
        qry_embeds = qry_embeds + trans[proto_embeds.shape[0]:]
        return qry_embeds
    
    def coord2dist(self, edge_index, qry_embeds, proto_embeds):
        x = torch.cat([proto_embeds, qry_embeds])
        row, col = edge_index
        coord_diff = x[row] - x[col]
        sqr_dist = torch.sum(coord_diff**2, 1).unsqueeze(1)

        return sqr_dist, coord_diff

    def forward(self, edge_index, neighbor, qry_embeds, proto_embeds):
        x = torch.cat([proto_embeds, qry_embeds])
        x_neighbor = torch.cat([proto_embeds, neighbor])
        row, col = edge_index
        sqr_dist, coord_diff = self.coord2dist(edge_index, qry_embeds, proto_embeds)
        msg = self.msg_model(x_neighbor[col], sqr_dist)
        qry_embeds = self.coord_model(
            qry_embeds, proto_embeds, edge_index, coord_diff, msg)
        return neighbor, qry_embeds

class clsf(nn.Module):
    def __init__(self, str_dim, in_dim, n_layers):
        super(clsf, self).__init__()
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, clsf_module(str_dim, in_dim))
        self.n_layers = n_layers
        self.LayerNorm = nn.LayerNorm(in_dim)

    def forward(self, qry_embeds, neighbor, proto_embeds, edge_index): 
        
        qry_embeds = self.LayerNorm(qry_embeds)
        for i in range(0, self.n_layers):
            neighbor, qry_embeds = self._modules["gcl_%d" % i](
                edge_index, neighbor, qry_embeds, proto_embeds)

        return neighbor, qry_embeds

def diff_mean(data, segment_ids, num_nodes):
    result_shape = (num_nodes, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

