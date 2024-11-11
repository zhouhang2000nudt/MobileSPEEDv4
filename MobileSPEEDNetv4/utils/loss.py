import torch
import torch.nn as nn

# ----------pos loss----------

class PosLoss(nn.Module):
    def __init__(self, loss_type: str = 'MSE', **kwargs):
        super().__init__()
        if loss_type == 'MSE':
            self.loss = nn.MSELoss(**kwargs)
        elif loss_type == 'L1':
            self.loss = nn.L1Loss(**kwargs)
        elif loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss(**kwargs)
        else:
            raise ValueError(f"pos loss_type must be 'MSE' or 'L1' or 'SmoothL1', got {loss_type}")
    
    def forward(self, pos_pred, pos_label):
        return self.loss(pos_pred, pos_label)


# ----------ori loss----------

class HellingerLoss(nn.Module):
    def __init__(self):
        super(HellingerLoss, self).__init__()

    def forward(self, p, q):
        # 确保输入是概率分布，即和为 1
        assert torch.allclose(p.sum(dim=-1), torch.tensor(1.0, device=p.device), atol=1e-6)
        assert torch.allclose(q.sum(dim=-1), torch.tensor(1.0, device=q.device), atol=1e-6)

        # 计算 Hellinger 距离
        hellinger_dist = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q))**2, dim=-1)) / torch.sqrt(torch.tensor(2.0))
        return hellinger_dist.mean()

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, p, q):
        # 计算两个分布之间的 Wasserstein 距离
        return (torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)).abs().sum(dim=-1).mean()

class OriLoss(nn.Module):
    def __init__(self, loss_type: str = 'CE', **kwargs):
        super().__init__()
        if loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss(**kwargs)
        elif loss_type == 'KL':
            self.loss = nn.KLDivLoss(**kwargs)
        elif loss_type == 'Hellinger':
            self.loss = HellingerLoss()
        elif loss_type == 'Wasserstein':
            self.loss = WassersteinLoss()
    
    def forward(self, ori_pred, ori_label):
        return self.loss(ori_pred, ori_label)