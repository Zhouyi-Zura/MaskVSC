import torch
import torch.nn as nn
import kornia as K

def torch_norm(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-10)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Connect_Loss(nn.Module):
    def __init__(self):
        super(Connect_Loss, self).__init__()

    def forward(self, img, lab):
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).cuda()
        img = K.morphology.opening(img,kernel)
        lab = K.morphology.opening(lab,kernel)
        img = torch_norm(img)
        lab = torch_norm(lab)

        img[img>=0.5] = 1.
        img[img <0.5] = 0.
        lab[lab>=0.5] = 1.
        lab[lab <0.5] = 0.

        img_labels_out = K.contrib.connected_components(img, num_iterations=150)
        lab_labels_out = K.contrib.connected_components(lab, num_iterations=150)
        ccs = torch.unique(img_labels_out).size()
        ccsg= torch.unique(lab_labels_out).size()

        numSg = torch.count_nonzero(lab)
        C = torch.abs(torch.sub(ccsg[0],ccs[0]))/numSg

        return torch.mean(C)
