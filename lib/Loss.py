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

    def smoothness_penalty(self, tensor):
        # Compute gradient differences for smooth connectivity
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)

        return smoothness_loss

    def forward(self, img, lab):
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img.dtype).cuda()
        img = K.morphology.opening(img,kernel)
        lab = K.morphology.opening(lab,kernel)
        
        # Apply a sigmoid to approximate binarization in a differentiable way
        img = torch.sigmoid(10 * (img - 0.5))  # Sharpens values close to 0 or 1
        lab = torch.sigmoid(10 * (lab - 0.5))  # Sharpens values close to 0 or 1

        smoothness_img = self.smoothness_penalty(img)
        smoothness_lab = self.smoothness_penalty(lab)

        C = torch.abs(smoothness_img - smoothness_lab) / torch.count_nonzero(lab)

        return C
