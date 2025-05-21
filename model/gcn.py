import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x

class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
            .unfold(2, self.patch_size, self.patch_size).contiguous()\
            .view(B, -1, C, self.patch_size, self.patch_size)
        return x
    
class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(
            F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x

class VGNN(nn.Module):
    def __init__(self, num_channels=1, img_size=512, patch_size=16,
                 num_ViGBlocks=16, num_edges=9, head_num=1):
        super().__init__()

        in_features = num_channels*patch_size*patch_size
        out_feature = patch_size*patch_size
        num_patches = (img_size/patch_size)**2

        self.patchifier = SimplePatchifier(patch_size=patch_size)
        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(int(num_patches), int(out_feature)))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class ViGSeg_Fundus(nn.Module):
    def __init__(self, num_channels=1, img_size=512, patch_size=16,
                 num_ViGBlocks=16, num_edges=9, head_num=1, n_classes=1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.backbone = VGNN(num_channels=num_channels, img_size=img_size, patch_size=patch_size,
                             num_ViGBlocks=num_ViGBlocks, num_edges=num_edges, head_num=head_num)

        self.predictor = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
            )
        
        self.final = nn.Conv2d(n_classes*2, n_classes, kernel_size=1)

    def forward(self, x):
        input = x
        features = self.backbone(x)
        B, N, C = features.shape

        x = features.unsqueeze(2).view(B, N, self.patch_size, self.patch_size)
        
        x = self.predictor(x)

        for i in range(int(math.log2(self.img_size/self.patch_size))):
            temp_size = (self.patch_size << 1) << i
            x = F.interpolate(x, size=(temp_size, temp_size), mode='bilinear')

        x = F.sigmoid(x)
        x = torch.cat((x, input), dim=1)
        x = self.final(x)
        x = F.sigmoid(x)

        return x


class ViGSeg_OCTA(nn.Module):
    def __init__(self, num_channels=1, img_size=304, patch_size=16,
                 num_ViGBlocks=16, num_edges=9, head_num=1, n_classes=1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.backbone = VGNN(num_channels=num_channels, img_size=img_size, patch_size=patch_size,
                             num_ViGBlocks=num_ViGBlocks, num_edges=num_edges, head_num=head_num)

        self.predictor = nn.Sequential(
            # nn.Conv2d(361, 256, kernel_size=1), # ROSE1
            nn.Conv2d(576, 256, kernel_size=1), # OCTA500
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # For OCTA-500
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
            )
        
        # # SwinUnet
        # self.predictor = nn.Sequential(
        #     nn.Conv2d(196, 128, kernel_size=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
            
        #     # # For OCTA-500
        #     # nn.Conv2d(128, 128, kernel_size=1),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        #     # # For OCTA-500

        #     nn.Conv2d(128, 64, kernel_size=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     )
        
        self.final = nn.Conv2d(n_classes*2, n_classes, kernel_size=1)

    def forward(self, x):
        input = x
        features = self.backbone(x)
        B, N, C = features.shape

        x = features.unsqueeze(2).view(B, N, self.patch_size, self.patch_size)
        
        x = self.predictor(x)

        for i in range(int(math.log2(self.img_size/self.patch_size))):
            temp_size = (self.patch_size << 1) << i
            x = F.interpolate(x, size=(temp_size, temp_size), mode='bilinear')
        
        x = F.interpolate(x, size=(input.shape[2], input.shape[3]), mode='bilinear')

        x = F.sigmoid(x)
        x = torch.cat((x, input), dim=1)
        x = self.final(x)
        x = F.sigmoid(x)

        return x


class ViGSeg_2PFM(nn.Module):
    def __init__(self, num_channels=1, img_size=1024, patch_size=16,
                 num_ViGBlocks=16, num_edges=9, head_num=1, n_classes=1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.backbone = VGNN(num_channels=num_channels, img_size=img_size, patch_size=patch_size,
                             num_ViGBlocks=num_ViGBlocks, num_edges=num_edges, head_num=head_num)

        self.predictor = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
            )
        
        self.final = nn.Conv2d(n_classes*2, n_classes, kernel_size=1)

    def forward(self, x):
        input = x
        features = self.backbone(x)
        B, N, C = features.shape

        x = features.unsqueeze(2).view(B, N, self.patch_size, self.patch_size)
        
        x = self.predictor(x)

        for i in range(int(math.log2(self.img_size/self.patch_size))):
            temp_size = (self.patch_size << 1) << i
            x = F.interpolate(x, size=(temp_size, temp_size), mode='bilinear')
        
        x = F.interpolate(x, size=(input.shape[2], input.shape[3]), mode='bilinear')

        x = F.sigmoid(x)
        x = torch.cat((x, input), dim=1)
        x = self.final(x)
        x = F.sigmoid(x)

        return x

