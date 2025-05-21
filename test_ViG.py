import argparse, shutil, cv2, os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.datasets import *
from lib.EvalMetric import cal_metrics, normalization
from model.UNet import UNet
# from model.csnet import CSNet
# from model.vision_transformer import SwinUnet
from model.gcn import *

import warnings
warnings.filterwarnings("ignore") 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="./Results/", help="path of results after test")
parser.add_argument("--data_type", type=str, default="OCTA", help="Fundus, OCTA, or 2PFM")
parser.add_argument("--data_name", type=str, default="OCTA500", help="DRIVE, STARE, ROSE1, OCTA500, or 2PFM")
opt = parser.parse_args()

test_path  = os.path.join("./Data",opt.data_type,opt.data_name,"Test/")

Seg_model = UNet(n_channels=3, n_classes=1)
# Seg_model = CSNet(channels=3, classes=1)
# Seg_model = SwinUnet()

GCN_model = ViGSeg_Fundus(img_size=512, patch_size=16)
# GCN_model = ViGSeg_OCTA(img_size=384, patch_size=16)
# GCN_model = ViGSeg_2PFM(img_size=1024, patch_size=16)

cuda = True if torch.cuda.is_available() else False
if cuda:
    Seg_model = Seg_model.cuda()
    GCN_model = GCN_model.cuda()

transforms_ = [transforms.ToTensor(),]
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ---Testing---
def Test(best_Segtrain, best_GCNtrain, test_save_path):
    Seg_model.load_state_dict(torch.load(best_Segtrain))
    GCN_model.load_state_dict(torch.load(best_GCNtrain))
    test_dir = test_path + "image/"
    test_files = os.listdir(test_dir)
    transform = transforms.Compose(transforms_)
    Eval_path = test_save_path + "Eval/"

    for _,test_file in enumerate(test_files):
        open_name = test_dir + test_file
        save_name = test_save_path + "temp/" + test_file

        # Fundus
        img = Image.open(open_name)
        img = np.array(img.resize([512,512]))

        # OCTA & 2PFM
        # img = cv2.imread(open_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (384,384)) # OCTA500

        img = (transform(img)).unsqueeze(0)
        img = Variable(img.type(Tensor))

        seg_pred = Seg_model(img)
        pred_mask = GCN_model(seg_pred)

        pred = pred_mask.data.squeeze().cpu().numpy()
        pred = normalization(np.array(pred))
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0

        res_image = pred * 255
        cv2.imwrite(save_name, res_image)

    # copy results for calculate evaluation
    for _, img in enumerate(test_files):
        old = test_save_path + "temp/" + img
        new = Eval_path + "/" + img
        shutil.move(old, new)

    # calculate evaluation
    GT_path = test_path + "label/"
    cal_metrics(opt.data_type, Eval_path, GT_path)


if __name__ == "__main__":
    Test(best_Segtrain="SEG_MODEL_PTH",
         best_GCNtrain="ViG_MODEL_PTH",
         test_save_path=opt.save_path)
    