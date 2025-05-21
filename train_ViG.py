import argparse, shutil, cv2, time, datetime, sys, os, gc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lib.datasets import *
from lib.Loss import initialize_weights, Connect_Loss
from lib.EvalMetric import cal_metrics_train, normalization
from model.UNet import UNet
# from model.csnet import CSNet
# from model.vision_transformer import SwinUnet
from model.gcn import *

import warnings
warnings.filterwarnings("ignore") 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_path", type=str, default="./Results/", help="path of results after test")
parser.add_argument("--data_type", type=str, default="Fundus", help="Fundus, OCTA, 2PFM")
parser.add_argument("--data_name", type=str, default="DRIVE", help="DRIVE, STARE, ROSE1, OCTA500")
opt = parser.parse_args()

train_path = os.path.join("./Data",opt.data_type,opt.data_name,"Train/")
test_path  = os.path.join("./Data",opt.data_type,opt.data_name,"Test/")

Seg_model = UNet(n_channels=3, n_classes=1)
# Seg_model = CSNet(channels=3, classes=1)
# Seg_model = SwinUnet()

GCN_model = ViGSeg_Fundus(img_size=512, patch_size=16)
# GCN_model = ViGSeg_OCTA(img_size=304, patch_size=16)
# GCN_model = ViGSeg_2PFM(img_size=1024, patch_size=16)
initialize_weights(GCN_model)
criterion_BCE = nn.BCELoss()

cuda = True if torch.cuda.is_available() else False
if cuda:
    Seg_model = Seg_model.cuda()
    GCN_model = GCN_model.cuda()
    criterion_BCE.cuda()

optimizer = torch.optim.Adam(GCN_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
transforms_ = [transforms.ToTensor(),]
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ---Eval---
def Eval(test_path, test_save_path):
    test_dir = test_path + "image/"
    test_files = os.listdir(test_dir)
    transform = transforms.Compose(transforms_)
    Eval_path = test_save_path + "Eval/"

    for i,test_file in enumerate(test_files):
        open_name = test_dir + test_file
        save_name = test_save_path + "temp/" + test_file[:-4] + ".png"
        
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

    # copy results for calculate metrics
    for _, img in enumerate(test_files):
        old = test_save_path + "temp/" + img
        new = Eval_path + "/" + img
        shutil.move(old, new)

    # calculate evaluation
    GT_path = test_path + "label/"
    Dice = cal_metrics_train(opt.data_type, Eval_path, GT_path)

    return Dice


# ---Training---
def train(best_seg_pretrain=None):
    print("Load Segmentation Model: %s!" % best_seg_pretrain)
    Seg_model.load_state_dict(torch.load(best_seg_pretrain))

    prev_time = time.time()
    best_dice = 0.0

    for epoch in range(opt.n_epochs):        
        # ---Dataloader---
        if opt.data_type == "Fundus":
            train_dataloader = DataLoader(
                ImageDataset_GCN_Fundus(train_path, transforms_=transforms_),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        elif opt.data_type == "OCTA":
            train_dataloader = DataLoader(
                ImageDataset_GCN_OCTA(train_path, transforms_=transforms_),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        elif opt.data_type == "2PFM":
            train_dataloader = DataLoader(
                ImageDataset_GCN_2PFM(train_path, transforms_=transforms_),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        else:
            raise Exception("Invalid data type!", opt.data_type)
        
        for i, batch in enumerate(train_dataloader):
            img = Variable(batch["image"].type(Tensor))
            lab = Variable(batch["label"].type(Tensor))
            
            seg_pred = Seg_model(img)
            pred = GCN_model(seg_pred)

            loss = criterion_BCE(pred, lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---Log Progress---
            batches_done = epoch * len(train_dataloader) + i
            batches_left = opt.n_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s"
                % (epoch, opt.n_epochs, i, len(train_dataloader), loss.item(), time_left))

        # ---best model---
        Dice = Eval(test_path, opt.save_path)
        if Dice > best_dice and Dice > 0.5:
            best_dice = Dice
            torch.save(GCN_model.state_dict(), "./saved_models/Epo%d_%.4f_GCN.pth" % (epoch,best_dice))
            print("\n============ Save GCN model! ============")

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train(best_seg_pretrain="SEG_MODEL_PTH")
    