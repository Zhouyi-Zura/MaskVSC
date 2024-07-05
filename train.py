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
from model.csnet import CSNet
from model.vision_transformer import SwinUnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_path", type=str, default="./Results/", help="path of results after test")
parser.add_argument("--data_type", type=str, default="Fundus", help="Fundus, OCTA")
parser.add_argument("--data_name", type=str, default="DRIVE", help="DRIVE, STARE, ROSE1, OCTA500")
parser.add_argument('--mask_type', type=str, default="MaskVSC", help='MaskVSC, or None')
opt = parser.parse_args()

train_path = os.path.join("./Data",opt.data_type,opt.data_name,"Train/")
test_path  = os.path.join("./Data",opt.data_type,opt.data_name,"Test/")

generator = UNet(n_channels=3, n_classes=1)
# generator = CSNet(n_channels=3, n_classes=1)
# generator = SwinUnet()

initialize_weights(generator)

criterion_BCE = nn.BCELoss()
criterion_Con = Connect_Loss()

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()
    criterion_BCE.cuda()
    criterion_Con.cuda()

optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = [
    transforms.ToTensor(),
]

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ---Eval---
def Eval(test_path, test_save_path):
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

        # OCTA
        # img = cv2.imread(open_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (384,384)) # OCTA500

        img = (transform(img)).unsqueeze(0)
        img = Variable(img.type(Tensor))

        pred = generator(img)
        pred = pred.data.squeeze().cpu().numpy()
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
def train():
    prev_time = time.time()
    best_dice = 0.0

    for epoch in range(opt.n_epochs):
        # ---Masking Ratio---
        epoch_ratio = epoch/opt.n_epochs
        if 0 <= epoch_ratio < 0.1 or 0.9 < epoch_ratio <= 1:
            curr_mask_ratio = 0
        elif 0.1 <= epoch_ratio <= 0.5:
            curr_mask_ratio = 0.4 * (epoch_ratio - 0.1) / 0.4
        elif 0.5 < epoch_ratio <= 0.9:
            curr_mask_ratio = 0.4 * (0.9 - epoch_ratio) / 0.4
        else:
            raise Exception("Invalid epoch_ratio!", epoch_ratio)

        # ---Dataloader---
        if opt.data_type == "Fundus":
            train_dataloader = DataLoader(
                ImageDataset_Fundus(train_path, transforms_=transforms_,
                                  mask_type=opt.mask_type, mask_ratio=curr_mask_ratio),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        elif opt.data_type == "OCTA":
            train_dataloader = DataLoader(
                ImageDataset_OCTA(train_path, transforms_=transforms_,
                                  mask_type=opt.mask_type, mask_ratio=curr_mask_ratio),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        else:
            raise Exception("Invalid data type!", opt.data_type)

        # ---Training one epoch---
        for i, batch in enumerate(train_dataloader):
            img = Variable(batch["image"].type(Tensor))
            lab = Variable(batch["label"].type(Tensor))
            
            pred = generator(img)
            
            # ---Loss Function---
            loss_BCE = criterion_BCE(pred, lab)
            loss_Con = criterion_Con(pred, lab)

            loss = loss_BCE + loss_Con

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
            torch.save(generator.state_dict(), "./saved_models/Epo%d_%.4f.pth" % (epoch,best_dice))
            print("\n============ Save model! ============")
        
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train()
