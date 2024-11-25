import argparse, shutil, cv2, os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.datasets import *
from lib.EvalMetric import cal_metrics, normalization
from model.UNet import UNet
# from model.csnet import CSNet
# from model.vision_transformer import SwinUnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="./Results/", help="path of results after test")
parser.add_argument("--data_type", type=str, default="Fundus", help="Fundus, OCTA, 2PFM")
parser.add_argument("--data_name", type=str, default="DRIVE", help="DRIVE, STARE, ROSE1, OCTA500")
opt = parser.parse_args()

test_path  = os.path.join("./Data",opt.data_type,opt.data_name,"Test/")

generator = UNet(n_channels=3, n_classes=1)
# generator = CSNet(channels=3, classes=1)
# generator = SwinUnet()

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()

transforms_ = [transforms.ToTensor(),]

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ---Testing---
def Test(best_pretrain, test_save_path):
    generator.load_state_dict(torch.load(best_pretrain))
    test_dir = test_path + "image/"
    test_files = os.listdir(test_dir)
    transform = transforms.Compose(transforms_)
    Eval_path = test_save_path + "Eval/"

    for test_file in test_files:
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

    # copy results for calculate evaluation
    for _, img in enumerate(test_files):
        old = test_save_path + "temp/" + img
        new = Eval_path + "/" + img
        shutil.move(old, new)

    # calculate evaluation
    GT_path = test_path + "label/"
    cal_metrics(opt.data_type, Eval_path, GT_path)


if __name__ == "__main__":
    Test(best_pretrain="xxx.pth", test_save_path=opt.save_path)
    
