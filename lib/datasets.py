import glob
import numpy as np
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from skimage import morphology
from skan import Skeleton
from skan import csr
import random
from skimage import morphology
from PIL import Image


class ImageDataset_Fundus(Dataset):
    def __init__(self, root, transforms_=None, mask_type="MaskVSC", mask_ratio=0.1):
        self.transform = transforms.Compose(transforms_)
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.image_files[index % len(self.image_files)])
        lab = Image.open(self.label_files[index % len(self.label_files)])
        
        if (np.random.random() < 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        # shape -> 512,512
        img = np.array(img.resize([512,512]))
        lab = np.array(lab.resize([512,512]))

        if self.mask_type == "MaskVSC":
            sk = morphology.thin(lab).astype(np.uint8)
            sk_copy = sk.copy()
            # object for all properties of the graph
            graph_class = Skeleton(sk)
            branch_data  = csr.summarize(graph_class)
            length = branch_data.shape[0]
            del_list = random.sample(range(0,length), int(self.mask_ratio*length))
            for del_index in del_list:
                integer_coords = tuple(graph_class.path_coordinates(del_index)[1:-1].T.astype(int))
                # remove the branch
                sk[integer_coords] = False
            sk_del = sk_copy-sk
            # dilate the deleted skeleton (kernel size 7*7 for DRIVE)
            kernel = np.ones((7,7),np.uint8)
            masked = cv2.dilate(sk_del, kernel, iterations = 1)*255
            gauss_r = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
            gauss_r = np.where(masked==255, gauss_r, masked)
            gauss_g = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
            gauss_g = np.where(masked==255, gauss_g, masked)
            gauss_b = np.random.normal(0,20,(lab.shape[0],lab.shape[1]))
            gauss_b = np.where(masked==255, gauss_b, masked)
            # save result
            mask_rgb = np.zeros((masked.shape[0],masked.shape[1],3))
            gauss_rgb = np.zeros((masked.shape[0],masked.shape[1],3))
            mask_rgb[:,:,0] = masked
            mask_rgb[:,:,1] = masked
            mask_rgb[:,:,2] = masked
            gauss_rgb[:,:,0] = gauss_r
            gauss_rgb[:,:,1] = gauss_g
            gauss_rgb[:,:,2] = gauss_b
            result = np.uint8(np.add(255-img, mask_rgb))
            result = np.uint8(np.add(result, gauss_rgb))
            img = 255-result            
        elif self.mask_type == "None":
            pass
        else:
            raise Exception("Invalid mask type!", self.mask_type)

        if np.max(lab)==255:
            lab = (lab >= 128).astype(np.uint8) * 255
        elif np.max(lab)==1:
            lab = (lab >= 0.5).astype(np.uint8)

        img = self.transform(img)
        lab = self.transform(lab)

        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)



class ImageDataset_OCTA(Dataset):
    def __init__(self, root, transforms_=None, mask_type="MaskVSC", mask_ratio=0.1):
        self.transform = transforms.Compose(transforms_)
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))

    def __getitem__(self, index):
        img = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(self.label_files[index % len(self.label_files)], cv2.IMREAD_GRAYSCALE)

        # shape resize for OCTA500
        img = cv2.resize(img, (384,384))
        lab = cv2.resize(lab, (384,384))

        # add mask and noise
        if self.mask_type == "MaskVSC":
            sk = morphology.thin(lab).astype(np.uint8)
            sk_copy = sk.copy()
            # object for all properties of the graph
            graph_class = Skeleton(sk)
            branch_data  = csr.summarize(graph_class)
            length = branch_data.shape[0]
            del_list = random.sample(range(0,length), int(self.mask_ratio*length))
            for del_index in del_list:
                integer_coords = tuple(graph_class.path_coordinates(del_index)[1:-1].T.astype(int))
                # remove the branch
                sk[integer_coords] = False
            sk_del = sk_copy-sk
            # dilate the deleted skeleton
            kernel = np.ones((3,3),np.uint8)
            masked = cv2.dilate(sk_del, kernel, iterations = 1)*255
            gauss = np.random.normal(0,100,(img.shape[0],img.shape[1]))
            gauss = np.where(masked==255, gauss, masked)
            # save result
            result = cv2.add(255-img, masked)
            result = np.add(result, gauss)
            img = 255-result
        elif self.mask_type == "None":
            pass
        else:
            raise Exception("Invalid mask type!", self.mask_type)

        if np.max(lab)==255:
            lab = (lab >= 128).astype(np.uint8) * 255
        elif np.max(lab)==1:
            lab = (lab >= 0.5).astype(np.uint8)

        img = self.transform(img)
        lab = self.transform(lab)
        
        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)



class ImageDataset_2PFM(Dataset):
    def __init__(self, root, transforms_=None, mask_type="MaskVSC", mask_ratio=0.1):
        self.transform = transforms.Compose(transforms_)
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))

    def __getitem__(self, index):
        img = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(self.label_files[index % len(self.label_files)], cv2.IMREAD_GRAYSCALE)

        # add mask and noise
        if self.mask_type == "MaskVSC":
            sk = morphology.thin(lab).astype(np.uint8)
            sk_copy = sk.copy()
            # object for all properties of the graph
            graph_class = Skeleton(sk)
            branch_data  = csr.summarize(graph_class)
            length = branch_data.shape[0]
            del_list = random.sample(range(0,length), int(self.mask_ratio*length))
            for del_index in del_list:
                integer_coords = tuple(graph_class.path_coordinates(del_index)[1:-1].T.astype(int))
                # remove the branch
                sk[integer_coords] = False
            sk_del = sk_copy-sk
            # dilate the deleted skeleton
            kernel = np.ones((11,11),np.uint8)
            masked = cv2.dilate(sk_del, kernel, iterations = 1)*255
            gauss = np.random.normal(0,20,(img.shape[0],img.shape[1]))
            gauss = np.where(masked==255, gauss, masked)
            # save result
            result = cv2.add(255-img, masked)
            result = np.add(result, gauss)
            img = 255-result
        elif self.mask_type == "None":
            pass
        else:
            raise Exception("Invalid mask type!", self.mask_type)

        if np.max(lab)==255:
            lab = (lab >= 128).astype(np.uint8) * 255
        elif np.max(lab)==1:
            lab = (lab >= 0.5).astype(np.uint8)

        img = self.transform(img)
        lab = self.transform(lab)
        
        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)
