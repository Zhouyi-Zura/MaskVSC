import numpy as np
import cv2, glob
from PIL import Image
from skimage import morphology
from skimage.morphology import skeletonize
from lib.BettiMatching import *


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice_metric(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))

    return 2*tprec*tsens/(tprec+tsens)

def Betti_number(img, lab):
    BM = BettiMatching(img, lab, filtration='superlevel')
    Betti_NE_0 = BM.Betti_number_error(dimensions=[0])
    Betti_NE_1 = BM.Betti_number_error(dimensions=[1])
    Betti_ME_0 = BM.loss(dimensions=[0])
    Betti_ME_1 = BM.loss(dimensions=[1])

    return Betti_NE_0, Betti_NE_1, Betti_ME_0, Betti_ME_1


def cal_metrics_train(data_type, pre_path, GT_path):
    TP=FPN=0
    Dice_=[]
    Jaccard_=[]
    clDice_=[]
    C_=[]
    A_=[]
    L_=[]
    CAL_=[]
    
    image_files = sorted(glob.glob(pre_path + "/*.*"))
    label_files = sorted(glob.glob(GT_path + "/*.*"))

    for i, image_file in enumerate(image_files):
        pre_file_path =image_file
        true_file_path=label_files[i]

        if data_type == "Fundus":
            img = Image.open(pre_file_path)
            lab = Image.open(true_file_path)
            img = img.resize([512,512])
            lab = lab.resize([512,512])
            img = (np.array(img,dtype=np.uint8)).clip(max=1)
            lab = (np.array(lab,dtype=np.uint8)).clip(max=1)
        elif data_type == "OCTA":
            img = cv2.imread(pre_file_path, cv2.IMREAD_GRAYSCALE)
            lab = cv2.imread(true_file_path,cv2.IMREAD_GRAYSCALE)
            lab = cv2.resize(lab, (384,384)) # OCTA-500
            img = img.clip(max=1)
            lab = lab.clip(max=1)
        else:
            raise Exception("Invalid data type!", data_type)
        
        if np.max(lab)==255:
            img[img<255] = 0
            lab[lab<255] = 0
        elif np.max(lab)==1:
            img[img<1] = 0
            lab[lab<1] = 0

        # Calculate Dice + Glob Jaccard
        TP=TP+np.sum(img*lab)
        FPN=FPN+np.sum(img)+np.sum(lab)
        single_I=np.sum(img*lab)
        single_U=np.sum(img)+np.sum(lab)-single_I
        Dice_.append(2*TP/(FPN))
        Jaccard_.append(single_I/single_U)

        # Calculate clDice
        clDice_.append(clDice_metric(img, lab))

        # Without calculate Betti number/matching errors for efficient training

        # Calculate connectivity
        ccs, _ = cv2.connectedComponents(img)
        ccsg, _ = cv2.connectedComponents(lab)
        numSg = np.count_nonzero(lab)
        C = 1 - min(abs(ccsg-ccs)/numSg, 1)
        C_.append(C)

        # Calculate area
        kernel=np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], dtype=np.uint8)
        SDil  = cv2.dilate(img, kernel)
        SgDil = cv2.dilate(lab, kernel)
        Anum = (SDil & lab) | (img & SgDil)
        Aden = img | lab
        A = np.count_nonzero(Anum)/np.count_nonzero(Aden)
        A_.append(A)

        # Calculate length
        SSkel  = morphology.skeletonize(img).astype(np.uint8)
        SgSkel = morphology.skeletonize(lab).astype(np.uint8)
        SDil  = cv2.dilate(SSkel, kernel)
        SgDil = cv2.dilate(SgSkel, kernel)
        lnum = (SSkel & SgDil) | (SDil & SgSkel)
        lden = SSkel | SgSkel
        L = np.count_nonzero(lnum)/np.count_nonzero(lden)
        L_.append(L)
        
        CAL = C*A*L
        CAL_.append(CAL)

    print("\nDice: %.4f ± %.4f" % (np.mean(Dice_)*100,np.std(Dice_)))
    print("Jac: %.4f ± %.4f" % (np.mean(Jaccard_)*100,np.std(Jaccard_)))
    print("clDice: %.4f ± %.4f" % (np.mean(clDice_)*100,np.std(clDice_)))
    print("C: %.4f" % (np.mean(C)*100))
    print("A: %.4f" % (np.mean(A)*100))
    print("L: %.4f" % (np.mean(L)*100))
    print("CAL: %.4f ± %.4f" % (np.mean(CAL_)*100,np.std(CAL_)))

    return np.mean(Dice_)


def cal_metrics(data_type, pre_path, GT_path):
    TP=FPN=0
    Dice_=[]
    Jaccard_=[]
    clDice_=[]
    C_=[]
    A_=[]
    L_=[]
    CAL_=[]
    Betti_NE_0_=[]
    Betti_NE_1_=[]
    Betti_ME_0_=[]
    Betti_ME_1_=[]
    
    image_files = sorted(glob.glob(pre_path + "/*.*"))
    label_files = sorted(glob.glob(GT_path + "/*.*"))

    for i, image_file in enumerate(image_files):
        pre_file_path =image_file
        true_file_path=label_files[i]

        if data_type == "Fundus":
            img = Image.open(pre_file_path)
            lab = Image.open(true_file_path)
            img = img.resize([512,512])
            lab = lab.resize([512,512])
            img = (np.array(img,dtype=np.uint8)).clip(max=1)
            lab = (np.array(lab,dtype=np.uint8)).clip(max=1)
        elif data_type == "OCTA":
            img = cv2.imread(pre_file_path, cv2.IMREAD_GRAYSCALE)
            lab = cv2.imread(true_file_path,cv2.IMREAD_GRAYSCALE)
            # lab = cv2.resize(lab, (384,384)) # OCTA-500
            img = img.clip(max=1)
            lab = lab.clip(max=1)
        else:
            raise Exception("Invalid data type!", data_type)
        
        if np.max(lab)==255:
            img[img<255] = 0
            lab[lab<255] = 0
        elif np.max(lab)==1:
            img[img<1] = 0
            lab[lab<1] = 0

        # Calculate Dice + Glob Jaccard
        TP=TP+np.sum(img*lab)
        FPN=FPN+np.sum(img)+np.sum(lab)
        single_I=np.sum(img*lab)
        single_U=np.sum(img)+np.sum(lab)-single_I
        Dice_.append(2*TP/(FPN))
        Jaccard_.append(single_I/single_U)

        # Calculate clDice
        clDice_.append(clDice_metric(img, lab))
        
        # Calculate Betti number/matching errors
        H,W = img.shape
        for i in range(0, H, 64):
            for j in range(0, W, 64):
                img_patch = img[i:i+64, j:j+64]
                lab_patch = lab[i:i+64, j:j+64]
                Betti_NE_0, Betti_NE_1, Betti_ME_0, Betti_ME_1 = Betti_number(img_patch,lab_patch)
                Betti_NE_0_.append(Betti_NE_0)
                Betti_NE_1_.append(Betti_NE_1)
                Betti_ME_0_.append(Betti_ME_0)
                Betti_ME_1_.append(Betti_ME_1)
        
        # Calculate connectivity
        ccs, _ = cv2.connectedComponents(img)
        ccsg, _ = cv2.connectedComponents(lab)
        numSg = np.count_nonzero(lab)
        C = 1 - min(abs(ccsg-ccs)/numSg, 1)
        C_.append(C)

        # Calculate area
        kernel=np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], dtype=np.uint8)
        SDil  = cv2.dilate(img, kernel)
        SgDil = cv2.dilate(lab, kernel)
        Anum = (SDil & lab) | (img & SgDil)
        Aden = img | lab
        A = np.count_nonzero(Anum)/np.count_nonzero(Aden)
        A_.append(A)

        # Calculate length
        SSkel  = morphology.skeletonize(img).astype(np.uint8)
        SgSkel = morphology.skeletonize(lab).astype(np.uint8)
        SDil  = cv2.dilate(SSkel, kernel)
        SgDil = cv2.dilate(SgSkel, kernel)
        lnum = (SSkel & SgDil) | (SDil & SgSkel)
        lden = SSkel | SgSkel
        L = np.count_nonzero(lnum)/np.count_nonzero(lden)
        L_.append(L)
        
        CAL = C*A*L
        CAL_.append(CAL)

    print("\nDice: %.4f ± %.4f" % (np.mean(Dice_)*100,np.std(Dice_)*100))
    print("Jac: %.4f ± %.4f" % (np.mean(Jaccard_)*100,np.std(Jaccard_)*100))
    print("clDice: %.4f ± %.4f" % (np.mean(clDice_)*100,np.std(clDice_)*100))
    print("C: %.4f ± %.4f" % (np.mean(C_)*100,np.std(C_)*100))
    print("A: %.4f ± %.4f" % (np.mean(A_)*100,np.std(A_)*100))
    print("L: %.4f ± %.4f" % (np.mean(L_)*100,np.std(L_)*100))
    print("CAL: %.4f ± %.4f" % (np.mean(CAL_)*100,np.std(CAL_)*100))
    print("BNE 0: %.4f ± %.4f" % (np.mean(Betti_NE_0_),np.std(Betti_NE_0_)))
    print("BNE 1: %.4f ± %.4f" % (np.mean(Betti_NE_1_),np.std(Betti_NE_1_)))
    print("BME 0: %.4f ± %.4f" % (np.mean(Betti_ME_0_),np.std(Betti_ME_0_)))
    print("BME 1: %.4f ± %.4f" % (np.mean(Betti_ME_1_),np.std(Betti_ME_1_)))
