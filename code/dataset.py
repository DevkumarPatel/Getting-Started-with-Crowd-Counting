from ctypes import resize
import os 
import cv2
import glob
import h5py 
import tqdm
import torch 
import random
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.cm as CM 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from scipy.ndimage.filters import gaussian_filter

from model import CSRMSA

def generate_h5py_density( image_path, matrix_path, h5_path ):
    """ Generates a density map file in h5py format. 
    Args:
        image_path (str) : The path to the set of images corresponding to matrix file
        matrix_path (str): The file paths containing .mat 
        h5_path (str): The output folder for all h5_files with format *.h5   
    Returns:
        List of image paths. 
    Example:
        train_data_list = generate_h5py_density(  
            'ShanghaiTech/part_A_final/train_data/images', 
            'ShanghaiTech/part_A_final/train_data/ground_truth', 
            'ShanghaiTech/part_A_final/train_data/h5'   
        )
    References:
        Adopted from CSRNet pytorch version
        Github: 
    """
    images = glob.glob( image_path+"/*.jpg")
    for image in tqdm.tqdm(images):
        img = plt.imread(image)
        np_img = np.zeros( (img.shape[0],img.shape[1]) )
        mat =  sio.loadmat( (matrix_path +"/"+ os.path.basename(image).replace('jpg', 'mat').replace('IMG_', 'GT_IMG_')) )
        gt_coord = mat["image_info"][0,0][0,0][0]
        for i in range( 0, len(gt_coord) ):
            if int(gt_coord[i][1]) < img.shape[0]:                              # checks if the coordinates given are under image dimensions
                if int(gt_coord[i][0]) < img.shape[1]:
                    np_img[ int(gt_coord[i][1]), int(gt_coord[i][0]) ]  = 1     # Populate the cell with 1. 
        np_img = gaussian_filter( np_img, sigma=10 )                            # Run a Gausian filter
        with h5py.File( (h5_path +"/"+ os.path.basename(image).replace('jpg','h5')), 'w' ) as h_file:
            h_file['density'] = np_img
    return images 

def display_density(path):
    """
    Loads a h5py file with dict key 'density'
    and then displays it using matplotlib.
    Args:
        path (str) : the file path to h5py file
    Example:
        img_gaus_density = display_density(
            "ShanghaiTech/part_A_final/train_data/h5/IMG_1.h5")
    Returns:
        np.array of 'density'
    """
    gt = h5py.File(path, 'r')
    density = np.asarray(gt['density'])
    plt.title("Count: " + str(int( density.sum()) ) )
    plt.imshow(density,cmap=CM.jet)
    return density

def predict_size(x):
    #x = (x+1)//2
    #x = (x+1)//2
    #x = (x+1)//2
    #x = (x+1)//2
    return x // 8

def load_data(img_path,model, train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])
    crop_size = (int(img.size[0]), int(img.size[1]))
    if train:
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)

        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    target = cv2.resize(target,( predict_size(target.shape[1]) ,predict_size(target.shape[0])),interpolation = cv2.INTER_CUBIC)*64
    
    return img,target

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, model=CSRMSA()):
        
        # if train:
        #     root = root *4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.model = model 
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.model, self.train)
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target