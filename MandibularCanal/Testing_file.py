#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[27]:


from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Input
from keras.layers import MaxPooling3D, UpSampling3D, concatenate, Conv3D, Cropping3D, ZeroPadding3D, Activation
from keras.layers import Conv3DTranspose as Deconvolution3D
from keras.models import Model

import numpy as np
import glob as glob
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import json


# In[6]:


def get_full_scan_old(folder_path):

    files_List  = glob.glob(folder_path + '/*')
    z_max = len(files_List)
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    full_scan = np.ndarray(shape=(z_max,rows,cols), dtype=float, order='F')

    for file in tqdm(files_List):
        img, n = dcm_image(file)
        n = int(n)
        full_scan[n-1,:,:] = img[0,:,:]

    return full_scan
    
def get_full_scan(folder_path):

    files_List  = glob.glob(folder_path + '/*.dcm')
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    mn = 1000
    mx = 0
    for file in tqdm(files_List):
        itkimage = sitk.ReadImage(file)
        mn = np.min([mn, int(itkimage.GetMetaData('0020|0013'))])
        mx = np.max([mx, int(itkimage.GetMetaData('0020|0013'))])
    full_scan = np.ndarray(shape=(mx-mn+1,rows,cols), dtype=float, order='F')

    for file in tqdm(files_List):
        img, n = dcm_image(file)
        n = int(n)
        full_scan[mn - n,:,:] = img[0,:,:]

    return full_scan


# In[7]:


def dcm_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
#     numpyImage = get_normalized(numpyImage,wc,wl)
    return numpyImage, float(itkimage.GetMetaData('0020|0013'))

def get_normalized(scan,mn,mx):
    np.clip(scan, mn, mx, out=scan)
    d = mx - mn
    scan = (scan-mn)/d
    return scan


# In[8]:


def normalize_3D_scan(scan):
    nscan = np.zeros((512, 512, scan.shape[2]), 'uint8')
    for i in tqdm(range(scan.shape[2])):
        s = scan[:,:,i]
        s = cv2.resize(s, (512, 512), interpolation=cv2.INTER_CUBIC)
        nscan[:,:,i] = s
    nnscan = np.zeros((512, 512, 512), 'uint8')
    
    for i in range(512):
        s = nscan[i,:,:]
        s = cv2.resize(s, (512, 512), interpolation=cv2.INTER_CUBIC)
        nnscan[i,:,:] = s
        
    return nnscan

def denormalize_3D_scan(mask, s):
    (x, y, z) = s
    nscan = np.zeros((x, y, 512), 'uint8')
    for i in tqdm(range(512)):
        s = mask[:,:,i]
        s = cv2.resize(s, (y, x), interpolation=cv2.INTER_NEAREST)
        nscan[:,:,i] = s
    nnscan = np.zeros((x, y, z), 'uint8')
    
    for i in range(x):
        s = nscan[i,:,:]
        s = cv2.resize(s, (z, y), interpolation=cv2.INTER_NEAREST)
        nnscan[i,:,:] = s
        
    return nnscan


# In[39]:


def get_mask(mask_path,scan_shape):
    mask_path = glob.glob(mask_path+'/*.json')
    mask_path = mask_path[0]
    with open(mask_path, 'r') as f:
        data = json.load(f)
    array = np.zeros((scan_shape))
    X = []
    Y = []
    Z = []
    R = len(data['annotation']['tooth']['Right']['coordinate'])
    Ones = data['annotation']['tooth']['Right']['coordinate']
    curr = 0
    while(curr<R):
        X= int(Ones[curr])
        curr = curr+1
        Y = int(Ones[curr])
        curr = curr+1
        Z = int(Ones[curr])
        curr = curr+1
        array[Z,Y,X]=1
    R = len(data['annotation']['tooth']['Left']['coordinate'])
    Ones = data['annotation']['tooth']['Left']['coordinate']
    curr = 0
    while(curr<R):
        X= int(Ones[curr])
        curr = curr+1
        Y = int(Ones[curr])
        curr = curr+1
        Z = int(Ones[curr])
        curr = curr+1
        array[Z,Y,X]=1
    return(array)


# In[19]:


def get_Filename(file):
    file = file.split('/')[-1]
    file = file.split('.')[0]
    return file
    
def make_full_contour(img,mask):
    s = img[:, 0:256,:]
    s_mips = np.amax(s, axis=1)
    s_mips = s_mips/s_mips.max()
    m = mask[:, 0:256,:]
    m_mips = np.amax(m, axis=1)
    con_ones = m_mips
    con_ones_L = np.zeros_like(m_mips)
    con_ones_R = np.zeros_like(m_mips)
#     con_ones_L[:,:255] = 0
#     con_ones_R[:,256:] = 0
    con_zeros = 1 - con_ones
    con_ones_L[:,256:] = m_mips[:,256:]
    con_ones_R[:,:255] = m_mips[:,:255]
    
    RGB_img = np.zeros([s_mips.shape[0],s_mips.shape[1],3],int)
    RGB_img[:,:,0] = s_mips*con_zeros*255 + con_ones_L*255
    RGB_img[:,:,2] = s_mips*con_zeros*255 + con_ones_R*255
    RGB_img[:,:,1] = s_mips*con_zeros*255    
    
    return RGB_img


# In[20]:


def unet(pretrained_weights = None):
    
    in_layer = Input((None, None, None, 1))
    
    bn = BatchNormalization()(in_layer)
    cn1 = Conv3D(8, 
                 kernel_size = (1, 5, 5), 
                 padding = 'same',
                 activation = 'relu')(bn)
    cn2 = Conv3D(8, 
                 kernel_size = (3, 3, 3),
                 padding = 'same',
                 activation = 'linear')(cn1)
    bn2 = Activation('relu')(BatchNormalization()(cn2))

    dn1 = MaxPooling3D((2, 2, 2))(bn2)
    cn3 = Conv3D(16, 
                 kernel_size = (3, 3, 3),
                 padding = 'same',
                 activation = 'linear')(dn1)
    bn3 = Activation('relu')(BatchNormalization()(cn3))

    dn2 = MaxPooling3D((1, 2, 2))(bn3)
    cn4 = Conv3D(32, 
                 kernel_size = (3, 3, 3),
                 padding = 'same',
                 activation = 'linear')(dn2)
    bn4 = Activation('relu')(BatchNormalization()(cn4))

    up1 = Deconvolution3D(16, 
                          kernel_size = (3, 3, 3),
                          strides = (1, 2, 2),
                         padding = 'same')(bn4)

    cat1 = concatenate([up1, bn3])

    up2 = Deconvolution3D(8, 
                          kernel_size = (3, 3, 3),
                          strides = (2, 2, 2),
                         padding = 'same')(cat1)

    pre_out = concatenate([up2, bn2])

    pre_out = Conv3D(1, 
                 kernel_size = (1, 1, 1), 
                 padding = 'same',
                 activation = 'sigmoid')(pre_out)

    pre_out = Cropping3D((1, 2, 2))(pre_out) # avoid skewing boundaries
    out = ZeroPadding3D((1, 2, 2))(pre_out)
    sim_model = Model(inputs = [in_layer], outputs = [out])
    if(pretrained_weights):
        sim_model.load_weights(pretrained_weights)
    return sim_model


# In[21]:


def get_canal_segmentation(scan):
    # Normalize the scan 
    scan = get_normalized(scan, -500, 1500)*255
    scan = scan.astype('uint8')  
    (x,y,z) = scan.shape
    flg = 0
    if scan.shape == (512, 512, 512):
        pass
    else:
        flg = 1
        scan = normalize_3D_scan(scan)

    # Extract the VOIs for left and right 
    scan_left = scan[256:480,58:298,255:455]
    scan_left = scan_left/255
    scan_left = scan_left.astype(np.float32)

    scan_right = scan[248:480,54:302,78:302]
    scan_right = scan_right/255
    scan_right = scan_right.astype(np.float32)

    scan_left = np.expand_dims(scan_left, axis=0)
    scan_left = np.expand_dims(scan_left, axis=-1)

    scan_right = np.expand_dims(scan_right, axis=0)
    scan_right = np.expand_dims(scan_right, axis=-1)

    # Load the models (left and right)
    left_model = unet(pretrained_weights = 'left_side_weights.hdf5')
    right_model = unet(pretrained_weights = 'right_side_weights.hdf5')
    # Get the segmentation for left and right side
    left_seg = left_model.predict(scan_left)
    left_seg = np.where(left_seg>0.01, 1, 0)
    left_seg = np.squeeze(left_seg)
    right_seg = right_model.predict(scan_right)
    right_seg = np.where(right_seg>0.01, 1, 0)
    right_seg = np.squeeze(right_seg)
    # Merge the Predictions (left + right)
    mask = np.zeros((512,512,512))
    mask[256:480,58:298,255:455] = left_seg
    mask[248:480,54:302, 78:302] = right_seg

    # Denormalize the merged mask 
    if flg == 1:
        mask = denormalize_3D_scan(mask, (x, y, z))
    return mask


# In[22]:


def dice_coefficient(predicted, target):
    smooth = 1
    product = np.multiply(predicted, target)
    intersection = np.sum(product)
    coefficient = (2 * intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
    return coefficient


# In[23]:


def get_IOU(predicted, target):
    smooth = 1
    intersection = np.sum(np.multiply(predicted, target))
    union = np.sum(predicted)+np.sum(target) - intersection
    coefficient = (intersection + smooth) / (union + smooth)
    return coefficient


# In[24]:


def dice_metric(predicted, Groundtruth):
    TP = predicted*Groundtruth
    FP = predicted - TP
    FN = Groundtruth - TP
    precision = np.sum(TP)/(np.sum(TP)+np.sum(FP))
    recall = np.sum(TP)/(np.sum(TP)+np.sum(FN))
    f1_score = 2*((precision*recall)/(precision+recall))
    dice_score = dice_coefficient(predicted, Groundtruth)
    IOU = get_IOU(predicted, Groundtruth)
    return [precision,recall,f1_score,dice_score,IOU]


# In[43]:


def write_to_file(scan_path,mask_path):
    scan = get_full_scan(scan_path)
    scan = np.flip(scan,axis=0)
    mask_pred = get_canal_segmentation(scan)
    scan_shape=scan.shape
    mask_GT = get_mask(mask_path,scan_shape)
    print(mask_GT.shape)
    scores = dice_metric(mask_GT, mask_pred)
    if(os.path.isfile("Results.csv")):
        with open("Results.csv", "a") as myfile:
            myfile.write(str(scan_path)+','+str(dice_metric(mask_GT, mask_pred))[1:-1]+'\n')
    else:
        with open("Results.csv", "w") as myfile:
            myfile.write("Scan_path,Precision,Recall,F1_score,DiceScore,IOU"+'\n')
            myfile.write(str(scan_path)+','+str(dice_metric(mask_GT, mask_pred))[1:-1]+'\n')
    a = np.where(mask_pred>0)
    ones = []
    for i in range(len(a[0])):
        ones.append(float(a[0][i]))
        ones.append(float(a[1][i]))
        ones.append(float(a[2][i]))
    dictionay = {
        'Mask_shape' : scan.shape,
        'Mask_indes' : ones
    }
    with open(scan_path.split('/')[-1]+'.json', 'w') as fp:
        json.dump(dictionay, fp)
    return(scores)


# In[1]:


import argparse


# In[3]:


parser = argparse.ArgumentParser(description='Keras DenseUnet Test')
parser.add_argument('-scan_path', type=str, default='left', help='path for input scan')
parser.add_argument('-mask_path', type=str, default='/livermask/',help='path for input annotation')
args = parser.parse_args()


# In[ ]:


if __name__ == '__main__':
    write_to_file(arg.scan_path,arg.mask_path)


# In[ ]:




