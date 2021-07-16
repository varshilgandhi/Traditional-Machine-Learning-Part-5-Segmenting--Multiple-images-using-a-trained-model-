# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:19:39 2021

@author: abc
"""

#PART 5 :SEGMENTING MULTIPLE IMAGES USING A SAVED MODEL

import numpy as np
import cv2
import pandas as pd
 

#Create a function

def feature_extraction(img):
    df = pd.DataFrame()
    #Multiple images can be used for training. For that, you need to concatenate the data
    #Save original image pixels into a data frame. This is our Feature #1.
    img2 = img.reshape(-1)
    df['Original Image'] = img2


#Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    #print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                
########################################
#Gerate OTHER FEATURES and add them to the data frame
                
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    
    #VARIANCE with size=3
    #variance_img = nd.generic_filter(img, np.var, size=3)
        #variance_img1 = variance_img.reshape(-1)
        #df['Variance s3'] = variance_img1  #Add column to original dataframe
        
    return df    
        
#####################################################################################

import glob
import pickle
from matplotlib import pyplot as plt


#give one filename
filename = 'sandstone_model'

#load that file into read and binary mode
load_model = pickle.load(open(filename, 'rb'))

#give it one folder to put this model
path = 'train_images/*.tif'

for file in glob.glob(path):
    #read our all images
    img1 = cv2.imread(file)
    #convert image into gray color
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #feature extraction
    X = feature_extraction(img)
    #Predict our extracted model
    result = load_model.predict(X)
    #Segmented our data
    segmented = result.reshape((img.shape))
    #split our all images
    name = file.split("e_")
    #Let's visualize our all images
    plt.imsave('C:\\Users\\abc\\Desktop\\Digital Sreeni\\Image Segmentation using Traditional machine learning\\segmented'+name[1], segmented, cmap='jet')
    
    
########################################################################################


                 #THANK YOU
                 












