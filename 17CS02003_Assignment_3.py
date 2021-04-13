import numpy as np
import pywt
import os
import random
import cv2
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

import math
from numpy.lib.histograms import histogram
from ipython_genutils.py3compat import xrange

image = 'coverImage.jpg'   
watermark = 'scrambledWatermark.jpg'
model ='haar'

def arnold():
    img= cv2.imread('watermark.png')
    ori_img = np.copy(img)
    new_img = np.copy(img)
    length=64
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_i = (i + j) % length
            new_j = (i + 2 * j) % length
            new_img[new_i][new_j] = img[i][j]

    cv2.imwrite('scrambledWatermark.jpg',new_img)

def inverseArnold():
    img = cv2.imread('extractedscrambledwm.jpg')
    ori_img = np.copy(img)
    new_img = np.copy(img)
    length = 64
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_i = (2*i - j) % length
            new_j = (-i +  j) % length
            new_img[new_i][new_j] = img[i][j]
    
    cv2.imwrite('extractedWatermark.png',new_img)

def readImage(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img_gray = img.convert('L')
    img_rgb = img.convert('RGB')
    img_gray.save('./' + image_name)
    image_array = np.array(img_gray.getdata(), dtype=np.float64).reshape((size, size))
    image_array_rgb = np.array(img_rgb.getdata(), dtype=np.float64).reshape((size, size,3))    

    return img, image_array, image_array_rgb

def generateImage(image_array, name):
  
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./' + name)
    return img

def DWT(imArray, model, level):
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    coeffs_H=list(coeffs)   
    return coeffs_H


def DCT(image_array):
    size = len(image_array[0])
    dctArray = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            block = image_array[i:i+8, j:j+8]
            dctBlock = dct(dct(block.T, norm="ortho").T, norm="ortho")
            dctArray[i:i+8, j:j+8] = dctBlock

    return dctArray


def InverseDCT(dctArray):
    size = len(dctArray[0])
    idctArray = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            idctBlock = idct(idct(dctArray[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            idctArray[i:i+8, j:j+8] = idctBlock

    return idctArray


def getWatermarkswithkey(key,length):
    random.seed(key)
    i= [random.random() for _ in xrange(length)]
    return i


def insertWatermark(watermark_array, original_image,key):
    temp = original_image
    watermark_array_size = len(watermark_array[0])
    watermark_flat = watermark_array.ravel()
    ind = 0
    lent =len(watermark_flat)
    j= getWatermarkswithkey(key,lent)
    for i in range(lent):
        watermark_flat[i] = watermark_flat[i]-j[i]

    for x in range (0, len(original_image), 8):
        for y in range (0, len(original_image), 8):
            if ind < len(watermark_flat):
                dctBlock = original_image[x:x+8, y:y+8]
                dctBlock[5][5] = watermark_flat[ind]
                #print(watermark_flat[ind])
                temp[x:x+8, y:y+8] = dctBlock
                ind += 1 
    return temp



def getWatermarkedCoefficients(dctWatermarkedCoefficients, watermark_size,key):    
    watermarkCoeff = []
    for x in range (0, len(dctWatermarkedCoefficients), 8):
        for y in range (0, len(dctWatermarkedCoefficients), 8):
            dctBlock = dctWatermarkedCoefficients[x:x+8, y:y+8]
            watermarkCoeff.append(dctBlock[5][5])
    

    j= getWatermarkswithkey(key,len(watermarkCoeff))
    for i in range(len(watermarkCoeff)):
        watermarkCoeff[i] = watermarkCoeff[i]+j[i]
            
    watermark = np.array(watermarkCoeff).reshape(watermark_size, watermark_size)*255

    return watermark

def recover_watermark(image_array, model, level, wmSize,key):
    #image_array= readImage('WatermarkedImage.png',2048)[2][:,:,2]
    watermarkedCoefficients = DWT(image_array, model, level=level)
    watermarkedCoefficients_l2 = DWT(watermarkedCoefficients[1][0], model, level=level)
    dctWatermarkedCoefficients = DCT(watermarkedCoefficients_l2[0])
    watermark_array = getWatermarkedCoefficients(dctWatermarkedCoefficients, wmSize,key)
    watermark_array =  np.uint8(watermark_array)
    #Save result
    img = Image.fromarray(watermark_array)
    img.save('./extractedscrambledwm.jpg')

def recoverWM(ImageWithWatermark,key):
    recover_watermark(image_array = ImageWithWatermark, model=model, level = 1, wmSize=64,key=key)
    extracted_wm = Image.open('./extractedscrambledwm.jpg').resize((128, 128), 1)
    print('\nc.Extracting scrambled watermark from watermarkedImage.....','done')
    inverseArnold()
    print('\nd.Recovering watermark.....','done')

def psnr(img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        
def NCIndex(img1,img2):
    return abs(np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2)))



def saltnoise(img: np.ndarray):
        img = img.copy()
        for k in range(1000):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img

def rotation(img):
    r, c = img.shape
    rotated = np.rot90(img, k=2, axes=(0, 1))
    rotated = np.reshape(rotated, (r,c))
    return rotated


def main(choice):
    arnold()
    key=123
    print('\na.scrambling Watermark image....','done')
    originalCoverImage, coverImageArray, coverImageArray_RGB = readImage(image, 2048)
    #0,1,2 = b,g,r

    #1. Apply dwt level 1
    dwtCoefficients = DWT(coverImageArray_RGB[:,:,0], model, level=1)
    #Apply dwt on mid frequency sub bands
    dwtCoefficientsL2 = DWT(dwtCoefficients[1][0], model, level=1)
    #print(dwtCoefficientsL2[1][0].__len__())

    #2. Apply dct to corresponding frequency bands from dwt
    dctCoefficients = DCT(dwtCoefficientsL2[0])
    originalWatermark, watermark_array, watermark_rgb = readImage(watermark, 64)
    watermark_array = watermark_array/255

    #embed watermark
    watermarkedImage = insertWatermark(watermark_array, dctCoefficients,key)

    dwtCoefficientsL2_temp = dwtCoefficientsL2
    #inverse dct & inverse dwt
    dwtCoefficientsL2_temp[0][:][:] = InverseDCT(watermarkedImage)
    temp = pywt.waverec2(dwtCoefficientsL2_temp, model)

    #reconstruct 
    dwtCoefficients_temp = dwtCoefficients
    dwtCoefficients_temp[1][0][:][:] = temp
    ImageWithWatermark=pywt.waverec2(dwtCoefficients_temp, model)

    reconstructImage_RGB = np.zeros((2048,2048,3))
    reconstructImage_RGB[:,:,2] = ImageWithWatermark
    reconstructImage_RGB[:,:,0] = coverImageArray_RGB[:,:,0]
    reconstructImage_RGB[:,:,1] = coverImageArray_RGB[:,:,1]


    reconstructedImage = generateImage(reconstructImage_RGB, 'WatermarkedImage.png')
    print('\nb.Generating Image with watermark.....','done')
    if choice == 2:
        ImageWithWatermark = saltnoise(ImageWithWatermark)
        recoverWM(ImageWithWatermark,key)
    elif choice == 3 :
        ImageWithWatermark = rotation(ImageWithWatermark)
        recoverWM(ImageWithWatermark,key)
    elif choice == 1:
        recoverWM(ImageWithWatermark,key)
    else :
        print('\n incorrect choice')
        exit()
    

    origVsStego = psnr(coverImageArray_RGB,reconstructImage_RGB)
    print("\nPSNR -->",origVsStego,'\n')
    print("\nNC Index -->",NCIndex(cv2.imread('watermark.png'),cv2.imread('extractedWatermark.png')),'\n')
    
def FrameCapture(path,folder):
    vidObj = cv2.VideoCapture(path)
    count = 0  
    success = 1
  
    while count<30:
        success, image = vidObj.read()
        cv2.imwrite("/content/drive/My Drive/sflab4/%s/frame%d.jpg" % folder,count, image) 
        count += 1
    
    video.release()

if __name__ == "__main__":
    choice =int(input('\n1.No Attack\n2.saltNoise attck\n3.rotation attack\n\nChoice : '))
    main(choice)
    

