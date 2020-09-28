import cv2
import numpy as np
from PIL import Image
import os

def CropImage4File(filepath,destpath):
    pathDir =  os.listdir(filepath)   
    for allDir in pathDir:
        a, b = os.path.splitext(allDir)
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
            h,w=image.shape[:2]
            cv2.rectangle(image, (25, 131), (67, 184), (0, 0, 255), 2)
            cv2.imwrite(destpath +a+b, image)
            cropImage = image[131:184,25:67]          
            height,width = cropImage.shape[:2] 
            enhanced_image = cv2.resize(cropImage,(3*width,3*height),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(croppath+a+b,enhanced_image)

def cover():
    IMAGE_PATH1=r'E:/project/image2/'
    IMAGE_PATH2=r'E:/project/image3/'
    IMAGE_FORAMT={'.jpg'}
    IMAGE_SAVE_PATH=r'E:/project/image4/'

    image_names1=[name for name in os.listdir(IMAGE_PATH1)]
    image_names1.sort()

    image_names2=[name for name in os.listdir(IMAGE_PATH2)]
    image_names2.sort()
    image_number=len(image_names1)

    for i in range(image_number):
 
        image=Image.open(os.path.join(IMAGE_PATH1+'\\'+image_names1[i]))
        w=image.size[0]
        h=image.size[1]
   
        toImage = Image.new('RGB', (w, h))
        toImage.paste(image,(0,0))
        
        enhanced_image=Image.open(os.path.join(IMAGE_PATH2+'\\'+image_names2[i]))
        width=enhanced_image.size[0]
        height=enhanced_image.size[1]
        toImage.paste(enhanced_image,(w-width,h-height,w,h))

        toImage.save(IMAGE_SAVE_PATH+'{}.png'.format(i))
           
if __name__ == '__main__':
    filepath =r'E:/project/image1/' 
    destpath=r'E:/project/image2/'
    croppath=r'E:/project/image3/'
    CropImage4File(filepath,destpath)
    cover()