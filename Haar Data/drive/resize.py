from PIL import Image
import os, sys
import cv2

path = "D:/drive/test.jpg"
#dirs = os.listdir( path )

im = Image.open(path)
img = cv2.imread(path)
im_resized = cv2.resize(img, (1280, 720))
cv2.imwrite(path,im_resized)

'''def resize():
    print("hello")
    for item in dirs:
    	im = Image.open(path+item)
    	img = cv2.imread(path+item,cv2.IMREAD_GRAYSCALE)
    	im_resized = cv2.resize(img, (60, 60))
    	cv2.imwrite(path+item,im_resized)
resize()'''


#if os.path.isfile(path+item):
         #   im = Image.open(path+item)
          #  f, e = os.path.splitext(path+item)
           # imResize = im.resize((200,200), Image.ANTIALIAS)
           # imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

        #   m = Image.open(item)
		#im_resized = im.resize((200,200), Image.ANTIALIAS)
		#im_resized.save(item,"JPEG")"""