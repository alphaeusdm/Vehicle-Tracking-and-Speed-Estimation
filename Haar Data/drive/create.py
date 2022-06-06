import os
import cv2
import numpy as np

def create_negative_description_file(path,file_name):
    for image in os.listdir(path):
        line = path+image
        with open(file_name,'a') as f:
            f.write(line)
            f.write('\n')



def create_positive_description_file(path,file_name):
    #image_names = os.listdir(path)
    for image in os.listdir(path):
        line = path+image+' 1 0 0 64 64'
        with open(file_name,'a') as f:
            f.write(line)
            f.write('\n')
            

create_positive_description_file('D:/drive/pos/','info.txt')
           
create_negative_description_file('D:/drive/neg/','bg.txt')


