#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np 

yolo = YOLO()

while True:
    #img = input('Input image filename:')
    img = '../input/global-wheat-detection/test/cc3532ff6.jpg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        boxes = yolo.get_boxes(image)
        #r_image = yolo.detect_image(image)
    for i in boxes: 
        print (i) 
    break 
    
