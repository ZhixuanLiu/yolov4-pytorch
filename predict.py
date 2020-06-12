#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image

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
        r_image = yolo.detect_image(image)
        #r_image.show()
        img_np = np.array(r_image)
        print ( img_np.shape )
