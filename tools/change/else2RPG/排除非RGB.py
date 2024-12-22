from PIL import Image
import os

path = 'S:/changebackbone/mmdet/data/coco/train2017/'  # 图片目录
for file in os.listdir(path):

    extension = file.split('.')[-1]
    if extension == 'jpg':
        fileLoc = path + file
        img = Image.open(fileLoc)
        if img.mode != 'RGB':
            print(file + ', ' + img.mode)
            my_file = path + file
            img.close()
            os.remove(my_file)
    elif extension == 'png':
        fileLoc = path + file
        img = Image.open(fileLoc)
        if img.mode != 'RGB':
            print(file + ', ' + img.mode)
            my_file = path + file
            img.close()
            os.remove(my_file)
    elif extension == 'PNG':
        fileLoc = path + file
        img = Image.open(fileLoc)
        if img.mode != 'RGB':
            print(file + ', ' + img.mode)
            my_file = path + file
            img.close()
            os.remove(my_file)
    elif extension == 'JPG':
        fileLoc = path + file
        img = Image.open(fileLoc)
        if img.mode != 'RGB':
            print(file + ', ' + img.mode)
            my_file = path + file
            img.close()
            os.remove(my_file)
print('Done')