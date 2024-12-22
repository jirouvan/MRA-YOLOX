from PIL import Image
#-- coding: UTF-8 --

work_path = r'S:/changebackbone/mmdet/data/coco/train2017/train'
work_path_new = r"S:/changebackbone/mmdet/data/coco/train2017/train"
count = "200"

for i in range(200,853):
    print(work_path)
    im = Image.open(work_path+"\\"+count+".jpg")
    print(im)

    image=im.convert('RGB')

    image.save(work_path_new+"\\"+count+".jpg")

    count = int(count)
    count = count + 1
    count = str(count)


