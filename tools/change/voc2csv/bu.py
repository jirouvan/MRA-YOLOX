# @TIME  :2019/7/20 21:48
# @File  :voc2csv.py


import os
import xml.dom.minidom

path_img = "S:\yolox-pytorch-main\VOCdevkit\VOC2007\JPEGImages"
path_xml = "./Annotations"

xml_list = []
for xml1 in os.listdir(path_xml):
    if xml1.endswith(".xml"):
        xml_list.append(xml1)

csv_labels = open("csv_labels.csv", "w")
csv_labels.write(
            'filename' + ","
            + 'xmin' + ","
            + 'ymin' + ","
            + 'width' + ","
            + 'height' + ","
            + 'attribution' + ","
            + 'photowidth' + ","
            + 'photoheight' + "\n")
for xml_file in xml_list:
    print(xml_file)
    image, ext = os.path.splitext(xml_file)
    abspath_img = os.path.abspath(path_img + "/" + image + ".jpg")

    DomTree = xml.dom.minidom.parse(path_xml + "/" + xml_file)
    annotation = DomTree.documentElement
    objectlist = annotation.getElementsByTagName('object')
    sizelist = annotation.getElementsByTagName('size')
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        # print("namelist:", namelist)
        objectname = namelist[0].childNodes[0].data
        print("abspath:", abspath_img)
        print("objectname:", objectname)
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            print(x1, y1, x2, y2)

        for size in sizelist:
            width = size.getElementsByTagName('width')
            height = size.getElementsByTagName('height')
            sizew = width[0].childNodes[0].data
            sizeh = height[0].childNodes[0].data
        w=str(x2-x1)
        h=str(y2-y1)
        csv_labels.write(
            image + '.jpg' + "," + str(x1) + "," + str(y1) + "," + w + "," + h + ","
            + objectname + "," + sizew + "," + sizeh + "\n")



csv_labels.close()