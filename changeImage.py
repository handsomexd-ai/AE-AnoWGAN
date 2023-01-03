# coding=utf-8
# @Time : 2022/9/29 16:57
# @Author : XiaoDong
# @File : changeImage.py
# @Software : PyCharm
from PIL import Image  # python3安装pillow库
import os.path
import glob


def convertSize(jpgfile, outdir, width=256, height=256):  # 图片的大小256*256
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        '''
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        '''
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("../IMS/IMS-1st-bear_4-channel_8/*.jpeg"):  # 修改该文件夹下的jpg图片
    convertSize(jpgfile, "./IMS Dataset bearing4_1st&2nd/dataset/test/1")  # 另存为的文件夹路径
