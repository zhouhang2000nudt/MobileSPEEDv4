# import sys
# # 在模块搜索路径的开头插入新路径
# sys.path.append('..')

from MobileSPEEDNetv4.data.aug import DropBlockSafe, CropAndPadSafe, CropAndPaste, RotateImage, AlbumentationAug, K
from MobileSPEEDNetv4.utils import visualize
import numpy as np
import cv2 as cv
import timeit
import json

image_name = "img007767.jpg"
# image_name = "img000001.jpg"
image_path = f"/home/zh/pythonhub/yaolu/datasets/speed/images/train/{image_name}"
label_path = "/home/zh/pythonhub/yaolu/datasets/speed/train_label.json"

# 测试数据增强函数

# 转为灰度图
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

label = json.load(open(label_path, "r"))
ori = np.array(label[image_name]["ori"])
pos = np.array(label[image_name]["pos"])
bbox = np.array(label[image_name]["bbox"], dtype=np.int32)

# image = CropAndPaste(image, bbox, p=1.0)
image = CropAndPadSafe(image, bbox, p=1.0)
cv.imwrite("cropandpad.jpg", image)
image = DropBlockSafe(image, bbox, 5, p=1.0)
cv.imwrite("dropblock.jpg", image)
image, pos, ori, bbox = RotateImage(image, pos, ori, bbox, 90, 5, p=1.0)
cv.imwrite("rotate.jpg", image)
image = AlbumentationAug(image=image, p=1.0)
cv.imwrite("albumentation.jpg", image)

image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)


visualize(image, bbox.reshape(1, 4), [1], {1: "star"}, ori, pos, K)


# #  测试函数执行时间

# # 定义要测试的函数
# def test_augmentations():
#     CropAndPaste(image, bbox, 0.5)
#     CropAndPadSafe(image, bbox, 0.5)
#     DropBlockSafe(image, bbox, 5, 0.5)
#     RotateImage(image, pos, ori, bbox, 90, 5, 0.8)
#     AlbumentationAug(image, p=1.0)
    

# image = cv.imread(image_path)
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# ori = np.array([-0.197107, 0.364316, 0.328288, 0.848911])
# pos = np.array([0.107809, -0.211386, 9.114421])
# bbox = np.array([850, 447, 1259, 720], dtype=np.int32)
# # 使用timeit测量执行时间
# number = 100  # 执行次数
# time_taken = timeit.timeit(test_augmentations, number=number)

# print(image.shape)
# print(f"平均执行时间: {time_taken / number:.6f} 秒")
# print(f"总执行时间 ({number} 次): {time_taken:.6f} 秒")