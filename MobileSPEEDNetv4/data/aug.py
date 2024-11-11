from typing import List

import albumentations as A
import numpy as np
import cv2 as cv

def CropAndPadSafe(image: np.ndarray, bbox: np.array, p: float):
    # 对图片进行随机区域裁剪
    # 保留bbox区域
    # 裁剪后padding回原大小
    # 获取图像尺寸
    
    if np.random.random() > p:
        return image
    
    h, w = image.shape[:2]
    
    # 获取bbox信息
    x_min, y_min, x_max, y_max = bbox
    
    # 根据bbox生成随机裁剪大小
    top = np.random.randint(0, y_min) if y_min > 0 else 0
    bottom = np.random.randint(0, h - y_max) if y_max < h else 0
    left = np.random.randint(0, x_min) if x_min > 0 else 0
    right = np.random.randint(0, w - x_max) if x_max < w else 0
    
    # 裁剪图像
    cropped = image[top:h-bottom, left:w-right]
    
    # 填充回原始大小
    padded = cv.copyMakeBorder(cropped, top, bottom, left, right, cv.BORDER_REPLICATE)
    
    return padded



def DropBlockSafe(image: np.ndarray, bbox: np.array, drop_n: int, p: float):
    # 随机丢弃[1, drop_n]个方形块区域
    # 这些区域不包括bbox区域
    
    if np.random.random() > p:
        return image
    
    h, w = image.shape[:2]
    
    # 获取bbox信息
    x_min, y_min, x_max, y_max = bbox
    
    # 随机丢弃[1, drop_n]个方形块区域
    drop_n = np.random.randint(1, drop_n)
    
    background = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 随机丢弃drop_n个方形块区域，这些区域不包括bbox区域
    for _ in range(drop_n):
        drop_x_min = np.random.randint(0, w)
        drop_y_min = np.random.randint(0, h)
        drop_x_max = np.random.randint(drop_x_min, w)
        drop_y_max = np.random.randint(drop_y_min, h)
        
        mask[drop_y_min:drop_y_max, drop_x_min:drop_x_max] = 1
        
    mask[y_min:y_max, x_min:x_max] = 0
    image = image * (1 - mask) + background * mask
    
    return image



# 先读取背景图片到内存中，减少IO开销
from pathlib import Path

current_path = Path(__file__).parent.resolve()
background_images = []
background_sizes = []

for i in range(1, 8):
    img_path = str(current_path / f"background/{i}.jpg")
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    background_images.append(img)
    background_sizes.append(img.shape[:2])

def CropAndPaste(image: np.ndarray, bbox: np.array, p: float):
    # 随机裁剪bbox区域，并粘贴到新图像背景中对应的位置
    
    if np.random.random() > p:
        return image
    
    target = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    idx = np.random.randint(0, len(background_images))
    background = background_images[idx]
    h, w = background_sizes[idx]
    
    # 随机裁剪一下背景，最小的宽度为image的1/2，最大宽度为background的宽度；最小高度为image的1/2，最大高度为background的高度
    min_h, min_w = image.shape[0] // 2, image.shape[1] // 2
    max_h, max_w = background.shape[0], background.shape[1]
    crop_h = np.random.randint(min_h, max_h + 1)
    crop_w = np.random.randint(min_w, max_w + 1)
    h_start = np.random.randint(0, h - crop_h + 1)
    w_start = np.random.randint(0, w - crop_w + 1)
    background = background[h_start:h_start+crop_h, w_start:w_start+crop_w]
    
    # 将background缩放至image大小
    background = cv.resize(background, (image.shape[1], image.shape[0]))
    
    # 将target粘贴到background中
    background[bbox[1]:bbox[3], bbox[0]:bbox[2]] = target
    
    return background



from scipy.spatial.transform import Rotation as R
fwx = 0.0176  # focal length[m]
fwy = 0.0176  # focal length[m]
ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
ppy = ppx  # vertical pixel pitch[m / pixel]
fx = fwx / ppx  # horizontal focal length[pixels]
fy = fwy / ppy  # vertical focal length[pixels]
K = np.array([[fx, 0, 1920 / 2], [0, fy, 1200 / 2], [0, 0, 1]])
K_inv = np.linalg.inv(K)

def RotateImage(image: np.ndarray, pos: np.array, ori: np.array, bbox: np.array, max_rotate_angle: float, limit_num: int, p: float):
    # 绕Z轴旋转图像
    
    if np.random.random() > p:
        return image, pos, ori, bbox
    
    h, w = image.shape[:2]
    original_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    t = 0
    while True:
        change = (np.random.random() - 0.5) * 2 * max_rotate_angle
        
        rotation = R.from_euler('YXZ', [0, 0, change], degrees=True)
        r_change = rotation.as_matrix()
        
        warp_matrix = K @ r_change @ K_inv
        
        # 更新bbox
        bbox_new = wrap_boxes(bbox.reshape(1, 4), warp_matrix, w, h).reshape(4)
        new_area = (bbox_new[2] - bbox_new[0]) * (bbox_new[3] - bbox_new[1])
        if new_area >= 0.7 * original_area:
            break
        else:
            t += 1
            if t > limit_num:
                return image, pos, ori, bbox
    
    image_warped = cv.warpPerspective(image, warp_matrix, (w, h), cv.WARP_FILL_OUTLIERS, flags=cv.INTER_LINEAR)
    
    # 更新pos和ori
    pos_new = r_change @ pos
    ori_new = rotation * R.from_quat([ori[1], ori[2], ori[3], ori[0]])
    ori_new = ori_new.as_quat(canonical=True)
    ori_new = np.concatenate([ori_new[3:], ori_new[:3]])
    
    return image_warped, pos_new, ori_new, bbox_new

def wrap_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))

        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # 所有点，矩阵变换，求变换后的点
        xy = xy @ M.T  # transform

        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width-1)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height-1)
        return xy.astype(np.float32)
    else:
        return boxes



Atransform = A.Compose([
        A.AdvancedBlur(),
        A.ColorJitter(),
        A.GaussNoise(var_limit=(10, 30))
    ])

def AlbumentationAug(image: np.ndarray, p: float):
    # 使用albumentations进行增强
    
    if np.random.random() > p:
        return image
    
    image = Atransform(image=image)["image"]
    
    return image