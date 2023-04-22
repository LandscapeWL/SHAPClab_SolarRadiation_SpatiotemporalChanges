from PIL import Image
import numpy as np
import cv2

img_L = np.array(Image.open(r'E:\202208CSAILVion\images_result\11.jpg').convert("L"))
img_RGB = np.array(Image.open(r'E:\202208CSAILVion\images_result\11.jpg').convert("RGB"))

temp = {}
for i in range(img_L.shape[0]):
    for j in range(img_L.shape[1]):
        if not temp.get(int(img_L[i][j])):
            temp[int(img_L[i][j])] = list(img_RGB[i][j])
print(temp)

#这里得到灰度像素值0对应(0,0,0),163对应(6,230,230)
color_0_0_0 = np.where(img_L == 0)[0].shape[0]
color_6_230_230 = np.where(img_L == 163)[0].shape[0]

pixel_sum = img_L.shape[0] * img_L.shape[1] * 0.785

print("0_0_0 像素个数：{} 占比：%{}".format(color_0_0_0,color_0_0_0/pixel_sum*100))
print("6_230_230 像素个数：{} 占比：%{}".format(color_6_230_230,color_6_230_230/pixel_sum*100))
