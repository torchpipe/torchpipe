import cv2
import numpy as np

target = 320
# 定义 720p 分辨率（宽×高）
width = target
height = target

# 生成随机 RGB 图片（形状：[高度, 宽度, 3]，3 为 RGB 三通道）
# np.random.randint 生成 0-255 的随机整数（uint8 类型）
img = cv2.imread(".././encode_jpeg/grace_hopper_517x606.jpg")
random_image = cv2.resize(img, (width, height))
# 保存图片到本地（自动转换为 BGR 格式，因 OpenCV 默认读取为 BGR）
# 若需 RGB 格式保存，需先转换：cv2.cvtColor(random_image, cv2.COLOR_RGB2BGR)
save_path = f"../{target}/{target}.jpg"
cv2.imwrite(save_path, random_image)

print(f"{target} 图片已保存至：{save_path}")
