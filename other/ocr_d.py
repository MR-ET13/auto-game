import ddddocr
import cv2
import numpy as np

# 读取图片
img = cv2.imread("debug_roi_clean_one.png", 0)

# =============================================
# 🔥 核心：加粗负号，让 ddddocr 能识别出来
# =============================================
kernel = np.ones((2, 3), np.uint8)  # 横向加粗
img_enhanced = cv2.dilate(img, kernel, iterations=1)

# 保存临时增强图
cv2.imwrite("temp_enhanced.png", img_enhanced)

# 开始识别
ocr = ddddocr.DdddOcr()
with open("temp_enhanced.png", "rb") as f:
    res = ocr.classification(f.read())

print("✅ 最终结果：", res)  # 输出：-11