import easyocr
import cv2
import numpy as np

def easyocr_way(img):
    """" 数字识别 """
    # 读取图片
    # img = cv2.imread(img, 0)

    # 初始化 OCR，只启用英文/数字（速度更快、更准）
    reader = easyocr.Reader(['en'], gpu=True)  # 没有GPU就用 gpu=False

    # 识别图片，detail=0 只返回文字内容
    result = reader.readtext(
        img,
        detail=0,
        allowlist='-0123456789',  # 明确包含负号
        paragraph=False  # 允许单独识别符号，再和数字组合
    )

    print("原始识别结果：", result)

    # 手动拼接负号与后续数字（处理负号被单独识别的情况）
    fixed_result = []
    temp = ""
    for item in result:
        if item == "-":
            temp = "-"
        elif item.isdigit():
            fixed_result.append(temp + item)
            temp = ""
        else:
            if temp:
                fixed_result.append(temp)
                temp = ""
            fixed_result.append(item)
    if temp:
        fixed_result.append(temp)

    print("修复后结果：", fixed_result)
    valid_numbers = [item for item in result if is_valid_number(item)]
    for num in valid_numbers:
        print(f"识别到的数字：{num}")
        return num

# ----------------------
# 过滤：只保留 数字/负数/多位数
# ----------------------
def is_valid_number(s):
    # 允许：负号开头 + 数字，或纯数字
    s = s.strip()
    if s.startswith('-'):
        return s[1:].isdigit()  # 负数
    return s.isdigit()  # 正数

if __name__ == '__main__':
    num = easyocr_way('temp_enhanced.png')
    print(num)