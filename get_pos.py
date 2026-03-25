import cv2
import numpy as np
import pytesseract
import time
import pyautogui
import c_img
import shutil

# ====================== 【必须配置】修改这里 ======================
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
TEMPLATE_PATH = "number_template.png"  # 你的模板图片

# 【关键修改】单个ROI覆盖两个数字+逗号（需根据实际画面调整）
# 建议先通过debug截图确认ROI能完整包含两个数字和中间的逗号
VALUE_ROI_WIDTH = 136      # 加宽：容纳两个数字+逗号（原55不够，需实测调整）
VALUE_ROI_HEIGHT = 37      # 高度保持（根据数字高度调整）

SINGLE_NUMBER_PIXEL = 17  # 单个数字的像素值用于计算逗号分隔

# =================================================================

# 【关键修改】OCR白名单加入逗号，保留psm 7（单行文本）
OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-, -c preserve_interword_spaces=0'
# OCR_CONFIG_1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-, -c preserve_interword_spaces=0'
OCR_CONFIG_1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'


def clean_background_lines(roi):
    """消除背景浅色干扰横线，只保留白色数字、负号和逗号"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. 强力二值化：过滤浅灰背景，只保留纯白字符
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 2. 形态学：清除细小横线干扰，保留字符轮廓
    kernel = np.ones((1, 2), np.uint8)  # 横向小核清横线
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. 强化负号和逗号（防止被过滤）
    kernel2 = np.ones((2, 1), np.uint8)
    cleaned = cv2.dilate(cleaned, kernel2, iterations=1)

    return cleaned


def capture_screen():
    """截取全屏并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def pos_cal():
    """模板匹配校准基准位置（可选保留，用于动态调整POS_X/POS_Y）"""
    frame = capture_screen()
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print("错误：模板不存在")
        return None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, gray_tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    h, w = gray_tpl.shape[:2]
    if max_val < 0.6:
        print("未找到目标模板")
        return None
    else:
        print(f"模板匹配成功，基准位置：{max_loc}")
        return max_loc[0] + w, max_loc[1]


def find_comma_position(clean_roi):
    """【兜底策略】通过像素特征定位逗号位置，弥补OCR识别失败"""
    # 逗号的像素特征：纵向窄列+中下部有连续白色像素
    roi_h, roi_w = clean_roi.shape
    comma_candidates = []

    # 遍历ROI横向区域（排除首尾数字区，只查中间段）
    for x in range(0, roi_w):
        # 取当前列的像素
        col_pixels = clean_roi[int(roi_h * 0.9), x]
        # 逗号的列白色像素占比阈值（需根据实际画面微调）
        if col_pixels == 255:
            comma_candidates.append(x)

    # 若找到候选列，说明存在逗号
    if len(comma_candidates) > 0:
        return comma_candidates  # 至少3列连续像素符合逗号特征
    else:
        return None


def check_is_negative_8(clean_roi, ocr_text):
    """
    【核心优化3】单数字校验：检测是否是-8被误识别为-48/-4，通过像素特征修正
    :param clean_roi: 预处理后的ROI图像
    :param ocr_text: OCR原始识别结果
    :return: 修正后的文本
    """
    roi_h, roi_w = clean_roi.shape
    # 匹配误识别特征：识别结果是-48/-4，且ROI是单数字区域（无逗号）
    if ocr_text == '-4':
        # 检测8的像素特征：ROI中下部有大面积闭合白色像素（8的特征，4无此特征）
        # 取数字核心区域（排除负号）：横向20%~80%，纵向10%~90%
        eight_core = clean_roi[int(roi_h*0.1):int(roi_h*0.9), SINGLE_NUMBER_PIXEL + 5:SINGLE_NUMBER_PIXEL * 2 + 5]
        # 白色像素占比：8的闭合轮廓占比远高于4
        white_ratio = np.sum(eight_core == 255) / (eight_core.shape[0] * eight_core.shape[1])
        # 经验阈值：8的白色占比一般>0.15，4<0.1（可根据实际画面微调）
        if white_ratio > 0.22:
            print(f"检测到误识别：{ocr_text} → 修正为-8")
            return '-8'
    return ocr_text  # 非误识别，返回原文本


def get_two_numbers_from_single_roi():
    """识别单个ROI内用逗号分隔的两个数字"""
    # 1. 计算ROI区域（覆盖两个数字+逗号）
    x1, y1 = pos_cal()
    frame = capture_screen()
    roi = frame[y1:y1 + VALUE_ROI_HEIGHT, x1:x1 + VALUE_ROI_WIDTH]

    # 2. 预处理ROI（去干扰、强化字符）
    clean_roi = clean_background_lines(roi)

    # 保存调试截图（确认ROI是否覆盖完整、预处理是否清晰）
    cv2.imwrite("debug_roi_full.png", roi)        # 原始ROI
    cv2.imwrite("debug_roi_clean.png", clean_roi) # 预处理后ROI

    # 3. OCR识别（包含逗号、负号、数字）
    ocr_text = pytesseract.image_to_string(clean_roi, config=OCR_CONFIG).strip()
    print(f"OCR原始识别结果：{ocr_text}")

    # 4. 【核心优化】逗号丢失的兜底处理
    has_comma_in_ocr = ',' in ocr_text
    has_comma_in_pixel = find_comma_position(clean_roi)

    roi_h, roi_w = clean_roi.shape
    # 情况1：OCR没识别到逗号，但像素特征显示有逗号 → 手动插入逗号
    if not has_comma_in_ocr and has_comma_in_pixel:
        print("检测到逗号像素但OCR未识别，手动补全逗号")
        # 按数字特征拆分：找到第一个数字的结束位置，插入逗号
        # 简单策略：按位置拆分（前半段=数字1，后半段=数字2）
        # 找到第一个非数字/负号的位置（若有）
        split_idx = int(sum(has_comma_in_pixel) / len(has_comma_in_pixel) // SINGLE_NUMBER_PIXEL)
        ocr_text = ocr_text[:split_idx] + ',' + ocr_text[split_idx:]
        print(f"补全逗号后文本：{ocr_text}")

    # 4. 解析文本：按逗号拆分并转换为数字
    number_list = []
    # 按逗号分割（处理可能的空格，比如 "123 , -456" 转为 ["123","-456"]）
    parts = [part.strip() for part in ocr_text.split(',') if part.strip()]

    for part in parts:
        # 转换为浮点数
        try:
            num = float(part)
            number_list.append(num)
        except ValueError:
            print(f"无法转换为数字：{part}")
            number_list.append(None)

    # 确保返回两个结果（不足补None，多余截断）
    while len(number_list) < 2:
        number_list.append(None)
    return number_list[0], number_list[1]


def get_one_nnumber_from_single_roi(o_x1, o_x2):
    """识别单个ROI的单个数字"""
    # 1. 计算ROI区域（覆盖两个数字+逗号）
    x1, y1 = pos_cal()
    frame = capture_screen()
    roi = frame[y1:y1 + VALUE_ROI_HEIGHT, x1 + o_x1:x1 + o_x2]

    # 2. 预处理ROI（去干扰、强化字符）
    clean_roi = clean_background_lines(roi)

    # 保存调试截图（确认ROI是否覆盖完整、预处理是否清晰）
    cv2.imwrite("debug_roi_full_one.png", roi)        # 原始ROI
    cv2.imwrite("debug_roi_clean_one.png", clean_roi) # 预处理后ROI

    # 3. OCR识别（包含逗号、负号、数字）
    ocr_text = pytesseract.image_to_string(clean_roi, config=OCR_CONFIG_1).strip()
    print(f"OCR原始识别结果：{ocr_text}")
    
    # -8检测
    ocr_text = check_is_negative_8(clean_roi, ocr_text)
    
    # 转为浮点数
    try:
        num = float(ocr_text)
        return num
    except ValueError:
        print(f"无法转换为数字：{ocr_text}")
        return None


def get_two_number_from_one():
    """识别单个ROI内用逗号分隔的两个数字"""
    # 1. 计算ROI区域（覆盖两个数字+逗号）
    x1, y1 = pos_cal()
    frame = capture_screen()
    roi = frame[y1:y1 + VALUE_ROI_HEIGHT, x1:x1 + VALUE_ROI_WIDTH]

    # 2. 预处理ROI（去干扰、强化字符）
    clean_roi = clean_background_lines(roi)

    pos_comma = find_comma_position(clean_roi)
    arv_pos = int(sum(pos_comma) / len(pos_comma))

    num1 = get_one_nnumber_from_single_roi(0, arv_pos - 5)
    num2 = get_one_nnumber_from_single_roi(arv_pos + 5, VALUE_ROI_WIDTH)

    return num1, num2


def get_img():
    """ 识别两个数字 """
    # 计算逗号的位置
    # 1. 计算ROI区域（覆盖两个数字+逗号）
    x1, y1 = pos_cal()
    frame = capture_screen()
    roi = frame[y1:y1 + VALUE_ROI_HEIGHT, x1:x1 + VALUE_ROI_WIDTH]

    # 2. 预处理ROI（去干扰、强化字符）
    clean_roi = clean_background_lines(roi)

    # 保存调试截图（确认ROI是否覆盖完整、预处理是否清晰）
    cv2.imwrite("debug_roi_clean.png", clean_roi)  # 预处理后ROI

# ====================== 运行测试 ======================
if __name__ == "__main__":
    # print("3秒后开始识别...")
    # time.sleep(3)
    #
    # # 识别单个ROI内的两个数字
    # num1, num2 = get_two_numbers_from_single_roi()
    # # num1, num2 = get_two_number_from_one()
    #
    # # 输出结果
    # if num1 is not None and num2 is not None:
    #     print(f"识别成功 → 数字1：{num1}，数字2：{num2}")
    # else:
    #     print(f"识别失败 → 数字1：{num1}，数字2：{num2}")
    i = 4
    num = '3'
    index = -1
    get_img()
    c1 = c_img.crop_text_max_rect('debug_roi_clean.png', 2)
    cv2.imwrite(r".\pro_img\c1.png", c1)
    if index == 1:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                      False, SINGLE_NUMBER_PIXEL * 0)
    elif index == 2:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                          False, SINGLE_NUMBER_PIXEL * 1)
    elif index == 3:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                          False, SINGLE_NUMBER_PIXEL * 2)
    elif index == -1:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                          True, SINGLE_NUMBER_PIXEL * 0)
    elif index == -2:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                          True, SINGLE_NUMBER_PIXEL * 1)
    elif index == -3:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL,
                                          True, SINGLE_NUMBER_PIXEL * 2)
    cv2.imwrite(r".\pro_img\c2.png", c2)
    c3 = c_img.pad_to_square_centered(r".\pro_img\c2.png")
    cv2.imwrite(f".\\pro_img\\{i}.png", c3)
    shutil.copy(f".\\pro_img\\{i}.png", f".\\dataset\\{num}")

