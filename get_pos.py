import cv2
import numpy as np
###################################################
import pytesseract
import time
import pyautogui
import c_img
import shutil
from doubao_torch import get_numberbytorch

# ====================== 【必须配置】修改这里 ======================
##################################################
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
TEMPLATE_PATH = "number_template.png"  # 你的模板图片

# 【关键修改】单个ROI覆盖两个数字+逗号（需根据实际画面调整）
# 建议先通过debug截图确认ROI能完整包含两个数字和中间的逗号
VALUE_ROI_WIDTH = 136      # 加宽：容纳两个数字+逗号（原55不够，需实测调整）
VALUE_ROI_HEIGHT = 37      # 高度保持（根据数字高度调整）
# VALUE_ROI_WIDTH = 50      # 加宽：容纳两个数字+逗号（原55不够，需实测调整）
# VALUE_ROI_HEIGHT = 15      # 高度保持（根据数字高度调整）

SINGLE_NUMBER_PIXEL = 17  # 单个数字的像素值用于计算逗号分隔,有待测试

TEMP_X, TEMP_Y = 3464, 285  # 临时位置
MODEL_PATH = 'my_own_model_a1.pth'  # 模型路径

# =================================================================

# 【关键修改】OCR白名单加入逗号，保留psm 7（单行文本）
OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-, -c preserve_interword_spaces=0'
# OCR_CONFIG_1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-, -c preserve_interword_spaces=0'
OCR_CONFIG_1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'


def clean_background_lines(roi):
    """
    消除背景浅色干扰横线，只保留白色数字、负号和逗号
    :param roi: 图片 
    :return: 清洗后的灰度图
    """
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
    """
    匹配模板并返回位置坐标
    :return: 右上角的x, y坐标
    """
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
        print(f"未找到目标模板，匹配度：{max_val}（阈值0.6）\n使用临时位置")
        return TEMP_X + w, TEMP_Y
    else:
        # print(f"模板匹配成功，基准位置：{max_loc}，匹配度：{max_val}")
        return max_loc[0] + w, max_loc[1]


def find_comma_position(clean_roi):
    """
    像素特征定位逗号位置
    :param clean_roi: 灰度图  
    :return: 识别到的逗号位置序列
    """
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


def get_twonumberby_torch():
    """
    通过自己训练的模型识别坐标
    :return: 识别的x, y坐标
    """
    get_numimg(0)
    image_path = f".\\pro_img\\c1.png"
    allnums, firstnums = watch_imgnums(image_path)
    print(f"截取的数字个数allnums: {allnums}, firstnums: {firstnums}")
    get_numimg(firstnums)
    res1 = get_numberbytorch(r".\pro_img\c2.png", MODEL_PATH)
    get_numimg(allnums - firstnums, 1, True)
    res2 = get_numberbytorch(r".\pro_img\c2.png", MODEL_PATH)

    number_list = []
    reslist = [res1, res2]
    for part in reslist:
        # 转换为浮点数
        try:
            num = float(part)
            number_list.append(num)
        except ValueError:
            print(f"无法转换为数字：{part}")
            number_list.append(None)

    return number_list[0], number_list[1]



def get_img():
    """
    截取世界坐标位置并清洗图片
    :return: 将图片保存到".\\debug_roi_clean.png"
    """
    # 计算逗号的位置
    # 1. 计算ROI区域（覆盖两个数字+逗号）
    x1, y1 = pos_cal()
    frame = capture_screen()
    roi = frame[y1:y1 + VALUE_ROI_HEIGHT, x1:x1 + VALUE_ROI_WIDTH]

    # 2. 预处理ROI（去干扰、强化字符）
    clean_roi = clean_background_lines(roi)

    # 保存调试截图（确认ROI是否覆盖完整、预处理是否清晰）
    cv2.imwrite("debug_roi_clean.png", clean_roi)  # 预处理后ROI

def get_testimg(i, num, index):
    """
    窗口截取数字测试用函数
    :param i: 保存图片的名称 
    :param num: 移动到数据集下的哪个分区
    :param index: 偏移特征
    :return: 保存一系列图片文件
    """
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
    shutil.move(f".\\pro_img\\{i}.png", f".\\my_dataset\\{num}")

def get_numimg(digit_num, index=1, from_back=False):
    """
    截取清洗图片，去除无用的黑色背景，窗口截取数字，数字方形填充
    :param digit_num: 数字位数，0表示保留全部
    :param index: 保存图片序号
    :return:
    """
    get_img()
    c1 = c_img.crop_text_max_rect('debug_roi_clean.png', 2)
    cv2.imwrite(f".\\pro_img\\c{index}.png", c1)
    if digit_num:
        c2 = c_img.crop_with_width_window(r".\pro_img\c1.png", SINGLE_NUMBER_PIXEL * digit_num,
                                          from_back, SINGLE_NUMBER_PIXEL * 0)
        cv2.imwrite(r".\pro_img\c2.png", c2)
        c3 = c_img.pad_to_square_centered(r".\pro_img\c2.png")
        cv2.imwrite(f".\\pro_img\\c3.png", c3)

def watch_imgnums(image_path):
    """
    返回清洗图片的数字分布
    :param image_path:清洗图片路径
    :return: 所有数字总和，逗号前一个数字总数
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    clean_roi = clean_background_lines(img)
    pos_comma = find_comma_position(clean_roi)
    firstnums = int(sum(pos_comma) / len(pos_comma)) // SINGLE_NUMBER_PIXEL
    allnums = (w - int(sum(pos_comma) / len(pos_comma)))  // SINGLE_NUMBER_PIXEL + firstnums
    if firstnums == allnums:
        allnums += 1
    return allnums, firstnums

def get_dataset_from_clearimg(image_name, save_startindex=0):
    """
    处理单个清洗后的图片，分割成单个正方形数字
    :param allnums: 图片中数字总和
    :param firstnums: 逗号前的数字总和
    :param image_name: 图片名称
    :param save_startindex: 保存开始序号
    :return: 下一个保存开始序号
    """
    save_index = save_startindex
    image_path = f".\\pro_img\\{image_name}.png"
    allnums, firstnums = watch_imgnums(image_path)
    print(allnums, firstnums)
    for i in range(firstnums):
        c2 = c_img.crop_with_width_window(image_path, SINGLE_NUMBER_PIXEL,
                                      False, SINGLE_NUMBER_PIXEL * i)
        cv2.imwrite(r".\pro_img\temp.png", c2)
        c3 = c_img.pad_to_square_centered(r".\pro_img\temp.png")
        cv2.imwrite(f".\\pro_img\\save_path\\d{save_index}.png", c3)
        save_index += 1
    for i in range(allnums - firstnums):
        c2 = c_img.crop_with_width_window(image_path, SINGLE_NUMBER_PIXEL,
                                          True, SINGLE_NUMBER_PIXEL * i)
        cv2.imwrite(r".\pro_img\temp.png", c2)
        c3 = c_img.pad_to_square_centered(r".\pro_img\temp.png")
        cv2.imwrite(f".\\pro_img\\save_path\\d{save_index}.png", c3)
        save_index += 1

    return save_index

def get_dataset(list_imgindex, save_startindex=0):
    """
    根据图片序号列表分割成单个数字图片
    :param list_imgindex: 图片序号列表
    :param save_startindex: 开始保存序号
    :return:
    """
    save_index = save_startindex
    for i in list_imgindex:
        img_name = f"c{i}"
        save_index = get_dataset_from_clearimg(img_name, save_index)

def ocr_clearimg(image_path):
    """
    用ocr识别单个图片
    :param image_path: 图片路径
    :return: 识别结果string
    """
    img = cv2.imread(image_path)
    clean_roi = clean_background_lines(img)
    ocr_text = pytesseract.image_to_string(clean_roi, config=OCR_CONFIG_1).strip()
    if ocr_text == '':
        ocr_text = '-'
        print(f"{image_path}\n识别为空，自动补为'-'")
    return ocr_text

def classify_img(list_imgindex, class_img=''):
    """
    根据之前训练的模型识别的图片内容分类到数据训练集
    :param list_imgindex: 图片序列
    :param class_img: 图片类别'', a, b, c, ...
    :return:
    """
    dst_path = r".\my_dataset"
    for i in list_imgindex:
        image_path = f".\\pro_img\\save_path\\d{i}.png"
        ocr_text = get_numberbytorch(image_path, MODEL_PATH)
        image_path_new = f".\\pro_img\\save_path\\d{class_img}{i}.png"
        shutil.move(image_path, image_path_new)
        if ocr_text == '-':
            shutil.move(image_path_new, dst_path + r"\-")
        elif ocr_text == '1':
            shutil.move(image_path_new, dst_path + r"\1")
        elif ocr_text == '2':
            shutil.move(image_path_new, dst_path + r"\2")
        elif ocr_text == '3':
            shutil.move(image_path_new, dst_path + r"\3")
        elif ocr_text == '4':
            shutil.move(image_path_new, dst_path + r"\4")
        elif ocr_text == '5':
            shutil.move(image_path_new, dst_path + r"\5")
        elif ocr_text == '6':
            shutil.move(image_path_new, dst_path + r"\6")
        elif ocr_text == '7':
            shutil.move(image_path_new, dst_path + r"\7")
        elif ocr_text == '8':
            shutil.move(image_path_new, dst_path + r"\8")
        elif ocr_text == '9':
            shutil.move(image_path_new, dst_path + r"\9")
        else:
            shutil.move(image_path_new, dst_path + r"\1")
    

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

    # get_testimg(23, '4', -2)
    # get_numimg(0)
    # save_index = get_dataset_from_clearimg(5, 2, "c4", 0)
    # print(save_index)

    # allnums, firstnums = watch_imgnums(r".\pro_img\c22.png")
    # print(allnums, firstnums)

    # 将main.py生成的位置图片截取为单个离散数字
    # list_imgindex = range(1, 60 + 1)
    # get_dataset(list_imgindex)

    # ocr_text = ocr_clearimg(r"F:\XUNIJI_FILE\PythonFile\autogame\pro_img\save_path\d39.png")
    # print(ocr_text)

    # 分类单个离散数字图片
    list_imgindex = range(0, 228 + 1)
    classify_img(list_imgindex, 'c')
    
    # 获取世界坐标
    # get_twonumberby_torch()