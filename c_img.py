import cv2
import numpy as np

def crop_text_max_rect(image_path, padding=0):
    """
    黑底白字图片：找到文字的最大外接矩形并裁剪
    :param image_path: 图片路径
    :param padding: 裁剪边距（让文字不贴边，默认0）
    :return: 裁剪后的图片（numpy数组），未检测到文字返回None
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None

    # 2. 转灰度 + 二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 3. 查找文字轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未检测到文字轮廓")
        return None

    # 4. 合并轮廓 → 计算最大外接矩形
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # 5. 添加边距（防止越界）
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding

    # 6. 裁剪
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def crop_with_width_window(
    image_path,
    window_width,
    from_back=False,
    offset=0  # 新增：偏移像素（基于左/右基准）
):
    """
    固定宽度窗口裁剪 + 自定义左右偏移
    :param image: 输入图片（OpenCV数组）
    :param window_width: 裁剪窗口宽度
    :param from_back: False = 从左边（前）为基准
                      True  = 从右边（后）为基准
    :param offset: 偏移像素
        - from_back=False（左基准）：offset 向右移
        - from_back=True（右基准）：offset 向左移
    :return: 裁剪后的图片
    """
    image = cv2.imread(image_path)
    h, img_w = image.shape[:2]

    # 安全判断
    if window_width >= img_w:
        return image.copy()

    # 计算起始 x
    if from_back:
        # 以右边为基准，向左偏移 offset
        x = img_w - window_width - offset
    else:
        # 以左边为基准，向右偏移 offset
        x = 0 + offset

    # 防止越界
    x = max(0, min(x, img_w - window_width))

    # 裁剪
    cropped = image[:, x : x + window_width]
    return cropped


def pad_to_square_centered(image_path):
    """
    将黑底白字图片 对称填充黑色 变成正方形
    左右居中、上下居中，新增区域全是黑色
    :param image: 输入图片（OpenCV数组）
    :return: 正方形图片
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    max_side = max(h, w)  # 正方形边长 = 宽高中最大的那个

    # 计算上下左右需要填充的像素
    top = (max_side - h) // 2        # 上填充
    bottom = max_side - h - top      # 下填充
    left = (max_side - w) // 2       # 左填充
    right = max_side - w - left      # 右填充

    # 对称填充黑色 (0,0,0)
    square_img = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # 黑色
    )
    return square_img


if __name__ == '__main__':
    # 调用方法（padding=5 表示文字周围留5像素空白）
    cropped = crop_text_max_rect("debug_roi_clean_one.png", padding=5)

    # 保存结果
    if cropped is not None:
        cv2.imwrite("cropped_result.png", cropped)
        print("裁剪完成，已保存")