import cv2
import numpy as np


def find_template_center(target_img_path, template_img_path, method=cv2.TM_CCOEFF_NORMED):
    """
    在目标图片中匹配模板，并返回模板中心坐标 (x, y)
    :param target_img_path: 目标大图路径
    :param template_img_path: 模板小图路径
    :param method: 模板匹配算法，默认使用归一化的相关系数法（效果较好）
    :return: 模板中心坐标 (center_x, center_y)，若未找到则返回 None
    """
    # 读取图片
    target = cv2.imread(target_img_path)
    template = cv2.imread(template_img_path)

    if target is None or template is None:
        print("错误：无法读取图片，请检查路径")
        return None

    # 获取模板的宽和高
    h, w = template.shape[:2]

    # 执行模板匹配
    result = cv2.matchTemplate(target, template, method)

    # 寻找匹配结果中的最大值和位置（对于 TM_SQDIFF 系列方法应取最小值）
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 根据方法选择最佳匹配位置
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # 计算中心坐标
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    # 可视化（可选）
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(target, top_left, bottom_right, (0, 255, 0), 2)
    cv2.circle(target, (center_x, center_y), 3, (0, 0, 255), -1)
    cv2.imshow("匹配结果", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (center_x, center_y)


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    target_path = "1.png"  # 替换为你的目标大图路径
    template_path = "battle_template1.png"  # 替换为你的模板小图路径
    center = find_template_center(target_path, template_path)
    if center:
        print(f"模板中心坐标：{center}")
