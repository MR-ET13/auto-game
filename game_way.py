import cv2
import numpy as np
import pyautogui
import pydirectinput
import time
import random
from env_var import EnvVar

# 移动相关
MOVE_LEFT_KEY = "left"  # 左移按键
MOVE_RIGHT_KEY = "right"  # 右移按键
MOVE_UP_KEY = "up"  # 上移按键
MOVE_DOWN_KEY = "down"  # 下移按键
# 模板与阈值
TEM1 = ".//template_battle//first_people.png"
THR1 = 0.8

def capture_screen():
    """
    截取全屏并转换为OpenCV的BGR格式
    :return: BGR格式
    """
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def is_in_battle(battle_template_path=TEM1, match_threshold=THR1):
    """
    检测是否处于战斗界面，返回布尔值
    :param battle_template_path: 检测路径
    :param match_threshold: 匹配阈值
    :return: True or False
    """
    try:
        # 读取模板图（灰度模式，减少计算量）
        template = cv2.imread(battle_template_path, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{battle_template_path}")

        # 截取屏幕并转为灰度图
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(result)[1]  # 只取最大匹配值

        if max_val >= match_threshold:
            print(f"✅ 检测到（匹配度：{max_val:.2f}）")
            return True
        else:
            print(f"❌️未检测到（匹配度：{max_val:.2f}）")
            return False
    except Exception as e:
        print(f"⚠️ 战斗检测出错：{e}")
        return False


def move_once(direction, duration):
    """
    执行单次方向移动：支持自定义移动时长，适配精准微调
    :param direction: 移动方向
    :param duration: 移动时间
    :return: None
    """
    key_map = {
        "up": (MOVE_UP_KEY, "向上", "⬆️"),
        "down": (MOVE_DOWN_KEY, "向下", "⬇️"),
        "left": (MOVE_LEFT_KEY, "向左", "⬅️"),
        "right": (MOVE_RIGHT_KEY, "向右", "➡️")
    }
    if direction not in key_map:
        print(f"⚠️ 无效移动方向：{direction}")
        return
    key, text, icon = key_map[direction]
    print(f"{icon} 执行{text}移动，时长 {duration} 秒")
    pydirectinput.keyDown(key)
    time.sleep(duration)
    pydirectinput.keyUp(key)
    time.sleep(0.05)


def get_single_template_center(template_path, threshold):
    """
    单模板匹配，返回中心坐标
    :param template_path: 匹配模板位置
    :param threshold: 匹配阈值
    :return: 匹配成功返回中心坐标，否则返回None
    """
    try:
        template = cv2.imread(template_path, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{template_path}")

        h, w = template.shape[:2]
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        # 规范解包：避免参数顺序错误
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            print(f"🎯 检测到目标对象（匹配度：{max_val:.2f}），中心坐标：({center_x}, {center_y})")
            return (center_x, center_y)
        else:
            print(f"❌ 未检测到目标对象，匹配度：{max_val:.2f}（阈值：{threshold}）")
            return None
    except Exception as e:
        print(f"⚠️ 获取目标坐标出错：{e}")
        return None


def presskey_times(key, times=1, sleep_time=0.5, outside=False):
    """
    多次点击指定按键
    :param key: 按键名称
    :param times: 按键次数
    :param sleep_time: 按键切换时间间隔
    :param outside: 是否处于遇敌状态，默认非遇敌状态
    :return:
    """
    for _ in range(times):
        pydirectinput.press(key)
        if outside:
            judgment_out(key)
        time.sleep(0.5)
    time.sleep(sleep_time)

def judgment_out(key):
    """
    判断是否正常移动
    :param key: 移动键值
    :return: True/False
    """
    if key == "d":
        if is_in_battle(".//template_war_vehicle//right.png"):
            print("移动正常")
        else:
            print("遇到战斗")
            take_battle(0.2)
            pydirectinput.press("d")
    elif key == "a":
        if is_in_battle(".//template_war_vehicle//left.png"):
            print("移动正常")
        else:
            print("遇到战斗")
            take_battle(0.2)
            pydirectinput.press("a")
    elif key == "w":
        if is_in_battle(".//template_war_vehicle//up.png"):
            print("移动正常")
        else:
            print("遇到战斗")
            take_battle(0.2)
            pydirectinput.press("w")
    elif key == "s":
        if is_in_battle(".//template_war_vehicle//down.png"):
            print("移动正常")
        else:
            print("遇到战斗")
            take_battle(0.2)
            pydirectinput.press("s")

def take_battle(buff_time=3):
    """
    持续战斗
    :param buff_time: 进入战斗缓冲时间
    :return:
    """
    print("🔴 战斗中，等待结束...")
    # 按下z键
    pydirectinput.keyDown("z")
    while not is_in_battle():
        time.sleep(0.1)
    print("🟢 战斗结束，等待返回地图界面...")
    pydirectinput.keyUp("z")


def move_by_files(keys_files, outside=False):
    """
    通过txt文本文件移动
    :param keys_files: 移动文件
    :param outside: 是否在户外
    :return: None
    """
    move_all = EnvVar(keys_files)
    dict_move = move_all.config
    for key, value in dict_move.items():
        print(f"{key}")
        k, t, s = value.split("-")
        presskey_times(k, int(t), float(s), outside)


def encounter_enemy():
    """
    左右遇敌，轮域
    :return: None
    """
    # 初始化无战斗计时
    no_battle_start_time = time.time()

    while True:
        # 随机决定本次循环的第一个移动方向
        # first_dir = random.choice(["left", "right", "up", "down"])
        # first_dir = random.choice(["left", "right"])
        first_dir = "right"
        if first_dir in ["left", "right"]:
            second_dir = "right" if first_dir == "left" else "left"
        else:
            second_dir = "up" if first_dir == "down" else "down"

        # 执行：先随机方向，再切换方向
        move_once(first_dir, 0.5)
        if not is_in_battle():
            current_time = time.time()
            delta_time_no_battle = current_time - no_battle_start_time
            if delta_time_no_battle > 20:
                pydirectinput.press("j")
                pydirectinput.press("k", 5)
                no_battle_start_time = time.time()
        else:
            no_battle_start_time = time.time()
        take_battle(0.8)

        move_once(second_dir, 0.5)
        if not is_in_battle():
            current_time = time.time()
            delta_time_no_battle = current_time - no_battle_start_time
            if delta_time_no_battle > 20:
                pydirectinput.press("k", 4)
                no_battle_start_time = time.time()
        else:
            no_battle_start_time = time.time()
        take_battle(0.8)


def test_move():
    """
	move_once(), move_to_target()调试，通过env_var.txt设置变量
	:return:
	"""
    while True:
        input("按键并激活游戏界面继续...")
        evar = EnvVar("env_var.txt")
        time.sleep(3)

        if evar.get_val("select") == "move":
            move_once(evar.get_val("direction"), evar.get_val("time"))
        elif evar.get_val("select") == "move_by_press":
            presskey_times(evar.get_val("key"), evar.get_val("time"))

def left_or_right(time):
    """
    左右遇敌，黎明
    :param time: 移动时间
    :return: None
    """
    dir = "left"
    while True:
        if is_in_battle():
            move_once(dir, time)
            dir = "right" if dir == "left" else "left"
        else:
            take_battle()


def lure_enemy():
    """
    诱敌刷怪，飞狗
    :return : None
    """
    MAX_LURE = 24 # 最大诱敌数
    lure_number = 24 # 初始诱敌数
    save_number1 = 16 # 1车的营帐数
    save_number2 = 13 # 2车的营帐数
    while True:
        if not is_in_battle(): # 进入战斗
            take_battle()
            lure_number -= 1
        elif lure_number > 0: # 诱敌飞狗
            pydirectinput.press("x")
            time.sleep(0.2)
            pydirectinput.press('down', presses=2)
            pydirectinput.press("z", presses=2)
            time.sleep(0.2)
            pydirectinput.press("z")
            time.sleep(0.2)
            pydirectinput.press("up")
            pydirectinput.press("z", presses=4)
        elif save_number1 > 0: # 单次诱敌数用尽，使用1车营帐
            pydirectinput.press("x")
            time.sleep(0.2)
            pydirectinput.press("z")
            time.sleep(0.2)
            pydirectinput.press("right")
            pydirectinput.press("z")
            time.sleep(0.2)
            pydirectinput.press("z", presses=3, interval=0.2)
            time.sleep(0.2)
            pydirectinput.press("up")
            pydirectinput.press("z")
            time.sleep(0.2)
            save_number1 -= 1
            lure_number = MAX_LURE
        else: # 1车营帐用尽，使用2车营帐
            pydirectinput.press("x")
            time.sleep(0.2)
            pydirectinput.press("z")
            time.sleep(0.2)
            pydirectinput.press("right")
            pydirectinput.press("down")
            pydirectinput.press("z")
            time.sleep(0.2)
            pydirectinput.press("z", presses=3, interval=0.2)
            time.sleep(0.2)
            pydirectinput.press("up")
            pydirectinput.press("z")
            time.sleep(0.2)
            save_number2 -= 1
            lure_number = MAX_LURE
    
    


if __name__ == "__main__":
    # test_move()
    time.sleep(1)
    print("yidong")
    # move_by_files("move_keys1.txt")
    # move_by_files("move_keys2.txt", True)
    # encounter_enemy()
    # left_or_right(0.1)  #黎明-新手村左右遇敌
    lure_enemy()