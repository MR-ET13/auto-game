import pyautogui
import time
import cv2
import numpy as np

BATTLE_TEMPLATE_PATH = "battle_template2.png"
MATCH_THRESHOLD = 0.75
BATTLE_END_DELAY = 3.0

MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"

MOVE_DEFULT_TIME = 3
MOVE_SPEED = 315
MOVE_1 = 370
MOVE_2 = 500
MOVE_3 = 500
MOVE_4 = 800
MOVE_5 = 2000
MOVE_6 = 917
MOVE_7 = 1000
MOVE_8 = 190
MOVE_9 = 3500 # 偏大
MOVE_10 = 160
MOVE_11 = 800
MOVE_12 = 1000
MOVE_13 = 1000
MOVE_14 = 200
MOVE_15 = 1300
MOVE_16 = 300
##
MOVE_17 = 320
MOVE_18 = 487
MOVE_19 = 400
MOVE_20 = 212
MOVE_21 = 501
MOVE_22 = 1000
MOVE_23 = 800
MOVE_24 = 50
MOVE_25 = 500
#
MOVE_26 = 915
MOVE_27 = 100
MOVE_28 = 650
MOVE_29 = 650
MOVE_30 = 418
MOVE_31 = 100
MOVE_32 = 50
MOVE_33 = 605

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def move_once(direction, duration=MOVE_DEFULT_TIME):
    """执行单次方向移动：支持自定义移动时长，适配精准微调"""
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
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)
    time.sleep(0.05)


def presskey_times(key, times=1):
    for _ in range(times):
        pyautogui.press(key)
        time.sleep(0.5)


def is_in_battle():
    try:
        template = cv2.imread(BATTLE_TEMPLATE_PATH, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{BATTLE_TEMPLATE_PATH}")
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(result)[1]
        if max_val >= MATCH_THRESHOLD:
            print(f"✅ 检测到战斗界面（匹配度：{max_val:.2f}）")
            return True
        return False
    except Exception as e:
        print(f"⚠️ 战斗检测出错：{e}")
        return False

def take_battle():
    time.sleep(2)
    # 1. 战斗检测：遇到战斗等待结束
    if is_in_battle():
        print("🔴 战斗中，等待结束...")
        while is_in_battle():
            time.sleep(1)
        print("🟢 战斗结束，等待返回地图界面...")
        time.sleep(BATTLE_END_DELAY)

    
def dungeon():
    move_once("down", MOVE_1 / MOVE_SPEED)
    move_once("left", MOVE_2 / MOVE_SPEED)
    take_battle()

    move_once("down", MOVE_3 / MOVE_SPEED)
    move_once("right", MOVE_4 / MOVE_SPEED)
    take_battle()

    move_once("right", MOVE_5 / MOVE_SPEED)
    take_battle()

    move_once("right", MOVE_6 / MOVE_SPEED)
    move_once("up", MOVE_7 / MOVE_SPEED)
    take_battle()
    
    move_once("down", MOVE_8 / MOVE_SPEED)
    move_once("left", MOVE_9 / MOVE_SPEED)
    move_once("up", MOVE_10 / MOVE_SPEED)
    move_once("left", MOVE_11 / MOVE_SPEED)
    take_battle()
    
    move_once("left", MOVE_12 / MOVE_SPEED)
    move_once("up", MOVE_13 / MOVE_SPEED)
    take_battle()
    
    move_once("left", MOVE_14 / MOVE_SPEED)
    move_once("up", MOVE_15 / MOVE_SPEED)
    move_once("right", MOVE_16 / MOVE_SPEED)
    take_battle()
    
    move_once("right", MOVE_17 / MOVE_SPEED)
    move_once("up", MOVE_18 / MOVE_SPEED)
    move_once("right", MOVE_19 / MOVE_SPEED)
    take_battle()
    
    move_once("left", MOVE_20 / MOVE_SPEED)
    move_once("down", MOVE_21 / MOVE_SPEED)
    move_once("right", MOVE_22 / MOVE_SPEED)
    take_battle()
    
    move_once("right", MOVE_23 / MOVE_SPEED)
    move_once("up", MOVE_24 / MOVE_SPEED)
    move_once("right", MOVE_25 / MOVE_SPEED)
    take_battle()
    
    move_once("right", MOVE_26 / MOVE_SPEED)
    move_once("down", MOVE_27 / MOVE_SPEED)
    take_battle()
    
    move_once("down", MOVE_28 / MOVE_SPEED)
    move_once("right", MOVE_29 / MOVE_SPEED)
    move_once("down", MOVE_30 / MOVE_SPEED)
    move_once("left", MOVE_31 / MOVE_SPEED)
    take_battle()
    
    move_once("right", MOVE_32 / MOVE_SPEED)
    move_once("up", MOVE_33 / MOVE_SPEED)
    take_battle()
    
    move_once("right", 50 / MOVE_SPEED)
    presskey_times("j")
    presskey_times("w")
    presskey_times("j", 2)
    time.sleep(3)
    presskey_times("j")
    presskey_times("s", 3)
    presskey_times("j")
    presskey_times("d", 2)
    presskey_times("j")
    time.sleep(1)
    presskey_times("k", 2)

    move_once("up", 0.5)
    presskey_times("j", 2)
    presskey_times("w")
    presskey_times("j")
    time.sleep(3)
    
    
if __name__ == "__main__":
    print("请激活游戏")
    time.sleep(1)
    print("开始测试")
    while True:
        dungeon()
    print("测试结束")

    

