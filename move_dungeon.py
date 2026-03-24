import cv2
import numpy as np
import pyautogui
import time
import random
from winsize import set_window_size

MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"

MOVE_DEFULT_TIME = 3
MOVE_SPEED = 305
MOVE_1 = 452
MOVE_2 = 794
MOVE_3 = 452


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
    
def dungeon():
    # move_once("down", MOVE_1 / MOVE_SPEED)
    # move_once("left", MOVE_2 / MOVE_SPEED)
    move_once("up", MOVE_3 / MOVE_SPEED)
    
if __name__ == "__main__":
    print("请激活游戏")
    time.sleep(3)
    print("开始测试")
    dungeon()
    print("测试结束")

    

