import pyautogui
import time
import cv2
import numpy as np
import random
# from get_pos import get_twonumberby_torch
from get_pos import get_numimg
from main_name import find_skt_center, get_single_template_center
from env_var import EnvVar

class ImageWrite:
    """ 截图信息类 """

    def __init__(self, t0, idx0):
        self.t = t0
        self.idx = idx0

class TemplateInfo:
    """ 模板信息类(暂未启用) """
    
    def __init__(self):
        pass
        
imageInfo = ImageWrite(time.time(), 1)  # 截图的信息，时间和序号        

BATTLE_TEMPLATE_PATH = "battle_template2.png"  # 战斗匹配模板
MATCH_THRESHOLD = 0.75  # 战斗模板匹配阈值
BATTLE_END_DELAY = 3.0  # 战斗结束后等待时间
IMG_TIME = 40  # 世界坐标位置截图间隔
IMG_IS = False  # 截图是否启用

TARGET1_THRESHOLD = 0.5  # 目标模本匹配阈值

# 移动键
MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"

# 生态副本移动过程
MOVE_SPEED = 315  # 移动速度


def capture_screen():
    """
    截取全屏并转为OpenCV格式
    :return:
    """
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def move_once(direction, duration=3.0):
    """
    执行单次方向移动：支持自定义移动时长，适配精准微调
    :param direction: 移动方向
    :param duration: 移动时长
    :return:
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
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)
    time.sleep(0.05)


def presskey_times(key, times=1):
    """
    多次点击指定按键
    :param key: 按键名称
    :param times: 按键次数 
    :return: 
    """
    for _ in range(times):
        pyautogui.press(key)
        time.sleep(0.5)


def is_in_battle():
    """
    判断是否处于战斗
    :return: 
    """
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
    """
    持续战斗
    :return: 
    """
    # 进入战斗缓冲时间
    time.sleep(3)
    # 1. 战斗检测：遇到战斗等待结束
    if is_in_battle():
        print("🔴 战斗中，等待结束...")
        while is_in_battle():
            time.sleep(1)
        print("🟢 战斗结束，等待返回地图界面...")
        time.sleep(BATTLE_END_DELAY)

def take_image():
    """
    截取图片，截取信息在imageInfo中
    :return: 
    """
    
    current_time = time.time()
    if not IMG_IS:    
        imageInfo.t = time.time()  # 取消截图
    delta_img_time = current_time - imageInfo.t
    if delta_img_time > IMG_TIME:
        print("开始截取坐标图片")
        get_numimg(0, imageInfo.idx)
        imageInfo.idx += 1
        imageInfo.t = time.time()


def move_to_target(target, x_or_y='x', delta=1.0, z=True):
    """
    移动到目标模板x/y对齐位置
    :param target: 模板路径 
    :param x_or_y: 对齐方向
    :param delta: 速度修正系数
    :param z: 是否二次匹配(True表示可能存在遮挡，不进行二次匹配)，返回更精确的修正系数    
    :return: 新的速度修正系数，默认为delta
    """
    print("开始接近目标")
    MOVE_SPEED_NEW = MOVE_SPEED * delta
    delta_new = delta
    if x_or_y == 'x':
        while True:
            if find_skt_center() and get_single_template_center(target, TARGET1_THRESHOLD): 
                play_x, _ = find_skt_center()
                target_pos, _ = get_single_template_center(target, TARGET1_THRESHOLD)
                
                dx = target_pos - play_x
                if dx > 0:                
                    move_once("right", dx / MOVE_SPEED_NEW)            
                else:            
                    move_once("left", -dx / MOVE_SPEED_NEW)
                
                if not z:
                    play_x_new, _ = find_skt_center()
                    target_pos, _ = get_single_template_center(target, TARGET1_THRESHOLD)
                    delta_new = (abs(dx) - abs(target_pos - play_x_new)) / abs(dx)
                
                break
                
            else:
                time.sleep(2)
                dir_move = random.choice(["left", "right"])
                move_once(dir_move, 0.2)
                print(f"疑似遮挡，已向{dir_move}移动0.2s")
                        
                
    else:
        while True:
            if find_skt_center() and get_single_template_center(target, TARGET1_THRESHOLD):
                _, play_y = find_skt_center()
                _, target_pos = get_single_template_center(target, TARGET1_THRESHOLD)
                
                
                dy = target_pos - play_y
                if dy > 0:           
                    move_once("down", dy / MOVE_SPEED_NEW)
                    
                else:           
                    move_once("up", -dy / MOVE_SPEED_NEW)
                
                if not z:
                    _, play_y_new = find_skt_center()
                    _, target_pos = get_single_template_center(target, TARGET1_THRESHOLD)
                    delta_new = (abs(dy) - abs(target_pos - play_y_new)) / abs(dy)
                    
                break
            else:
                time.sleep(2)
                # k键退回
                presskey_times("k", 5)
                dir_move = random.choice(["up", "down"])
                move_once(dir_move, 0.2)
                print(f"疑似遮挡，已向{dir_move}移动0.2s")
    
    print(f"移动结束, 原速度：{MOVE_SPEED}, 新速度：{MOVE_SPEED_NEW}, 推荐修正率：{delta_new}")
    
    return delta_new
    
def dungeon1():
    """
    英雄生态副本自动化移动
    :return: 
    """
    
    print("=" * 20)
    print("生态英雄副本")
    print("=" * 20)
    
    # 初始化
    pass_num = 0
    count_boss_max = 1000
    
    while True:
        print("当前时间：", time.strftime("%H:%M"))
        print(f"通过副本次数{pass_num}")
    
        move_once("down", 0.1)
        move_to_target(r".\target_template\t1.png", 'y', 0.963)
        move_once("left", 0.5)
        take_battle()
        
        move_to_target(r".\target_template\t2.png", 'y', 0.952)
        move_once("right", 1.5)
        take_battle()
        
        move_once("right", 5.5)
        take_battle()
        
        move_to_target(r".\target_template\t3.png", 'x', 0.947)
        move_once("up", 1.5)
        take_battle()
        
        move_once("down", 2)
        move_once("left", 9)
        move_to_target(r".\target_template\t4.png", 'y', 0.984)
        move_once("left", 5)
        take_battle()
        
        move_once("left", 3.5)
        move_once("up", 2.5)
        take_battle()
        
        move_once("left", 0.8)
        move_once("up", 4.2)
        move_once("right", 0.8)
        take_battle()
        
        move_to_target(r".\target_template\t5.png", 'x', 0.98, True)
        move_to_target(r".\target_template\t6.png", 'y', 0.90)
        move_once("right", 1)
        take_battle()
        
        move_to_target(r".\target_template\t7.png", 'x', 1.01)
        move_to_target(r".\target_template\t8.png", 'y', 0.99)
        move_once("right", 2.5)
        take_battle()
        
        move_once("right", 0.9)
        move_to_target(r".\target_template\t9.png", 'y', 1.2)
        move_once("right", 1.8)
        take_battle()
        
        move_to_target(r".\target_template\t10.png", 'x', 0.94, True)
        move_once("down", 0.5)
        take_battle()
        
        move_once("down", 2)
        move_once("right", 2)
        move_once("down", 1.4)
        count_boss_max -= 1
        if count_boss_max < 0:

            move_once("left", 0.5)
            take_battle()
            recover()  # 补血
            count_boss_max = 3
        
        move_once("right", 0.5)
        move_to_target(r".\target_template\t15.png", 'y', 0.96)
        
        if get_single_template_center(r".\target_template\t13.png", 1.1):
            move_to_target(r".\target_template\t13.png", 'x', 0.96)
            move_once("up", 1.5)
            take_battle()

            move_once("up", 1)
            move_once("right", 1.1)
            move_once("up", 1.5)
            take_battle()
			
            move_once("down", 1)
        else:
            # 没有隐藏
            move_to_target(r".\target_template\t16.png", 'x', 0.96)
            move_once("up", 1.5)
             
        move_to_target(r".\target_template\t14.png", 'x', 1.02, True)
        move_once("up", 1.5)
        take_back()


    
def take_back():
    """
    副本主要移动结束后的重置操作 
    :return: 
    """
    presskey_times("j")
    presskey_times("w")
    presskey_times("j")
    time.sleep(1)
    presskey_times("j")
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
    presskey_times("w", 2)
    presskey_times("j")
    time.sleep(4)


def recover():
    presskey_times("j")
    presskey_times("s")
    presskey_times("j")
    time.sleep(0.5)
    
    presskey_times("i")
    presskey_times("j")
    presskey_times("s")
    presskey_times("j")  
    presskey_times("w")
    presskey_times("j")  # 补充数量1
    presskey_times("s")
    presskey_times("j", 2)  # 补充数量2
    
    presskey_times("k", 5)
    time.sleep(1)
    

def test_move():
    while True:
        input("按键并激活游戏界面继续...")
        evar = EnvVar("env_var.txt")
        time.sleep(3)

        if evar.get_val("select") == "move":
            move_once(evar.get_val("direction"), evar.get_val("time"))
        elif evar.get_val("select") == "move_by_tar":
            move_to_target(r".\target_template" + evar.get_val("template"),
                           evar.get_val("first_dir"), evar.get_val("delta"), False)
            
    
if __name__ == "__main__":
    # recover()
    dungeon1()
    # test_move()

    

