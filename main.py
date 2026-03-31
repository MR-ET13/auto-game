import cv2
import numpy as np
import pyautogui
import time
import random
from winsize import set_window_size
from get_pos import get_numimg, get_twonumberby_torch
from main_name import find_skt_center, get_single_template_center
from move_dungeon import presskey_times, move_to_target
from main_backup import move_dungeon as md

# ==================== 全局配置（可根据游戏调整） ====================
# 模板匹配相关
BATTLE_TEMPLATE_PATH = "battle_template2.png"  # 战斗界面的模板图路径，1表示虚拟机，2表示主机
MINE_TEMPLATE = "mine_template.png"  # 矿的模板路径
MINE_THRESHOLD = 0.65  # 矿匹配阈值
MOVE_CONST_MAX = 500  # 默认大移动幅度
MOVE_CONST_MIN = 100  # 默认小移动幅度
MOVE_SPEED = 315  # 移动速度
MATCH_THRESHOLD = 0.75  # 战斗模板匹配的置信度阈值
# 移动相关
MOVE_LEFT_KEY = "a"  # 左移按键
MOVE_RIGHT_KEY = "d"  # 右移按键
MOVE_UP_KEY = "w"  # 上移按键
MOVE_DOWN_KEY = "s"  # 下移按键
MOVE_DURATION = 3  # 默认移动时间
MOVE_DURATION = 1.0  # 单次移动时长（秒）
# 延迟相关
BATTLE_END_DELAY = 3.0  # 战斗结束后等待返回地图的时间
INTERFACE_DELAY = 3.0  # 脚本启动后的初始等待时间
NO_BATTLE_TIMEOUT = 20.0  # 无战斗超时阈值（秒）
IMG_TIME = 20  # 截图时间间隔
IMG_START_IDX = 33  # 截图开始序号
IMG_IS = False
# 安全配置
pyautogui.PAUSE = 0.1  # 所有pyautogui操作的间隔
pyautogui.FAILSAFE = True  # 鼠标移到屏幕四角触发紧急停止
# 窗口位置大小设置 w, h, x, y
# 主机
WINDOWS_ZZJB = [2380, 1400, 1376, 170]
WINDOWS_MLH = [1180, 1760, 130, 170]
# 虚拟机
# WINDOWS_ZZJB = [850, 600, 500, 50]
# WINDOWS_MLH = [300, 650, 15, 50]



# ==================== 核心功能函数 ====================
def capture_screen():
    """截取全屏并转换为OpenCV的BGR格式"""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def is_in_battle():
    """检测是否处于战斗界面，返回布尔值"""
    try:
        # 读取模板图（灰度模式，减少计算量）
        template = cv2.imread(BATTLE_TEMPLATE_PATH, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{BATTLE_TEMPLATE_PATH}")

        # 截取屏幕并转为灰度图
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(result)[1]  # 只取最大匹配值

        if max_val >= MATCH_THRESHOLD:
            print(f"✅ 检测到战斗界面（匹配度：{max_val:.2f}）")
            return True
        return False
    except Exception as e:
        print(f"⚠️ 战斗检测出错：{e}")
        return False

def is_in_mine():
    """检测是否有矿，返回布尔值"""
    try:
        # 读取模板图（灰度模式，减少计算量）
        template = cv2.imread(MINE_TEMPLATE, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{MINE_TEMPLATE}")

        # 截取屏幕并转为灰度图
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(result)[1]  # 只取最大匹配值

        if max_val >= MINE_THRESHOLD:
            print(f"✅ 检测到有矿（匹配度：{max_val:.2f}）")
            return True
        return False
    except Exception as e:
        print(f"⚠️ 矿检测出错：{e}")
        return False

def move_once(direction, duration=MOVE_DURATION):
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


def execute_timeout_operation():
    """执行超时无战斗时的按键操作"""
    print("\n⏰ 超过20秒未进入战斗，执行特殊操作...")
    # 延时2秒
    time.sleep(2)
    print("🔹 按下并弹起 D 键")
    pyautogui.press(MOVE_RIGHT_KEY)  # press方法自动完成按下+弹起
    # 延时1秒
    time.sleep(1)
    print("🔹 再次按下并弹起 D 键")
    pyautogui.press(MOVE_RIGHT_KEY)
    # 延时1秒
    time.sleep(1)
    print("🔹 按下并弹起 J 键")
    pyautogui.press("j")
    # 延时1秒
    time.sleep(1)
    print("🔹 按下并弹起 K 键")
    pyautogui.press("k")
    print("✅ 超时操作执行完成，恢复正常移动\n")


# ==================== 主逻辑（核心玩法） ====================
def main():
    print("=" * 60)
    print("游戏自动遇敌脚本（随机起始方向 + 20秒无战斗触发特殊操作）")
    print(f"移动规则：脱离战斗后随机选方向，然后交替移动")
    print(f"超时规则：20秒未战斗 → 2s延时→D→1s→D→1s→J→1s→K")
    print(f"按键配置：左({MOVE_LEFT_KEY}) 右({MOVE_RIGHT_KEY}) | 紧急停止：鼠标移屏幕四角")
    print("=" * 60)
    # set_window_size("重装机兵:墟", WINDOWS_ZZJB[0], WINDOWS_ZZJB[1], WINDOWS_ZZJB[2], WINDOWS_ZZJB[3])  # 设置窗口为指定大小和位置
    # set_window_size("Windows PowerShell", WINDOWS_MLH[0], WINDOWS_MLH[1], WINDOWS_MLH[2], WINDOWS_MLH[3])  # 设置窗口为指定大小和位置

    time.sleep(INTERFACE_DELAY)  # 启动后等待，给你切换到游戏窗口的时间

    # 初始化无战斗计时
    no_battle_start_time = time.time()
    img_time = time.time()
    img_index = IMG_START_IDX

    try:
        while True:

            if find_skt_center() and get_single_template_center("mine_template.png", 0.6):
                try:
                    play_x, play_y = find_skt_center()
                    mine_x, mine_y = get_single_template_center("mine_template.png", 0.6)
                except ValueError:
                    print("解包失败，重试")
                    continue

                if 40 < mine_y - play_y < 120:
                    move_to_target("mine_template.png", 'x', 0.96)
                    time.sleep(0.5)
                    presskey_times("j")
                    time.sleep(0.5)
                    presskey_times("j")
                    time.sleep(6)
                    presskey_times("k", 5)
            # 1. 检测到战斗：重置计时，等待战斗结束
            if is_in_battle():
                # no_battle_start_time = time.time()  # 重置无战斗计时器
                print("🔴 进入战斗状态，等待战斗结束...")
                while is_in_battle():  # 循环检测，直到脱离战斗
                    time.sleep(1)
                print("🟢 战斗结束，等待返回地图界面...")
                time.sleep(BATTLE_END_DELAY)
                no_battle_start_time = time.time()  # 战斗结束后重新计时

            # 2. 未检测到战斗：检查是否超时
            else:
                if not IMG_IS:
                    img_time = time.time()  # 不截图操作
                    
                current_time = time.time()
                elapsed_time = current_time - no_battle_start_time
                
                delta_img_time = current_time - img_time

                # 2.1 超过20秒未战斗：执行特殊操作
                if elapsed_time > NO_BATTLE_TIMEOUT:
                    execute_timeout_operation()
                    no_battle_start_time = time.time()  # 重置计时器
                    
                # 达到截图时间间隔
                elif delta_img_time > IMG_TIME:
                    print("开始截取坐标图片")
                    get_numimg(0, img_index)
                    img_index += 1
                    img_time = time.time()

                # 2.2 未超时：执行正常移动
                else:
                    print(f"⏳ 未检测到战斗，已计时 {elapsed_time:.1f} 秒（阈值：20秒）")
                    # 随机决定本次循环的第一个移动方向
                    # first_dir = random.choice(["left", "right", "up", "down"])
                    first_dir = random.choice(["left", "right"])
                    if first_dir in ["left", "right"]:
                        second_dir = "right" if first_dir == "left" else "left"
                    else:
                        second_dir = "up" if first_dir == "down" else "down"

                    # 执行：先随机方向，再切换方向
                    move_once(first_dir, 1.5)
                    move_once(second_dir, 1.5)

    except KeyboardInterrupt:
        print("\n🛑 脚本已手动停止（Ctrl+C）")
    except Exception as e:
        print(f"\n❌ 脚本异常终止：{str(e)}")


def dig_mine():
    """
    挖矿功能
    :return: 
    """
    print("=" * 20)
    print("挖矿")
    print("=" * 20)
    # 初始化参数
    set_posx = -4
    set_posy = 24
    num_mine = 0
    
    try:
        while True:
            print("当前时间：", time.strftime("%H:%M"))
            if is_in_mine() and find_skt_center():
                mine_x, mine_y = get_single_template_center(MINE_TEMPLATE, MINE_THRESHOLD)
                play_x, play_y = find_skt_center()
                
                
                dx = mine_x - play_x
                move_time = dx / MOVE_SPEED
                if dx > 0:        
                    move_once("right", move_time)
                    move_dir = "right"
                    # 进入战斗缓冲时间
                    time.sleep(2)
                    # 1. 战斗检测：遇到战斗等待结束
                    if is_in_battle():
                        print("🔴 战斗中，等待结束...")
                        while is_in_battle():
                            time.sleep(1)
                        print("🟢 战斗结束，等待返回地图界面...")
                        time.sleep(BATTLE_END_DELAY)
                        continue
                else:
                    move_time = -move_time
                    move_once("left", move_time)
                    move_dir = "left"
                    # 进入战斗缓冲时间
                    time.sleep(2)
                    # 1. 战斗检测：遇到战斗等待结束
                    if is_in_battle():
                        print("🔴 战斗中，等待结束...")
                        while is_in_battle():
                            time.sleep(1)
                        print("🟢 战斗结束，等待返回地图界面...")
                        time.sleep(BATTLE_END_DELAY)
                        continue

                
                move_once("up", MOVE_CONST_MAX / MOVE_SPEED)
                # 进入战斗缓冲时间
                time.sleep(2)
                # 1. 战斗检测：遇到战斗等待结束
                if is_in_battle():
                    print("🔴 战斗中，等待结束...")
                    while is_in_battle():
                        time.sleep(1)
                    print("🟢 战斗结束，等待返回地图界面...")
                    time.sleep(BATTLE_END_DELAY)
                    continue
                
                presskey_times("j", 2)
                time.sleep(6)
                
                num_mine += 1
                
                move_once("down", MOVE_CONST_MIN / MOVE_SPEED)
                # 进入战斗缓冲时间
                time.sleep(2)
                # 1. 战斗检测：遇到战斗等待结束
                if is_in_battle():
                    print("🔴 战斗中，等待结束...")
                    while is_in_battle():
                        time.sleep(1)
                    print("🟢 战斗结束，等待返回地图界面...")
                    time.sleep(BATTLE_END_DELAY)
                    continue
                    
                    
                    
                if move_dir == "right":
                    move_once("left", move_time)
                    # 进入战斗缓冲时间
                    time.sleep(2)
                    # 1. 战斗检测：遇到战斗等待结束
                    if is_in_battle():
                        print("🔴 战斗中，等待结束...")
                        while is_in_battle():
                            time.sleep(1)
                        print("🟢 战斗结束，等待返回地图界面...")
                        time.sleep(BATTLE_END_DELAY)
                        continue
                else:
                    move_once("right", move_time)
                    # 进入战斗缓冲时间
                    time.sleep(2)
                    # 1. 战斗检测：遇到战斗等待结束
                    if is_in_battle():
                        print("🔴 战斗中，等待结束...")
                        while is_in_battle():
                            time.sleep(1)
                        print("🟢 战斗结束，等待返回地图界面...")
                        time.sleep(BATTLE_END_DELAY)
                        continue
                
            else:
                # k键退回
                presskey_times("k", 5)
                # 归位 and 显示信息
                print(f"已挖到{num_mine}矿")
                pos_x, pos_y = get_twonumberby_torch()
                if pos_x != set_posx or pos_y != set_posy:
                    md(set_posx, set_posy)
                
                            
    
    except KeyboardInterrupt:
        print("\n🛑 脚本已手动停止（Ctrl+C）")
    except Exception as e:
        print(f"\n❌ 脚本异常终止：{str(e)}")
                
def dig_mine1():
    """
    蹲矿
    :return:
    """
    x, y = 2405, 815
    target_color = (198, 101, 0)
    set_posx, set_posy = -2, -12

    num_mine = 0

    while True:
        # 带容错值（允许微小色差，适合屏幕色差场景）
        is_match_tolerance = pyautogui.pixelMatchesColor(x, y, target_color, tolerance=10)
        print(f"带容错的颜色匹配：{is_match_tolerance}")
        if not is_match_tolerance:
            move_once("up", 0.1)
            presskey_times("j")
            time.sleep(0.5)
            presskey_times("j")
            time.sleep(6)
            presskey_times("k", 5)
            num_mine += 1
            print(f"已挖到矿：{num_mine}")
            is_match_tolerance_ = pyautogui.pixelMatchesColor(x, y, target_color, tolerance=10)
            if not is_match_tolerance_:
                print("位置异常")
                move_once("down", 0.5)
                num_mine -= 1
                pos_x, pos_y = get_twonumberby_torch()
                if pos_x != set_posx or pos_y != set_posy:
                    md(set_posx, set_posy, 'x')

        pos_x, pos_y = get_twonumberby_torch()
        if pos_x != set_posx or pos_y != set_posy:
            move_once("down", 0.5)
            md(set_posx, set_posy, 'x')

def dig_mine2():
    set_posx, set_posy = -2, -13
    num_mine = 0
    print_time = time.time()
    while True:
        current_time = time.time()
        if current_time - print_time > 60:
            print(f"已挖到矿：{num_mine}")
            print_time = time.time()

        if is_in_mine():
            try:
                mine_x, mine_y = get_single_template_center(MINE_TEMPLATE, MINE_THRESHOLD)
            except TypeError:
                continue

            if get_single_template_center(".\\tan_template\\left.png"):
                try:
                    player_x, player_y = get_single_template_center(".\\tan_template\\left.png")
                except TypeError:
                    continue
            elif get_single_template_center(".\\tan_template\\up.png"):
                try:
                    player_x, player_y = get_single_template_center(".\\tan_template\\up.png")
                except TypeError:
                    continue
            elif get_single_template_center(".\\tan_template\\down.png"):
                try:
                    player_x, player_y = get_single_template_center(".\\tan_template\\down.png")
                except TypeError:
                    continue
            elif get_single_template_center(".\\tan_template\\right.png"):
                try:
                    player_x, player_y = get_single_template_center(".\\tan_template\\right.png")
                except TypeError:
                    continue
            else:
                time.sleep(0.5)
                # k键退回
                presskey_times("k", 5)
                dir_move = random.choice(["left", "right"])
                move_once(dir_move, 0.2)
                print(f"疑似遮挡，已向{dir_move}移动0.2s")
                continue


            if abs(player_x - mine_x) > 100:
                move_once("right", 2)
                presskey_times("j")
                time.sleep(0.5)
                presskey_times("j")
                time.sleep(6)
                presskey_times("k", 5)
                num_mine += 1
                md(set_posx, set_posy, 'x')
            else:
                move_once("up", 0.8)
                presskey_times("j")
                time.sleep(0.5)
                presskey_times("j")
                time.sleep(6)
                presskey_times("k", 5)
                num_mine += 1
                md(set_posx, set_posy, 'y')
        else:
            pos_x, pos_y = get_twonumberby_torch()
            if pos_x != set_posx or pos_y != set_posy:
                print("等待位置异常")
                move_once("left", 0.3)
                md(set_posx, set_posy, 'y')


def get_pex():
    """
    获得坐标位置以及像素颜色
    :return:
    """
    print("按 Ctrl+C 停止程序")
    try:
        while True:
            time.sleep(1)
            # 获取鼠标当前坐标
            x, y = pyautogui.position()
            # 获取当前坐标颜色
            color = pyautogui.pixel(x, y)
            # 清空行并打印
            print(f"鼠标位置：({x}, {y}) | 颜色 RGB：{color}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n程序已停止")


if __name__ == "__main__":
    main()
    # dig_mine()
    # dig_mine1()
    # get_pex()
    # dig_mine2()