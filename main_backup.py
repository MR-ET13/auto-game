import cv2
import numpy as np
import pyautogui
import time
import random
from winsize import set_window_size
# from get_pos import get_two_numbers_from_single_roi
# from get_pos import get_two_number_from_one
from get_pos import get_twonumberby_torch

# ==================== 全局配置（新增/调整） ====================
# 模板匹配相关
BATTLE_TEMPLATE_PATH = "battle_template2.png"
MATCH_THRESHOLD = 0.75
# 移动相关
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"
MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_DURATION = 1.0
# 延迟相关
BATTLE_END_DELAY = 3.0
INTERFACE_DELAY = 3.0
NO_BATTLE_TIMEOUT = 20.0
# 地牢移动专属配置（新增）
MOVE_TOLERANCE = 0.5       # 坐标误差容忍度
BLOCK_DETECTION_THRESH = 0.1  # 移动后坐标变化小于此值判定为被阻挡
MAX_BLOCK_RETRIES = 3      # 单一方向最大阻挡重试次数
STEP_DURATION = 0.2        # 小步移动时长
STEP_DURATION_BIG = 1        # 大步移动时长
# 安全配置
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True
# 窗口配置
# 主机
WINDOWS_ZZJB = [2380, 1400, 1376, 170]
WINDOWS_MLH = [1180, 1760, 130, 170]
# 虚拟机
# WINDOWS_ZZJB = [850, 600, 500, 50]
# WINDOWS_MLH = [300, 650, 15, 50]

# ==================== 原有核心函数（保留） ====================
def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

def move_once(direction, duration=MOVE_DURATION):
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
    print("\n⏰ 超过20秒未进入战斗，执行特殊操作...")
    time.sleep(2)
    print("🔹 按下并弹起 D 键")
    pyautogui.press(MOVE_RIGHT_KEY)
    time.sleep(1)
    print("🔹 再次按下并弹起 D 键")
    pyautogui.press(MOVE_RIGHT_KEY)
    time.sleep(1)
    print("🔹 按下并弹起 J 键")
    pyautogui.press("j")
    time.sleep(1)
    print("🔹 按下并弹起 K 键")
    pyautogui.press("k")
    print("✅ 超时操作执行完成，恢复正常移动\n")

# ==================== 优化后的地牢移动函数 ====================
def move_dungeon(target_x, target_y, first='y'):
    """
    优化版地牢移动：增加阻挡检测，避免单一方向死磕
    :param target_x: 目标X坐标
    :param target_y: 目标Y坐标
    """
    print(f"\n🎯 开始移动到目标坐标：X={target_x}, Y={target_y}")
    block_retry_count = 0  # 连续阻挡重试计数器
    last_move_dir = None    # 上一次移动方向

    while True:
        # 1. 战斗检测：遇到战斗等待结束
        if is_in_battle():
            print("🔴 战斗中，等待结束...")
            while is_in_battle():
                time.sleep(1)
            print("🟢 战斗结束，等待返回地图界面...")
            time.sleep(BATTLE_END_DELAY)
            break

        # 2. 获取当前坐标
        current_x, current_y = get_twonumberby_torch()
        if current_x is None or current_y is None:
            print("⚠️ 坐标获取失败，随机移动，重试...")
            alt_dir = random.choice(["up", "down", "left", "right"])
            move_once(alt_dir, duration=STEP_DURATION * 3)
            time.sleep(0.5)
            continue

        # 3. 到达目标判定
        if abs(current_x - target_x) <= MOVE_TOLERANCE and abs(current_y - target_y) <= MOVE_TOLERANCE:
            print(f"✅ 已到达目标坐标！当前({current_x:.1f}, {current_y:.1f}) | 目标({target_x}, {target_y})")
            break

        # 4. 计算坐标差值，确定优先移动方向
        dx = target_x - current_x
        dy = target_y - current_y
        move_dir = None

        # 优先处理y轴（竖直）
        if first == 'y':
            if abs(dy) > MOVE_TOLERANCE:
                move_dir = "up" if dy > 0 else "down"
            elif abs(dx) > MOVE_TOLERANCE:
                move_dir = "right" if dx > 0 else "left"
        else:
            if abs(dx) > MOVE_TOLERANCE:
                move_dir = "right" if dx > 0 else "left"
            elif abs(dy) > MOVE_TOLERANCE:
                move_dir = "up" if dy > 0 else "down"

        # 5. 阻挡检测：记录移动前坐标，执行移动后对比
        pre_move_x, pre_move_y = current_x, current_y
        if move_dir:
            print(f"📌 当前({current_x:.1f}, {current_y:.1f}) | 目标偏差 X:{dx:.1f}, Y:{dy:.1f} | 计划移动：{move_dir}")
            if move_dir in ["left", "right"]:
                duration = STEP_DURATION_BIG if abs(dx) > 5  else STEP_DURATION
                move_once(move_dir, duration)
                time.sleep(2)
                # 1. 战斗检测：遇到战斗等待结束
                if is_in_battle():
                    print("🔴 战斗中，等待结束...")
                    while is_in_battle():
                        time.sleep(1)
                    print("🟢 战斗结束，等待返回地图界面...")
                    time.sleep(BATTLE_END_DELAY)
                    break
            else:
                duration = STEP_DURATION_BIG if abs(dy) > 5 else STEP_DURATION
                move_once(move_dir, duration)
                time.sleep(2)
                # 1. 战斗检测：遇到战斗等待结束
                if is_in_battle():
                    print("🔴 战斗中，等待结束...")
                    while is_in_battle():
                        time.sleep(1)
                    print("🟢 战斗结束，等待返回地图界面...")
                    time.sleep(BATTLE_END_DELAY)
                    break

            time.sleep(0.1)  # 移动后等待坐标更新

            # 获取移动后坐标
            post_x, post_y = get_twonumberby_torch()
            if post_x is None or post_y is None:
                post_x, post_y = pre_move_x, pre_move_y

            # 计算移动偏移量
            move_delta_x = abs(post_x - pre_move_x)
            move_delta_y = abs(post_y - pre_move_y)
            total_delta = np.hypot(move_delta_x, move_delta_y)

            # 判定是否被阻挡
            if total_delta < BLOCK_DETECTION_THRESH:
                block_retry_count += 1
                print(f"🚫 检测到阻挡！{move_dir}方向移动无效（偏移量：{total_delta:.2f}）| 连续阻挡次数：{block_retry_count}")

                # 达到最大重试次数，切换绕行方向
                if block_retry_count >= MAX_BLOCK_RETRIES:
                    print(f"🔀 连续{MAX_BLOCK_RETRIES}次阻挡，切换绕行方向")
                    # 原方向是水平→切换垂直，原方向是垂直→切换水平
                    if move_dir in ["left", "right"]:
                        # 随机选上下
                        alt_dir = random.choice(["up", "down"])
                    else:
                        # 随机选左右
                        alt_dir = random.choice(["left", "right"])
                    move_once(alt_dir, duration=STEP_DURATION * 3)  # 绕行移动（加长时长）
                    block_retry_count = 0  # 重置计数器
                    last_move_dir = alt_dir
            else:
                # 移动有效，重置阻挡计数器
                block_retry_count = 0
                last_move_dir = move_dir
        else:
            # 无有效移动方向（理论上不会触发）
            print(f"⚠️ 无有效移动方向！当前({current_x:.1f}, {current_y:.1f}) | 目标({target_x}, {target_y})")
            time.sleep(0.5)



# ==================== 主逻辑（保留） ====================
def main():
    print("=" * 60)
    print("游戏自动遇敌脚本（随机起始方向 + 20秒无战斗触发特殊操作）")
    print(f"移动规则：脱离战斗后随机选方向，然后交替移动")
    print(f"超时规则：20秒未战斗 → 2s延时→D→1s→D→1s→J→1s→K")
    print(f"按键配置：左({MOVE_LEFT_KEY}) 右({MOVE_RIGHT_KEY}) | 紧急停止：鼠标移屏幕四角")
    print("=" * 60)
    set_window_size("重装机兵:墟", WINDOWS_ZZJB[0], WINDOWS_ZZJB[1], WINDOWS_ZZJB[2], WINDOWS_ZZJB[3])
    set_window_size("Windows PowerShell", WINDOWS_MLH[0], WINDOWS_MLH[1], WINDOWS_MLH[2], WINDOWS_MLH[3])
    time.sleep(INTERFACE_DELAY)

    no_battle_start_time = time.time()
    try:
        while True:
            if is_in_battle():
                print("🔴 进入战斗状态，等待战斗结束...")
                while is_in_battle():
                    time.sleep(1)
                print("🟢 战斗结束，等待返回地图界面...")
                time.sleep(BATTLE_END_DELAY)
                no_battle_start_time = time.time()
            else:
                current_time = time.time()
                elapsed_time = current_time - no_battle_start_time
                if elapsed_time > NO_BATTLE_TIMEOUT:
                    execute_timeout_operation()
                    no_battle_start_time = time.time()
                else:
                    print(f"⏳ 未检测到战斗，已计时 {elapsed_time:.1f} 秒（阈值：20秒）")
                    first_dir = random.choice(["left", "right"])
                    second_dir = "right" if first_dir == "left" else "left"
                    move_once(first_dir)
                    move_once(second_dir)
    except KeyboardInterrupt:
        print("\n🛑 脚本已手动停止（Ctrl+C）")
    except Exception as e:
        print(f"\n❌ 脚本异常终止：{str(e)}")

def presskey_times(key, times=1):
    for _ in range(times):
        pyautogui.press(key)
        time.sleep(0.5)

if __name__ == "__main__":
    # main()
    while True:
        # move_dungeon(-8, -5, 'y')

        # move_dungeon(0, -11, 'y')
        #
        # move_dungeon(17, -11, 'x')
        #
        # move_dungeon(25, -8, 'x')
        #
        # move_dungeon(-15, -11, 'y')
        # move_dungeon(-15, -8, 'y')

        # move_dungeon(-24, -6, 'x')
        # move_dungeon(-25, -3, 'x')

        move_dungeon(-25, 9, 'x')
        move_dungeon(-22, 9, 'x')

        move_dungeon(-20, 15, 'x')
        move_dungeon(-15, 15, 'x')

        move_dungeon(-20, 8, 'x')
        move_dungeon(-10, 8, 'x')

        move_dungeon(-8, 9, 'x')
        move_dungeon(-3, 9, 'x')

        move_dungeon(8, 8, 'x')

        move_dungeon(13, 4, 'y')
        move_dungeon(13, -2, 'y')

        move_dungeon(13, 6, 'x')


        move_once("right", 0.5)
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


