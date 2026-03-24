import cv2
import numpy as np
import pyautogui
import time
import random
from winsize import set_window_size

# ==================== 全局配置（可根据游戏调整） ====================
# 模板匹配相关
BATTLE_TEMPLATE_PATH = "battle_template1.png"  # 战斗界面的模板图路径，1表示虚拟机，2表示主机
MATCH_THRESHOLD = 0.75  # 模板匹配的置信度阈值
# 移动相关
MOVE_LEFT_KEY = "a"  # 左移按键
MOVE_RIGHT_KEY = "d"  # 右移按键
MOVE_DURATION = 1.0  # 单次移动时长（秒）
# 延迟相关
BATTLE_END_DELAY = 3.0  # 战斗结束后等待返回地图的时间
INTERFACE_DELAY = 3.0  # 脚本启动后的初始等待时间
NO_BATTLE_TIMEOUT = 20.0  # 无战斗超时阈值（秒）
# 安全配置
pyautogui.PAUSE = 0.1  # 所有pyautogui操作的间隔
pyautogui.FAILSAFE = True  # 鼠标移到屏幕四角触发紧急停止
# 窗口位置大小设置 w, h, x, y
# 主机
# WINDOWS_ZZJB = [2380, 1400, 1376, 170]
# WINDOWS_MLH = [1180, 1760, 130, 170]
# 虚拟机
WINDOWS_ZZJB = [850, 600, 500, 50]
WINDOWS_MLH = [300, 650, 15, 50]



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


def move_once(direction):
    """执行单次方向移动"""
    # 映射方向到按键和提示文本
    key_map = {"left": (MOVE_LEFT_KEY, "向左", "⬅️"), "right": (MOVE_RIGHT_KEY, "向右", "➡️")}
    key, text, icon = key_map[direction]

    print(f"{icon} 执行{text}移动，时长 {MOVE_DURATION} 秒")
    pyautogui.keyDown(key)
    time.sleep(MOVE_DURATION)
    pyautogui.keyUp(key)
    time.sleep(0.5)  # 移动后短暂停顿，模拟真人操作


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
    set_window_size("重装机兵:墟", WINDOWS_ZZJB[0], WINDOWS_ZZJB[1], WINDOWS_ZZJB[2], WINDOWS_ZZJB[3])  # 设置窗口为指定大小和位置
    set_window_size("命令提示符", WINDOWS_MLH[0], WINDOWS_MLH[1], WINDOWS_MLH[2], WINDOWS_MLH[3])  # 设置窗口为指定大小和位置

    time.sleep(INTERFACE_DELAY)  # 启动后等待，给你切换到游戏窗口的时间

    # 初始化无战斗计时
    no_battle_start_time = time.time()

    try:
        while True:
            # 1. 检测到战斗：重置计时，等待战斗结束
            if is_in_battle():
                no_battle_start_time = time.time()  # 重置无战斗计时器
                print("🔴 进入战斗状态，等待战斗结束...")
                while is_in_battle():  # 循环检测，直到脱离战斗
                    time.sleep(1)
                print("🟢 战斗结束，等待返回地图界面...")
                time.sleep(BATTLE_END_DELAY)
                no_battle_start_time = time.time()  # 战斗结束后重新计时

            # 2. 未检测到战斗：检查是否超时
            else:
                current_time = time.time()
                elapsed_time = current_time - no_battle_start_time

                # 2.1 超过20秒未战斗：执行特殊操作
                if elapsed_time > NO_BATTLE_TIMEOUT:
                    execute_timeout_operation()
                    no_battle_start_time = time.time()  # 重置计时器

                # 2.2 未超时：执行正常移动
                else:
                    print(f"⏳ 未检测到战斗，已计时 {elapsed_time:.1f} 秒（阈值：20秒）")
                    # 随机决定本次循环的第一个移动方向
                    first_dir = random.choice(["left", "right"])
                    second_dir = "right" if first_dir == "left" else "left"

                    # 执行：先随机方向，再切换方向
                    move_once(first_dir)
                    move_once(second_dir)

    except KeyboardInterrupt:
        print("\n🛑 脚本已手动停止（Ctrl+C）")
    except Exception as e:
        print(f"\n❌ 脚本异常终止：{str(e)}")


if __name__ == "__main__":
    main()