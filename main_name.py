import cv2
import numpy as np
import pyautogui
import time
import random
from winsize import set_window_size

# ==================== 全局配置（可根据游戏调整） ====================

# ==================== 【新增：SKT名字白色文字模板配置】 ====================
SKT_TEMPLATE_PATH = "waet_template.png"  # 你的skt小截图
try:
    # 自动提取白色文字，无视背景
    _tpl = cv2.imread(SKT_TEMPLATE_PATH)
    _tpl_gray = cv2.cvtColor(_tpl, cv2.COLOR_BGR2GRAY)
    _, SKT_WHITE_MASK = cv2.threshold(_tpl_gray, 220, 255, cv2.THRESH_BINARY)  # 220需要和匹配函数参数一致
    SKT_H, SKT_W = SKT_WHITE_MASK.shape[:2]
except:
    SKT_WHITE_MASK = None
    SKT_H = SKT_W = 0

# 移动相关
SKT_MATCH = 0.5  # skt名字匹配阈值
MATCH_THRESHOLD = 0.75  # 战斗匹配阈值
NAVIGATE_THRESHOLD = 0.8  #  移动目标匹配阈值
OFFSET_TOLERANCE = 50  # 移动目标像素差
MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"
MOVE_DURATION = 0.1  # 远离目标移动时间
SHORT_MOVE_DURATION = 0.05  # 靠近目标移动时间
BATTLE_TEMPLATE_PATH = "battle_template1.png"  # 战斗匹配模板
BATTLE_END_DELAY = 3.0  # 战斗结束延时
INTERFACE_DELAY = 3.0  # 程序启动保护延时
NO_BATTLE_TIMEOUT = 20.0  # 无战斗计时，避免组队干扰
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True
# 窗口位置和尺寸设置
WINDOWS_ZZJB = [2380, 1400, 1376, 170]
WINDOWS_MLH = [1180, 1760, 130, 170]

# 原有配置保留，新增：
OBSTACLE_TOLERANCE = 5  # 移动后坐标变化小于该值，判定为遇到障碍物（像素）
OBSTACLE_MAX_RETRY = 3  # 同一方向连续撞墙最大重试次数
OBSTACLE_DELAY = 0.5    # 绕开障碍物时的停顿时间
NAVIGATE_MAX_STUCK = 10 # 导航整体卡住的最大重试次数（避免无限循环）

TARGET_OBJ_TEMPLATE_PATH = "target_template.png"  # 目标对象模板（静止）

# ==================== 核心功能函数 ====================
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
            return True
        return False
    except:
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
    time.sleep(2)
    pyautogui.press(MOVE_RIGHT_KEY)
    time.sleep(1)
    pyautogui.press(MOVE_RIGHT_KEY)
    time.sleep(1)
    pyautogui.press("j")
    time.sleep(1)
    pyautogui.press("k")

# ==================== 【新增：识别玩家头顶白色名字 skt】 ====================
def find_skt_center(threshold=SKT_MATCH):
    """
    识别白色文字 skt，返回中心坐标 (cx, cy)
    无视模板背景、无视界面背景
    """
    if SKT_WHITE_MASK is None:
        print("⚠️ 未加载skt模板")
        return None

    frame = capture_screen()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_white = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    result = cv2.matchTemplate(frame_white, SKT_WHITE_MASK, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # y_loc, x_loc = np.where(result >= threshold)
    #
    # for x, y in zip(x_loc, y_loc):
    #     cx = x + SKT_W // 2
    #     cy = y + SKT_H // 2
    #     return (cx, cy)
    # return None
    h, w = SKT_WHITE_MASK.shape[:2]
    if max_val >= threshold:
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        print(f"🎯 检测到移动对象（匹配度：{max_val:.2f}），中心坐标：({center_x}, {center_y})")
        return (center_x, center_y)
    else:
        print(f"❌ 未检测到移动对象，匹配度：{max_val:.2f}（阈值：{threshold}）")
        return None

# ==================== 【新增：可视化验证（不影响主逻辑）】 ====================
def show_skt_position():
    center = find_skt_center()
    frame = capture_screen()
    if center:
        cx, cy = center
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"SKT {cx},{cy}", (cx+10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
    cv2.imshow("SKT Detect", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_single_template_center(template_path, threshold=NAVIGATE_THRESHOLD):
    """单模板匹配，返回中心坐标"""
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
# ==================== 核心优化：新增移动有效性检测 + 障碍物规避 ====================
def check_move_effectiveness(before_xy, after_xy, tolerance=OBSTACLE_TOLERANCE):
    """
    检测移动是否有效（是否遇到障碍物）
    :param before_xy: 移动前的坐标 (x, y)
    :param after_xy: 移动后的坐标 (x, y)
    :param tolerance: 坐标变化容忍度（像素）
    :return: True=移动有效（无障碍物），False=移动无效（有障碍物）
    """
    dx = abs(after_xy[0] - before_xy[0])
    dy = abs(after_xy[1] - before_xy[1])
    is_effective = dx > tolerance or dy > tolerance
    if not is_effective:
        print(f"🚫 检测到障碍物：移动后坐标变化 X={dx}，Y={dy}（容忍度：{tolerance}）")
    return is_effective

def get_avoid_directions(blocked_dir):
    """
    获取绕开障碍物的备选方向（排除被阻挡的方向）
    :param blocked_dir: 被阻挡的方向（如"right"）
    :return: 备选方向列表
    """
    all_dirs = ["up", "down", "left", "right"]
    avoid_dirs = [d for d in all_dirs if d != blocked_dir]
    # 优先选择与阻挡方向垂直的方向（如右被挡，优先上下），提高绕开效率
    vertical_map = {
        "left": ["up", "down"],
        "right": ["up", "down"],
        "up": ["left", "right"],
        "down": ["left", "right"]
    }
    priority_dirs = vertical_map.get(blocked_dir, avoid_dirs)
    # 合并优先方向+剩余方向，保证有备选
    final_dirs = priority_dirs + [d for d in avoid_dirs if d not in priority_dirs]
    print(f"🔀 绕开障碍物：阻挡方向【{blocked_dir}】，备选方向：{final_dirs}")
    return final_dirs

# ==================== 优化后导航方法（解决障碍物卡住问题） ====================
def navigate_to_target():
    """
    适配多形态+障碍物规避的导航：遇到障碍物自动绕开，避免卡住
    """
    print("=" * 60)
    print("📡 启动自动导航（多形态+障碍物规避）：移动对象 → 目标对象")
    print(f"🎯 偏移容忍度：{OFFSET_TOLERANCE} | 障碍物检测阈值：{OBSTACLE_TOLERANCE} 像素")
    print(f"⚠️  检测到战斗/障碍物会自动处理 | 紧急停止：鼠标移屏幕四角")
    print("=" * 60)

    stuck_count = 0  # 整体卡住计数
    while True:
        # 全局卡住保护：超过阈值则退出或重置
        if stuck_count >= NAVIGATE_MAX_STUCK:
            print(f"⚠️  导航连续卡住 {NAVIGATE_MAX_STUCK} 次，尝试全局重置...")
            # 随机移动两步，脱离卡住状态
            move_once(random.choice(["up", "down"]), SHORT_MOVE_DURATION)
            move_once(random.choice(["left", "right"]), SHORT_MOVE_DURATION)
            stuck_count = 0
            time.sleep(OBSTACLE_DELAY)
            continue

        # 战斗检测：暂停导航
        if is_in_battle():
            print("⚔️  检测到战斗，暂停导航，等待战斗结束...")
            while is_in_battle():
                time.sleep(1)
            print("✅ 战斗结束，恢复导航...")
            time.sleep(BATTLE_END_DELAY)
            stuck_count = 0  # 战斗后重置卡住计数
            continue

        # 1. 获取自身坐标（多模板匹配）
        move_obj_result = find_skt_center()
        if move_obj_result is None:
            print("🔄 未检测到自身，1秒后重试...\n")
            time.sleep(1)
            stuck_count += 1
            continue
        move_x, move_y = move_obj_result

        # 2. 获取目标坐标（单模板）
        target_obj_xy = get_single_template_center(TARGET_OBJ_TEMPLATE_PATH)
        if target_obj_xy is None:
            print("🔄 未检测到目标，1秒后重试...\n")
            time.sleep(1)
            stuck_count += 1
            continue
        target_x, target_y = target_obj_xy

        # 3. 计算偏移量，判断是否到达目标
        dx = target_x - move_x  # X偏移：正=目标在右，负=目标在左
        dy = target_y - move_y  # Y偏移：正=目标在下，负=目标在上
        print(f"\n📏 偏移量：X={dx:.0f}，Y={dy:.0f}")

        if abs(dx) < OFFSET_TOLERANCE and abs(dy) < OFFSET_TOLERANCE:
            print("🎉 已到达目标对象位置，导航结束！")
            break

        # 4. 决定移动方向和时长
        move_dirs = []
        duration = MOVE_DURATION if (abs(dx) > OFFSET_TOLERANCE*1.5 or abs(dy) > OFFSET_TOLERANCE*1.5) else SHORT_MOVE_DURATION

        # 横向移动
        if abs(dx) >= OFFSET_TOLERANCE:
            move_dirs.append("right" if dx > 0 else "left")
        # 纵向移动
        if abs(dy) >= OFFSET_TOLERANCE:
            move_dirs.append("down" if dy > 0 else "up")

        # 5. 执行移动 + 障碍物检测
        current_block_retry = 0  # 单次方向撞墙重试计数
        for dir in move_dirs:
            # 记录移动前坐标（用于检测是否撞墙）
            before_move_xy = (move_x, move_y)
            
            # 执行移动
            move_once(dir, duration)
            time.sleep(0.1)  # 移动后停顿，确保坐标更新

            # 重新获取移动后坐标
            after_move_result = find_skt_center()
            if after_move_result is None:
                print(f"⚠️  移动【{dir}】后未检测到自身，跳过有效性检测")
                current_block_retry += 1
                stuck_count += 1
                continue
            after_x, after_y= after_move_result

            # 检测是否遇到障碍物
            if not check_move_effectiveness(before_move_xy, (after_x, after_y)):
                current_block_retry += 1
                stuck_count += 1

                # 同一方向连续撞墙，触发绕开逻辑
                if current_block_retry >= OBSTACLE_MAX_RETRY:
                    print(f"🔴 方向【{dir}】连续撞墙 {OBSTACLE_MAX_RETRY} 次，启动绕开逻辑...")
                    # 获取备选方向并随机选择一个
                    avoid_dirs = get_avoid_directions(dir)
                    new_dir = random.choice(avoid_dirs)
                    # 执行绕开移动
                    print(f"➡️  切换方向至【{new_dir}】，尝试绕开障碍物...")
                    move_once(new_dir, duration)
                    time.sleep(OBSTACLE_DELAY)
                    current_block_retry = 0  # 重置单次撞墙计数
            else:
                # 移动有效，重置卡住计数
                stuck_count = 0
                current_block_retry = 0

            # 更新当前坐标为移动后坐标
            move_x, move_y = after_x, after_y

# ==================== 主逻辑（完全没改动！） ====================
def main():
    print("=" * 60)
    print("游戏自动遇敌脚本")
    print("=" * 60)
    set_window_size("重装机兵:墟", WINDOWS_ZZJB[0], WINDOWS_ZZJB[1], WINDOWS_ZZJB[2], WINDOWS_ZZJB[3])
    set_window_size("命令提示符", WINDOWS_MLH[0], WINDOWS_MLH[1], WINDOWS_MLH[2], WINDOWS_MLH[3])
    time.sleep(INTERFACE_DELAY)
    no_battle_start_time = time.time()

    try:
        # 先执行多形态适配的导航
        navigate_to_target()
        # 导航完成后执行原有遇敌逻辑
        print("\n📌 导航完成，启动原有自动遇敌逻辑...")
        no_battle_start_time = time.time()
        while True:
            if is_in_battle():
                no_battle_start_time = time.time()
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
                    first_dir = random.choice(["left", "right", "up", "down"])
                    second_dir = random.choice([d for d in ["left", "right", "up", "down"] if d != first_dir])
                    move_once(first_dir)
                    move_once(second_dir)
    except KeyboardInterrupt:
        print("\n🛑 脚本已手动停止（Ctrl+C）")
    except Exception as e:
        print(f"\n❌ 脚本异常终止：{str(e)}")

# ==================== 入口 ====================
if __name__ == "__main__":
    # 原来的功能不变
    main()

    # 如果你想测试识别，注释上面 main()，打开下面这句
    # show_skt_position()