import cv2
import numpy as np
import pyautogui
import time
import random

# ==================== 全局配置（新增自身多模板路径） ====================
# 形态与移动方向的绑定关系（核心：移动方向对应唯一形态）
SHAPE_DIR_MAP = {
    "up": "up",      # 按W键上移 → 匹配向上形态
    "down": "down",  # 按S键下移 → 匹配向下形态
    "left": "left",  # 按A键左移 → 匹配向左形态
    "right": "right" # 按D键右移 → 匹配向右形态
}
# 防误匹配：仅在自身历史坐标±范围內匹配（排除其他玩家）
COORD_FILTER_RANGE = 150  # 像素范围（根据游戏视野调整）
# 原有配置保留，新增：
OBSTACLE_TOLERANCE = 5  # 移动后坐标变化小于该值，判定为遇到障碍物（像素）
OBSTACLE_MAX_RETRY = 1  # 同一方向连续撞墙最大重试次数
OBSTACLE_DELAY = 0.5    # 绕开障碍物时的停顿时间
NAVIGATE_MAX_STUCK = 10 # 导航整体卡住的最大重试次数（避免无限循环）
# 原有配置保留，新增：自身对象4个形态模板路径
MOVE_OBJ_TEMPLATES = {
    "up": "tantop.png",    # 向上形态模板
    "down": "tandown.png",# 向下形态模板
    "left": "tanleft.png",# 向左形态模板
    "right": "tanright_template.png"# 向右形态模板
}
TARGET_OBJ_TEMPLATE_PATH = "tong_template.png"  # 目标对象模板（静止）
# 其他原有配置（阈值、按键、时长等）不变
MATCH_THRESHOLD = 0.75
NAVIGATE_THRESHOLD = 0.8
OFFSET_TOLERANCE_X = (80 + 64 + 5)/2  # 像素阈值X
OFFSET_TOLERANCE_Y = (80 + 53 + 5)/2  # 像素阈值Y
MOVE_UP_KEY = "w"
MOVE_DOWN_KEY = "s"
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"
MOVE_DURATION = 0.2
SHORT_MOVE_DURATION = 0.1
BATTLE_TEMPLATE_PATH = "battle_template2.png"
BATTLE_END_DELAY = 3.0
INTERFACE_DELAY = 3.0
NO_BATTLE_TIMEOUT = 20.0
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True

# ==================== 原有基础方法（保留，仅修改get_obj_center） ====================
def capture_screen():
    """截取全屏并转换为OpenCV的BGR格式"""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def is_in_battle():
    """检测是否处于战斗界面，返回布尔值"""
    try:
        template = cv2.imread(BATTLE_TEMPLATE_PATH, 0)
        if template is None:
            raise FileNotFoundError(f"模板文件不存在：{BATTLE_TEMPLATE_PATH}")
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        # 规范解包：min_val(弃用), max_val(核心), min_loc(弃用), max_loc(弃用)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= MATCH_THRESHOLD:
            print(f"✅ 检测到战斗界面（匹配度：{max_val:.2f}）")
            return True
        return False
    except Exception as e:
        print(f"⚠️ 战斗检测出错：{e}")
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

# ==================== 核心优化：定向形态匹配 + 坐标过滤防误匹配 ====================
def get_target_shape_center(expected_shape, last_xy, template_dict, threshold=NAVIGATE_THRESHOLD):
    """
    定向匹配指定形态（优先）+ 坐标范围过滤，防止匹配到其他玩家
    :param expected_shape: 预期形态（如"right"，由移动方向决定）
    :param last_xy: 上一次的自身坐标 (x,y)，用于范围过滤
    :param template_dict: 形态-模板路径字典
    :param threshold: 匹配阈值
    :return: (center_x, center_y, matched_shape) 或 None
    """
    try:
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_match = None
        last_x, last_y = last_xy

        # 第一步：优先匹配预期形态（移动后应该的形态）
        if expected_shape in template_dict:
            template_path = template_dict[expected_shape]
            template = cv2.imread(template_path, 0)
            if template is not None:
                h, w = template.shape[:2]
                result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # 坐标过滤：仅保留历史坐标±COORD_FILTER_RANGE内的匹配
                match_x = max_loc[0] + w//2
                match_y = max_loc[1] + h//2
                if (abs(match_x - last_x) <= COORD_FILTER_RANGE and 
                    abs(match_y - last_y) <= COORD_FILTER_RANGE and 
                    max_val >= threshold):
                    best_match = (max_val, match_x, match_y, expected_shape)
                    print(f"✅ 优先匹配到预期形态【{expected_shape}】（匹配度：{max_val:.2f}），坐标：({match_x}, {match_y})")
                else:
                    print(f"⚠️  预期形态【{expected_shape}】匹配结果超出坐标范围（或匹配度不足），尝试全形态匹配")

        # 第二步：预期形态匹配失败，降级为全形态匹配（但仍过滤坐标）
        if best_match is None:
            for shape, template_path in template_dict.items():
                template = cv2.imread(template_path, 0)
                if template is None:
                    continue
                
                h, w = template.shape[:2]
                result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # 坐标过滤：排除其他玩家
                match_x = max_loc[0] + w//2
                match_y = max_loc[1] + h//2
                if (abs(match_x - last_x) <= COORD_FILTER_RANGE and 
                    abs(match_y - last_y) <= COORD_FILTER_RANGE and 
                    max_val >= threshold):
                    if best_match is None or max_val > best_match[0]:
                        best_match = (max_val, match_x, match_y, shape)

        # 返回最终匹配结果
        if best_match:
            max_val, center_x, center_y, shape = best_match
            print(f"📌 最终匹配到自身形态【{shape}】（匹配度：{max_val:.2f}），坐标：({center_x}, {center_y})")
            return (center_x, center_y, shape)
        else:
            print(f"❌ 未检测到有效自身形态（坐标范围：{last_x}±{COORD_FILTER_RANGE}, {last_y}±{COORD_FILTER_RANGE}）")
            return None
    except Exception as e:
        print(f"⚠️ 定向形态匹配出错：{e}")
        return None

# ==================== 核心优化：多模板匹配自身对象 ====================
def get_multi_template_center(template_dict, threshold=NAVIGATE_THRESHOLD):
    """遍历多个模板，匹配当前显示的形态并返回中心坐标"""
    try:
        frame = capture_screen()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_match = None

        for shape, template_path in template_dict.items():
            template = cv2.imread(template_path, 0)
            if template is None:
                print(f"⚠️ 跳过不存在的模板：{shape} -> {template_path}")
                continue

            h, w = template.shape[:2]
            result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
            # 规范解包：明确每个返回值的含义
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                if best_match is None or max_val > best_match[0]:
                    best_match = (max_val, center_x, center_y, shape)

        if best_match:
            max_val, center_x, center_y, shape = best_match
            print(f"📌 检测到自身形态【{shape}】（匹配度：{max_val:.2f}），中心坐标：({center_x}, {center_y})")
            return (center_x, center_y, shape)
        else:
            print(f"❌ 未检测到任何有效自身形态（阈值：{threshold}）")
            return None
    except Exception as e:
        print(f"⚠️ 多模板匹配出错：{e}")
        return None


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
    极简避绕：仅返回被阻挡方向的**垂直方向2个**，随机二选一即可
    :param blocked_dir: 被阻挡的方向（left/right/up/down）
    :return: 垂直方向列表（固定2个元素）
    """
    # 固定映射：横向挡→选上下，纵向挡→选左右（无多余逻辑）
    vertical_map = {
        "left": ["up", "down"],
        "right": ["up", "down"],
        "up": ["left", "right"],
        "down": ["left", "right"]
    }
    avoid_dirs = vertical_map[blocked_dir]
    print(f"🔀 避绕：{blocked_dir}方向被挡，仅从垂直方向{avoid_dirs}中选择")
    return avoid_dirs

# ==================== 优化后导航方法 ====================
def navigate_to_target():
    """
    核心优化：移动方向→形态绑定 + 坐标过滤，防止匹配到其他玩家；同时保留障碍物规避
    """
    print("=" * 60)
    print("📡 启动自动导航（定向形态匹配+防误匹配+障碍物规避）")
    print(f"🎯 防误匹配范围：±{COORD_FILTER_RANGE} 像素 | 障碍物阈值：{OBSTACLE_TOLERANCE} 像素")
    print(f"⚠️  仅匹配自身移动后的预期形态，排除其他玩家 | 紧急停止：鼠标移屏幕四角")
    print("=" * 60)

    stuck_count = 0
    # 初始化：先获取一次自身坐标（全形态），作为初始历史坐标
    init_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
    if init_result is None:
        print("❌ 初始化失败：未检测到自身，脚本退出")
        return
    last_x, last_y, last_shape = init_result
    print(f"🔄 初始化完成，初始坐标：({last_x}, {last_y})，初始形态：{last_shape}")

    while True:
        # 全局卡住保护
        if stuck_count >= NAVIGATE_MAX_STUCK:
            print(f"⚠️  导航连续卡住 {NAVIGATE_MAX_STUCK} 次，全局重置...")
            move_once(random.choice(["up", "down"]), SHORT_MOVE_DURATION)
            move_once(random.choice(["left", "right"]), SHORT_MOVE_DURATION)
            stuck_count = 0
            time.sleep(OBSTACLE_DELAY)
            # 重置后重新获取初始坐标
            init_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
            if init_result:
                last_x, last_y, last_shape = init_result
            continue

        # 战斗检测
        if is_in_battle():
            print("⚔️  检测到战斗，暂停导航...")
            while is_in_battle():
                time.sleep(1)
            print("✅ 战斗结束，恢复导航...")
            time.sleep(BATTLE_END_DELAY)
            stuck_count = 0
            # 战斗后重新获取坐标
            init_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
            if init_result:
                last_x, last_y, last_shape = init_result
            continue

        # 1. 获取目标坐标（单模板，不变）
        target_obj_xy = get_single_template_center(TARGET_OBJ_TEMPLATE_PATH)
        if target_obj_xy is None:
            print("🔄 未检测到目标，1秒后重试...\n")
            time.sleep(1)
            stuck_count += 1
            continue
        target_x, target_y = target_obj_xy

        # 2. 计算偏移量，判断是否到达目标
        dx = target_x - last_x
        dy = target_y - last_y
        print(f"\n📏 偏移量：X={dx:.0f}，Y={dy:.0f} | 上一次形态：{last_shape}")

        if abs(dx) < OFFSET_TOLERANCE_X and abs(dy) < OFFSET_TOLERANCE_Y:
            print("🎉 已到达目标对象位置，导航结束！")
            break

        # 3. 决定移动方向和时长
        move_dirs = []
        duration = MOVE_DURATION if (abs(dx) > OFFSET_TOLERANCE_X*1.5 or abs(dy) > OFFSET_TOLERANCE_Y*1.5) else SHORT_MOVE_DURATION

        if abs(dx) >= OFFSET_TOLERANCE_X:
            move_dirs.append("right" if dx > 0 else "left")
        if abs(dy) >= OFFSET_TOLERANCE_Y:
            move_dirs.append("down" if dy > 0 else "up")

        # 4. 执行移动 + 定向形态匹配（核心：防止匹配其他玩家）
        current_block_retry = 0
        for dir in move_dirs:
            # 确定本次移动的预期形态（移动方向→形态绑定）
            expected_shape = SHAPE_DIR_MAP[dir]
            print(f"\n🔹 准备移动【{dir}】，预期匹配形态【{expected_shape}】")
            
            # 记录移动前坐标
            before_move_xy = (last_x, last_y)
            
            # 执行移动
            move_once(dir, duration)
            time.sleep(0.1)

            # 核心：移动后仅匹配预期形态（+坐标过滤），防止匹配其他玩家
            after_move_result = get_target_shape_center(
                expected_shape=expected_shape,
                last_xy=before_move_xy,  # 基于移动前坐标过滤
                template_dict=MOVE_OBJ_TEMPLATES
            )

            # 处理匹配结果
            if after_move_result is None:
                print(f"⚠️  移动【{dir}】后未匹配到预期形态，尝试全形态匹配...")
                # 降级为全形态匹配（仍带坐标过滤）
                fallback_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
                if fallback_result:
                    after_x, after_y, after_shape = fallback_result
                    # 再次坐标过滤，确保不是其他玩家
                    if abs(after_x - last_x) <= COORD_FILTER_RANGE * 2 and abs(after_y - last_y) <= COORD_FILTER_RANGE * 2:
                        last_x, last_y, last_shape = after_x, after_y, after_shape
                    else:
                        print(f"🚫 全形态匹配结果超出坐标范围，放弃更新坐标")
                        current_block_retry += 1
                        stuck_count += 1
                        continue
                else:
                    print(f"❌ 全形态匹配也失败，跳过本次移动")
                    current_block_retry += 1
                    stuck_count += 1
                    continue
            else:
                after_x, after_y, after_shape = after_move_result
                # 更新历史坐标和形态
                last_x, last_y, last_shape = after_x, after_y, after_shape

            # 障碍物检测（复用原有逻辑）
            if not check_move_effectiveness(before_move_xy, (after_x, after_y)):
                current_block_retry += 1
                stuck_count += 1
                if current_block_retry >= OBSTACLE_MAX_RETRY:
                    print(f"🔴 方向【{dir}】连续撞墙，启动绕开逻辑...")
                    avoid_dirs = get_avoid_directions(dir)
                    new_dir = random.choice(avoid_dirs)
                    new_expected_shape = SHAPE_DIR_MAP[new_dir]
                    print(f"➡️  切换方向至【{new_dir}】，预期形态【{new_expected_shape}】")
                    move_once(new_dir, duration)
                    time.sleep(OBSTACLE_DELAY)
                    # 绕开后更新坐标（定向匹配）
                    avoid_result = get_target_shape_center(new_expected_shape, (last_x, last_y), MOVE_OBJ_TEMPLATES)
                    if avoid_result:
                        last_x, last_y, last_shape = avoid_result
                    current_block_retry = 0
            else:
                stuck_count = 0
                current_block_retry = 0

# ==================== 主逻辑（保留原有+优化后导航） ====================
def main():
    print("=" * 60)
    print("游戏自动遇敌+多形态导航脚本（WASD自动移到目标）")
    print(f"移动规则：脱离战斗后随机选方向，交替移动；导航时精准WASD移动")
    print(f"适配：自身对象4种形态（上下左右），目标对象静止")
    print(f"按键配置：上(W) 下(S) 左(A) 右(D) | 紧急停止：鼠标移屏幕四角")
    print("=" * 60)
    time.sleep(INTERFACE_DELAY)

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

if __name__ == "__main__":
    # main()
    time.sleep(3)
    navigate_to_target()