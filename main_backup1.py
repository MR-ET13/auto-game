import cv2
import numpy as np
import pyautogui
import time
import random

# ==================== 全局配置（新增障碍物规避参数） ====================
# 原有配置保留，新增：
OBSTACLE_TOLERANCE = 5  # 移动后坐标变化小于该值，判定为遇到障碍物（像素）
OBSTACLE_MAX_RETRY = 3  # 同一方向连续撞墙最大重试次数
OBSTACLE_DELAY = 0.5    # 绕开障碍物时的停顿时间
NAVIGATE_MAX_STUCK = 10 # 导航整体卡住的最大重试次数（避免无限循环）

# 其余原有配置（MOVE_OBJ_TEMPLATES、TARGET_OBJ_TEMPLATE_PATH等）不变...

# ==================== 原有基础方法（capture_screen/is_in_battle等）不变... ====================

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
        move_obj_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
        if move_obj_result is None:
            print("🔄 未检测到自身，1秒后重试...\n")
            time.sleep(1)
            stuck_count += 1
            continue
        move_x, move_y, current_shape = move_obj_result

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
        print(f"\n📏 偏移量：X={dx:.0f}，Y={dy:.0f} | 当前形态：{current_shape}")

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
            after_move_result = get_multi_template_center(MOVE_OBJ_TEMPLATES)
            if after_move_result is None:
                print(f"⚠️  移动【{dir}】后未检测到自身，跳过有效性检测")
                current_block_retry += 1
                stuck_count += 1
                continue
            after_x, after_y, _ = after_move_result

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

# ==================== 其余方法（main/get_multi_template_center等）不变... ====================