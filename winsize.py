import win32gui
import win32con
import cv2
import numpy as np
import pyautogui
import time
import random

def set_window_size(window_title, width, height, x=0, y=0):
    """
    设置指定窗口的大小和位置
    :param window_title: 窗口标题（支持模糊匹配，如"记事本"）
    :param width: 窗口宽度（像素）
    :param height: 窗口高度（像素）
    :param x: 窗口左上角x坐标（默认0）
    :param y: 窗口左上角y坐标（默认0）
    """
    # 1. 根据窗口标题查找窗口句柄（模糊匹配）
    def get_window_handle(partial_title):
        hwnd_list = []
        # 枚举所有窗口，筛选匹配标题的窗口
        def callback(hwnd, extra):
            if partial_title in win32gui.GetWindowText(hwnd):
                hwnd_list.append(hwnd)
            return True
        win32gui.EnumWindows(callback, None)
        return hwnd_list[0] if hwnd_list else None

    # 2. 获取窗口句柄
    hwnd = get_window_handle(window_title)
    if not hwnd:
        print(f"未找到标题包含「{window_title}」的窗口")
        return

    # 3. 先恢复窗口（避免最大化/最小化状态下设置无效）
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # 4. 设置窗口位置和大小（x, y, 宽, 高）
    # SWP_NOZORDER：不改变窗口层级；SWP_NOACTIVATE：不激活窗口
    win32gui.SetWindowPos(
        hwnd,
        0,
        x, y, width, height,
        win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
    )
    print(f"窗口「{window_title}」已设置为 {width}×{height}，位置({x},{y})")

# ==================== 示例调用 ====================
if __name__ == "__main__":
    # 主机
    WINDOWS_ZZJB = [2380, 1400, 1376, 170]
    WINDOWS_MLH = [1180, 1760, 130, 170]
    # 虚拟机
    # WINDOWS_ZZJB = [850, 600, 500, 50]
    # WINDOWS_MLH = [300, 650, 15, 50]
    set_window_size("重装机兵:墟", WINDOWS_ZZJB[0], WINDOWS_ZZJB[1], WINDOWS_ZZJB[2], WINDOWS_ZZJB[3])  # 设置窗口为指定大小和位置
    set_window_size("命令提示符", WINDOWS_MLH[0], WINDOWS_MLH[1], WINDOWS_MLH[2], WINDOWS_MLH[3])  # 设置窗口为指定大小和位置

    # print("运行")
    # set_window_size("Notepad++",800, 650, 15, 50)
