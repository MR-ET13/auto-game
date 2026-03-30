## 重装机兵:墟 自动化
> 游戏账号密码<br>
> 18858114669: 13795aetzzjb<br>
> 13249192199: 1234567

### main.py
左右遇敌，20s内未遇敌(判断为被组队干扰)自动取消<br>
增加move_dungeon()，直接识别两个字符加逗号，**淘汰**

### main_backup.py
move_dungeon()较main.py出众，通过识别单个数字的方法获取坐标

### main_name.py
在`main.py`的基础上，新增角色自动移动的功能<br>
识别角色头上的名字中心为坐标，识别目标中心为坐标，有阻挡检测，卡死检测

### winsize.py
设置窗口大小和位置

### move_dungeon.py
通过设定移动速度和移动距离来指定副本中的移动，目前基本调整好了速度
- <font color="red">需要新增意外模式，进入...</font>

### get_pos.py
返回世界坐标，第二个数字-8会识别出错为-4，-48<br>
新增识别单个字符的方法，通过逗号确定分割位置<br>
新增get_img()获取训练集的图片，目前还差较多数据

### ~~self_torch_self.py~~
~~训练识别数字的机器学习训练模型~~

### doubao_torch.py
机器学习训练模型，较好效果my_own_model_a1.pth
- 1容易被识别为5，通过计算像素总和区分
- 6容易被识别成5，<font color="red">需要寻找区分方式</font>，“逗号”是一个方式
- <font color="red">数字裁剪的大小有待调整</font>

### dev_sys
#### 需修改的地方
- `BATTLE_TEMPLATE_PATH = "battle_template2.png"` 战斗匹配模板
- `TEMPLATE_PATH = "number_template.png"  # 你的模板图片` “位置”定位模板图片
- `VALUE_ROI_WIDTH = 136 \n VALUE_ROI_HEIGHT = 37` 世界坐标截图的尺寸
- `SINGLE_NUMBER_PIXEL = 17` 单个数字的宽度
- `TEMP_X, TEMP_Y = 3464, 285` 临时位置