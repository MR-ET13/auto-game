import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# ======================
# 1. 类别定义（保持不变）
# 新增识别1的卷积层，效果更差
# ======================
CLASSES = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


# ======================
# 2. 优化后的模型结构
# 关键改进：
# - 增加卷积层深度 + 批归一化
# - 增加Dropout防止过拟合
# - 调整全连接层维度
# ======================
class DigitSymbolModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层（增加批归一化+更多通道）
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 新增识别1的卷积层
        self.conv4 = nn.Conv2d(128, 128, 1, 1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层（增加Dropout）
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 28→14→7→3 (三次池化后尺寸)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        # 卷积层：Conv → BN → ReLU → Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14→7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7→3

        # 启用识别1的卷积层
        x = F.relu(self.conv4(x))

        # 展平
        x = x.flatten(1)

        # 全连接层：Linear → BN → ReLU → Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


# ======================
# 3. 优化后的数据集（增加数据增强）
# 关键改进：
# - 随机旋转、平移、缩放（模拟真实场景的形变）
# - 随机反转（数字对称不影响识别）
# ======================
class MyOwnDataset(Dataset):
    def __init__(self, root_dir="my_dataset", train=True):
        self.root = root_dir
        self.train = train

        # 训练集：数据增强；测试集：仅基础变换
        if train:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                # 数据增强（核心）
                transforms.RandomRotation(10),  # 随机旋转±10度
                transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移±10%
                transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),  # 随机缩放
                transforms.RandomHorizontalFlip(p=0.1),  # 少量水平翻转
                # 基础变换
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.files = []
        for cls in CLASSES:
            folder = os.path.join(root_dir, cls)
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    self.files.append((os.path.join(folder, fname), cls))

        # 数据集洗牌（避免类别连续）
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, cls = self.files[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)
        label = CLASS_TO_IDX[cls]
        return img, torch.tensor(label, dtype=torch.long)


# ======================
# 4. 优化后的训练函数
# 关键改进：
# - 划分训练/验证集（监控过拟合）
# - 学习率调度器（动态调整LR）
# - 早停机制（避免无效训练）
# - 验证集准确率监控
# ======================
def train_my_model():
    full_dataset = MyOwnDataset(train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # ✅ 正常、干净、稳定的加载器（无任何加权）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitSymbolModel().to(device)

    # ✅ 普通损失函数，不加权！最稳！
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    patience = 4
    max_epochs = 30

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                _, predicted = torch.max(pred, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../my_own_model.pth")
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                print("早停 | 最佳准确率:", best_val_acc)
                break

    return model


# ======================
# 5. 预测函数（小幅优化）
# 关键改进：
# - 增加边界膨胀（避免字符切割不完整）
# - 统一使用GPU推理
# ======================
def predict_number(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    img_bin = (img_np > 127).astype(np.uint8) * 255

    # 列投影找字符区域
    h_proj = np.sum(img_bin, axis=0)
    regions = []
    start = None
    for i, val in enumerate(h_proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            # 边界膨胀（避免切割太窄）
            start = max(0, start - 1)
            end = min(img.width, i + 1)
            regions.append((start, end))
            start = None
    # 处理最后一个区域
    if start is not None:
        regions.append((max(0, start - 1), img.width))

    chars = []
    with torch.no_grad():
        for l, r in regions:
            char_img = img.crop((l, 0, r, img.height))
            tensor = transform(char_img).unsqueeze(0).to(device)
            out = model(tensor)
            idx = torch.argmax(out).item()
            chars.append(IDX_TO_CLASS[idx])

    return ''.join(chars)

def get_numberbytorch(image_path, model_path):
    """
    通过训练的模型识别数字
    :param image_path: 图片路径
    :param model_path: 训练好的模型路径
    :return: 识别结果
    """
    model = DigitSymbolModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    res = predict_number(image_path, model)
    if res == '':
        res = '-'
        print(f"{image_path}\n识别为空，自动补为'-'")

    return res

# ======================
# 6. 主程序（保持逻辑不变）
# ======================
def main():
    from get_pos import get_numimg
    print("获取预测用的图片...")
    get_numimg(1)

    MODEL_PATH = "../my_own_model.pth"
    IMAGE_PATH = r".\pro_img\c2.png"  # 你要识别的图片

    if os.path.exists(MODEL_PATH):
        print("✅ 加载训练好的模型...")
        model = DigitSymbolModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("🚀 开始训练优化后的模型...")
        model = train_my_model()

    res = predict_number(IMAGE_PATH, model)
    print("🎉 识别结果：", res)


if __name__ == "__main__":
    while True:
        input("按【回车】执行一次主函数，按 Ctrl+C 退出：")
        main()