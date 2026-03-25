import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os

# --------------------------
# 1. 字符集定义
# --------------------------
CLASSES = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# --------------------------
# 2. 模型结构
# --------------------------
class DigitSymbolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------
# 3. 数据集（训练用）
# --------------------------
class SyntheticDataset(Dataset):
    def __init__(self, size=50000):
        self.size = size
        self.mnist = MNIST(root='', train=True, download=True)
        self.transform = transforms.Compose([
            transforms.Grayscale(), transforms.Resize((28,28)),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        r = random.random()
        if r < 0.1:
            img = Image.new('L', (28,28), 0)
            d = ImageDraw.Draw(img)
            d.text((6,6), '-', fill=255, font=ImageFont.load_default(size=20))
            label = CLASS_TO_IDX['-']
        else:
            img, label = self.mnist[idx % len(self.mnist)]
        return self.transform(img), torch.tensor(label, dtype=torch.long)

# --------------------------
# 4. 训练函数
# --------------------------
def train_model():
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = DigitSymbolModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "multi_digit_model.pth")
    print("✅ 模型训练完成并保存")
    return model

# --------------------------
# 5. 预测函数
# --------------------------
def predict_number(image_path, model):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.Resize((28,28)),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    img_bin = (img_np > 127).astype(np.uint8) * 255

    h_proj = np.sum(img_bin, axis=0)
    regions = []
    start = None
    for i, val in enumerate(h_proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            regions.append((start, i))
            start = None

    chars = []
    with torch.no_grad():
        for l, r in regions:
            char_img = img.crop((l, 0, r, img.height))
            tensor = transform(char_img).unsqueeze(0)
            out = model(tensor)
            idx = torch.argmax(out).item()
            chars.append(IDX_TO_CLASS[idx])

    return ''.join(chars)

# --------------------------
# 6. 主程序：自动判断！第二次直接预测
# --------------------------
if __name__ == "__main__":
    MODEL_PATH = "multi_digit_model.pth"
    # 你的图片路径（每次改这里就行）
    IMAGE_PATH = "tselft.png"

    # ======================================
    # 核心：如果模型已存在，直接加载，不训练！
    # ======================================
    if os.path.exists(MODEL_PATH):
        print("🔍 检测到已训练模型，直接加载...")
        model = DigitSymbolModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        print("🚀 未找到模型，开始训练...")
        model = train_model()

    # 开始预测
    result = predict_number(IMAGE_PATH, model)
    print("\n🎉 识别结果：", result)