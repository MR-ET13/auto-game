import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from get_pos import get_numimg

# ======================
# 1. 你的类别：负号 + 0~9
# ======================
CLASSES = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ======================
# 2. 模型结构
# ======================
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

# ======================
# 3. 加载【你自己的数据集】🔥 核心
# ======================
class MyOwnDataset(Dataset):
    def __init__(self, root_dir="my_dataset"):
        self.root = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.files = []
        for cls in CLASSES:
            folder = os.path.join(root_dir, cls)
            for fname in os.listdir(folder):
                self.files.append( (os.path.join(folder, fname), cls) )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, cls = self.files[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)
        label = CLASS_TO_IDX[cls]
        return img, torch.tensor(label, dtype=torch.long)

# ======================
# 4. 训练函数
# ======================
def train_my_model():
    dataset = MyOwnDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = DigitSymbolModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.3f}")

    torch.save(model.state_dict(), "my_own_model.pth")
    print("✅ 训练完成！模型已保存")
    return model

# ======================
# 5. 预测函数（多位数/负数）
# ======================
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

# ======================
# 6. 主程序：自动训练/预测
# ======================
if __name__ == "__main__":
    import numpy as np

    print("获取预测用的图片...")
    get_numimg(1)

    MODEL_PATH = "my_own_model.pth"
    IMAGE_PATH = r".\pro_img\c2.png"  # 你要识别的图片

    if os.path.exists(MODEL_PATH):
        print("✅ 加载你自己训练的模型...")
        model = DigitSymbolModel()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    else:
        print("🚀 开始训练【你自己制作的数据集】")
        model = train_my_model()

    res = predict_number(IMAGE_PATH, model)
    print("🎉 识别结果：", res)
