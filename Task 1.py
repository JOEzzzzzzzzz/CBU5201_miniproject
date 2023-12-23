import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化MTCNN
mtcnn = MTCNN(keep_all=False, device=device, image_size=128, margin=30)
original_img_dir = 'genki4k/files'  # 原始图像文件夹路径
processed_img_dir = 'genki4k/processed_files'  # 处理后图像保存路径

# 创建处理后的图像文件夹
if not os.path.exists(processed_img_dir):
    os.makedirs(processed_img_dir)


processed_images = []
# 遍历并处理每张图像
for img_file in tqdm(os.listdir(original_img_dir), desc="Processing Images"):
    img_path = os.path.join(original_img_dir, img_file)
    img = Image.open(img_path).convert('RGB')
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_cropped_pil = to_pil_image(img_cropped)  # 将Tensor转换为PIL Image
        processed_img_path = os.path.join(processed_img_dir, img_file)
        img_cropped_pil.save(processed_img_path)  # 保存PIL Image
        processed_images.append(img_file)


# 数据集类
class MyDataset_T1(Dataset):
    def __init__(self, processed_img_dir, label_file, transform=None):
        self.img_dir = processed_img_dir
        self.transform = transform
        self.labels = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                img_name = f"file{int(line[0])+1:04d}.jpg"
                if img_name in processed_images:
                    self.labels.append(line)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = f'file{idx+1:04d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            return None

        image = Image.open(img_path)  # 直接加载处理后的图像

        if self.transform:
            image = self.transform(image)

        smile_label, yaw, pitch, roll = self.labels[idx]
        return image, torch.tensor(int(smile_label), dtype=torch.float32)

    

def collate_fn(batch):
    # 过滤掉整个为 None 的样本
    batch = [b for b in batch if b is not None]
    # 如果过滤后的批次为空，则返回两个空的张量
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    return default_collate(batch)
 


# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据集
dataset_T1 = MyDataset_T1(processed_img_dir='genki4k/processed_files', 
                    label_file='genki4k/labels.txt', 
                    transform=transform)

# 将数据集分为训练集、验证集和测试集
train_size = int(0.7 * len(dataset_T1))
valid_size = int(0.15 * len(dataset_T1))
test_size = len(dataset_T1) - train_size - valid_size

train_dataset_T1, valid_dataset_T1, test_dataset_T1 = random_split(dataset_T1, [train_size, valid_size, test_size])

# 数据加载器
batch_size = 256  # 根据您的硬件限制可以调整
train_loader_T1 = DataLoader(train_dataset_T1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader_T1 = DataLoader(valid_dataset_T1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader_T1 = DataLoader(test_dataset_T1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 加载ResNet-18模型并替换最后的全连接层
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#model = models.resnet18(weights=None)
pre_fc_in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(pre_fc_in_features, 512),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1)
)
model = model.to(device)  # 移动模型到 CUDA 设备



# 定义损失函数和优化器
criterion_T1 = torch.nn.BCEWithLogitsLoss()
optimizer_T1 = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_T1, step_size=5, gamma=0.5)


train_losses_T1 = []
valid_losses_T1 = []
accuracies = []


# 训练过程
num_epochs = 20  # 你可能需要根据任务的不同进行调整
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    progress_bar = tqdm(train_loader_T1, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
    for batch in progress_bar:
        if batch is None or batch[0] is None:
            continue  # 跳过这个批次
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_T1.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion_T1(outputs, labels)
        loss.backward()
        optimizer_T1.step()
        train_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': train_loss / (progress_bar.n + 1)})
    train_loss /= len(train_loader_T1)
    train_losses_T1.append(train_loss)
    scheduler.step()



# 验证模型
    model.eval()  # 设置模型为评估模式
    valid_loss = 0.0
    correct_smiles = 0
    total_smiles = 0
    with tqdm(valid_loader_T1, desc='Validating') as progress_bar:
        for batch in progress_bar:
            if batch is None or batch[0] is None:
                continue  # 跳过这个批次
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs).squeeze(1)
                loss = criterion_T1(outputs, labels)
                valid_loss += loss.item()
                predicted_smiles = torch.sigmoid(outputs) > 0.5
                correct_smiles += (predicted_smiles == labels.bool()).sum().item()
                total_smiles += labels.size(0)
                progress_bar.set_postfix({'Validation Loss': valid_loss / (progress_bar.n + 1)})
    valid_loss /= len(valid_loader_T1)
    accuracy = correct_smiles / total_smiles
    valid_losses_T1.append(valid_loss)
    accuracies.append(accuracy)

    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}')

# 保存模型参数
torch.save(model.state_dict(), 'model_smile.pth')


# 测试模型
def test_model(model, test_loader_T1, device):
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    correct_smiles = 0
    total_smiles = 0
    with tqdm(test_loader_T1, desc='Testing') as progress_bar:
        for batch in progress_bar:
            if batch is None or batch[0] is None:
                continue  # 跳过这个批次
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs).squeeze(1)
                loss = criterion_T1(outputs, labels)
                test_loss += loss.item()
                predicted_smiles = torch.sigmoid(outputs) > 0.5
                correct_smiles += (predicted_smiles == labels.bool()).sum().item()
                total_smiles += labels.size(0)
                progress_bar.set_postfix({'Test Loss': test_loss / (progress_bar.n + 1)})
    test_loss /= len(test_loader_T1)
    accuracy = correct_smiles / total_smiles
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}')


# 调用测试函数
test_model(model, test_loader_T1, device)


# 绘制训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses_T1, label='Training Loss')
plt.plot(valid_losses_T1, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_T1.png')  # 保存损失图
plt.show()

# 绘制准确率
plt.figure(figsize=(10, 5))
plt.plot(accuracies, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot_T1.png')  # 保存准确率图
plt.show()
