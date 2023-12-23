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
mtcnn = MTCNN(keep_all=False, device=device, image_size=128, margin= 30)
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
class MyDataset_T2(Dataset):
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

        image = Image.open(img_path)  

        if self.transform:
            image = self.transform(image)

        smile_label, yaw, pitch, roll = self.labels[idx]
        pose_label = torch.tensor([float(yaw)*10, float(pitch)*10, float(roll)*10], dtype=torch.float32)
        return image, pose_label


    

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
dataset_T2 = MyDataset_T2(processed_img_dir='genki4k/processed_files', 
                    label_file='genki4k/labels.txt', 
                    transform=transform)

# 将数据集分为训练集、验证集和测试集
train_size = int(0.7 * len(dataset_T2))
valid_size = int(0.15 * len(dataset_T2))
test_size = len(dataset_T2) - train_size - valid_size

train_dataset_T2, valid_dataset_T2, test_dataset_T2 = random_split(dataset_T2, [train_size, valid_size, test_size])

# 数据加载器
batch_size = 128  # 根据您的硬件限制可以调整
train_loader_T2 = DataLoader(train_dataset_T2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader_T2 = DataLoader(valid_dataset_T2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader_T2 = DataLoader(test_dataset_T2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 加载预训练的ResNet-18模型
model2 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
pre_fc_in_features2 = model2.fc.in_features
# 替换最后的全连接层以输出3个特征
model2.fc = nn.Sequential(
    nn.Linear(pre_fc_in_features2, 512),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(512, 3)  # 输出yaw, pitch, roll
)
model2 = model2.to(device)

# 定义损失函数和优化器
criterion_T2 = torch.nn.MSELoss()  # MSE损失函数
optimizer_T2 = torch.optim.Adam(model2.parameters(), lr=0.001,  weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_T2, step_size=10, gamma=0.5)

train_losses_T2 = []
valid_losses_T2 = []

# 训练过程
num_epochs = 50  # 根据任务的不同可能需要调整
for epoch in range(num_epochs):
    model2.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader_T2, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
    for batch in progress_bar:
        if batch is None or batch[0] is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_T2.zero_grad()
        outputs = model2(imgs)
        loss = criterion_T2(outputs, labels)
        loss.backward()
        optimizer_T2.step()
        train_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': train_loss / (progress_bar.n + 1)})
    train_loss /= len(train_loader_T2)
    train_losses_T2.append(train_loss)
    # scheduler.step()

    # 验证模型
    model2.eval()
    valid_loss = 0.0
    with tqdm(valid_loader_T2, desc='Validating') as progress_bar:
        for batch in progress_bar:
            if batch is None or batch[0] is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model2(imgs)
                loss = criterion_T2(outputs, labels)
                valid_loss += loss.item()
            progress_bar.set_postfix({'Validation Loss': valid_loss / (progress_bar.n + 1)})
        valid_loss /= len(valid_loader_T2)
        valid_losses_T2.append(valid_loss)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')


# 保存模型参数
torch.save(model2.state_dict(), 'model_dir.pth')


# 测试模型并计算MSE、MAE和RMSE
def test_model_with_metrics(model, test_loader_T2, device, criterion_T2):
    model.eval()
    test_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    with tqdm(test_loader_T2, desc='Testing') as progress_bar:
        for batch in progress_bar:
            if batch is None or batch[0] is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)

                # 计算损失
                loss = criterion_T2(outputs, labels)
                test_loss += loss.item()  # 累加测试损失

                # 缩放模型输出回原始尺度
                outputs_rescaled = outputs / 10
                labels_rescaled = labels / 10

                # 计算均方误差
                mse = torch.mean((outputs_rescaled - labels_rescaled) ** 2, dim=1)
                total_mse += mse.sum().item()

                # 计算平均绝对误差
                mae = torch.mean(torch.abs(outputs_rescaled - labels_rescaled), dim=1)
                total_mae += mae.sum().item()

                # 更新样本总数
                total_samples += labels.size(0)

            progress_bar.set_postfix({'Test Loss': test_loss / (progress_bar.n + 1)})
    test_loss /= len(test_loader_T2)
    average_mse = total_mse / total_samples
    average_mae = total_mae / total_samples
    rmse = torch.sqrt(torch.tensor(average_mse))
    print(f'Test Loss: {test_loss:.8f}, MSE: {average_mse:.6f}, MAE: {average_mae:.6f}, RMSE: {rmse:.6f}')


# 调用测试函数
test_model_with_metrics(model2, test_loader_T2, device, criterion_T2)


# 绘制训练和验证损失
plt.figure(figsize=(12, 6))
plt.plot(train_losses_T2, label='Training Loss')
plt.plot(valid_losses_T2, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_T2.png')
plt.show()



