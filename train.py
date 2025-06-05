import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime

# 设置matplotlib后端
plt.switch_backend('Agg')

# 配置参数
DATA_ROOT = 'plantvillage_processed'
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001

# 创建保存目录
save_dir = f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_data = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(DATA_ROOT, 'test'), transform=test_transform)

# 类别平衡采样
class_counts = [len(os.listdir(os.path.join(DATA_ROOT, 'train', cls))) for cls in train_data.classes]
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
samples_weights = weights[train_data.targets]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# 初始化模型
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 冻结特征层
for param in model.features.parameters():
    param.requires_grad = False

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_acc = 0.0
metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


# 可视化函数
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean


def save_aug_samples():
    loader = DataLoader(train_data, batch_size=6, shuffle=True)
    images, labels = next(iter(loader))

    plt.figure(figsize=(12, 6))
    for i in range(6):
        img = denormalize(images[i]).numpy().transpose(1, 2, 0)
        plt.subplot(2, 3, i + 1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(train_data.classes[labels[i]])
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'visualizations', 'aug_samples.jpg'), bbox_inches='tight')
    plt.close()


save_aug_samples()

# 训练循环
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_data)
    train_acc = correct / total

    # 验证阶段
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(test_data)
    val_acc = val_correct / val_total

    # 记录指标
    metrics['epoch'].append(epoch + 1)
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_acc)
    metrics['val_loss'].append(val_loss)
    metrics['val_acc'].append(val_acc)

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'state_dict': model.state_dict(),
            'class_names': train_data.classes,
            'normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'epoch': epoch + 1,
            'accuracy': best_acc
        }, os.path.join(save_dir, 'models', 'best_model.pth'))

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

# 保存训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(metrics['train_loss'], label='Train')
plt.plot(metrics['val_loss'], label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(metrics['train_acc'], label='Train')
plt.plot(metrics['val_acc'], label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'visualizations', 'training_curves.jpg'), dpi=300)
plt.close()

# 生成混淆矩阵
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(save_dir, 'visualizations', 'confusion_matrix.jpg'), dpi=300, bbox_inches='tight')
plt.close()

# 保存指标到CSV
pd.DataFrame(metrics).to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)

print(f'训练完成，所有结果保存在: {save_dir}')