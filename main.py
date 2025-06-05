import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决 OpenMP 冲突

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import numpy as np

# 创建保存结果的文件夹
save_dir = f"training_results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
print(f"训练结果将保存至: {save_dir}")

# 数据增强与归一化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("加载数据集...")
# 加载数据集
train_data = datasets.ImageFolder('plantvillage_processed/train', transform=train_transform)
test_data = datasets.ImageFolder('plantvillage_processed/test', transform=test_transform)

print(f"训练集类别数: {len(train_data.classes)}")
print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")

# 类别平衡采样
print("计算类别权重...")
class_counts = [len(os.listdir(os.path.join("plantvillage_processed/train", cls))) for cls in train_data.classes]
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
samples_weights = weights[[train_data.targets[i] for i in range(len(train_data.targets))]]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_data, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=16)

print("初始化模型...")
"""# 模型定义
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))

# 冻结特征层
for param in model.features.parameters():
    param.requires_grad = False
"""
##################################################
model = models.resnet34(pretrained=True)

# 冻结所有参数（如需保留预训练权重）
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层全连接层
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),   # 添加一个隐藏层
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(train_data.classes))
)

# 解冻最后一层参数（如需微调）
for param in model.fc.parameters():
    param.requires_grad = True
#####################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = model.to(device)

# 优化器与损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 记录训练指标的列表
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 创建CSV日志文件
with open(f'{save_dir}/training_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

best_test_acc = 0.0
print("开始训练...")
for epoch in range(10):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        # 每10个batch打印一次进度
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # 计算训练指标
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total

    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    # 计算测试指标
    epoch_test_loss = test_loss / test_total
    epoch_test_acc = test_correct / test_total

    # 记录指标
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)

    # 写入CSV
    with open(f'{save_dir}/training_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc])

    # 实时绘制并保存图表
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, 'o-', label='Train Loss')
    plt.plot(range(1, epoch + 2), test_losses, 'o-', label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, 'o-', label='Train Acc')
    plt.plot(range(1, epoch + 2), test_accuracies, 'o-', label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 确保y轴范围在0-1之间
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch + 1}.png')  # 保存当前epoch图表
    plt.close()

    # 打印进度
    print(f"\nEpoch {epoch + 1}/10 结果:")
    print(f"训练损失: {epoch_train_loss:.4f} | 准确率: {epoch_train_acc:.4f}")
    print(f"测试损失: {epoch_test_loss:.4f} | 准确率: {epoch_test_acc:.4f}")

    # 保存最佳模型
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': train_data.classes,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'epoch': epoch + 1,
            'test_acc': best_test_acc
        }, f'{save_dir}/best_model.pth')
        print(f"保存最佳模型，测试准确率: {best_test_acc:.4f}")

# 训练结束后保存最终图表
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), train_losses, 'o-', label='Train Loss')
plt.plot(range(1, 11), test_losses, 'o-', label='Test Loss')
plt.title('最终损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), train_accuracies, 'o-', label='Train Acc')
plt.plot(range(1, 11), test_accuracies, 'o-', label='Test Acc')
plt.title('最终准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{save_dir}/final_result.png')
plt.close()

# 保存训练摘要
with open(f'{save_dir}/summary.txt', 'w') as f:
    f.write(f"训练摘要\n{'=' * 30}\n")
    f.write(f"数据集: plantvillage_processed\n")
    f.write(f"训练样本数: {len(train_data)}\n")
    f.write(f"测试样本数: {len(test_data)}\n")
    f.write(f"类别数: {len(train_data.classes)}\n")
    f.write(f"最佳测试准确率: {best_test_acc:.4f}\n\n")

    f.write("训练配置\n")
    f.write(f"模型: MobileNetV2\n")
    f.write(f"优化器: Adam (lr=0.0001)\n")
    f.write(f"批次大小: 16\n")
    f.write(f"训练轮数: 10\n\n")

    f.write("训练结果\n")
    for epoch in range(10):
        f.write(f"Epoch {epoch + 1}: ")
        f.write(f"Train Loss={train_losses[epoch]:.4f}, Acc={train_accuracies[epoch]:.4f} | ")
        f.write(f"Test Loss={test_losses[epoch]:.4f}, Acc={test_accuracies[epoch]:.4f}\n")

print(f"\n训练完成! 所有结果已保存至: {save_dir}")
print(f"最佳测试准确率: {best_test_acc:.4f}")