import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime


def main():
    # 配置参数
    DATA_ROOT = 'plantvillage_processed'
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3
    NUM_WORKERS = 4 if os.name != 'nt' else 0  # Windows设为0
    IMG_SIZE = 224

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 数据集加载
    train_data = datasets.ImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(DATA_ROOT, 'test'), transform=test_transform)

    # 类别平衡采样
    class_counts = [len(os.listdir(os.path.join(DATA_ROOT, 'train', cls))) for cls in train_data.classes]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = weights[train_data.targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # 数据加载器
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 模型定义（使用最新API）
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 更新权重参数
    num_features = model.fc.in_features

    # 替换分类层
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(train_data.classes))
    )
    model = model.to(device)

    # 解冻最后两层
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 验证梯度设置
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params}")

    # 优化器配置
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    # AMP设置（新API）
    scaler = GradScaler()  # 显式指定设备类型
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 使用新AMP API
            with autocast():  # 明确设备类型
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_acc = correct_train / total_train
        metrics['train_loss'].append(train_loss / len(train_loader))
        metrics['train_acc'].append(train_acc)

        # 记录学习率
        metrics['lr'].append(optimizer.param_groups[0]['lr'])

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted_val = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_acc = correct_val / total_val
        metrics['val_loss'].append(val_loss / len(test_loader))
        metrics['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'accuracy': best_acc,
                'classes': train_data.classes
            }, os.path.join(save_dir, 'models', 'best_model.pth'))

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train Loss: {metrics['train_loss'][-1]:.4f} | Val Loss: {metrics['val_loss'][-1]:.4f} | Train Acc: {metrics['train_acc'][-1]:.4f} | Val Acc: {val_acc:.4f}")

    # 可视化训练曲线
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'visualizations', 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 学习率变化曲线
    plt.figure()
    plt.plot(metrics['lr'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(save_dir, 'visualizations', 'lr_schedule.png'), dpi=300)
    plt.close()

    # 混淆矩阵
    model.load_state_dict(torch.load(os.path.join(save_dir, 'models', 'best_model.pth'))['model_state'])
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_data.classes,
                yticklabels=train_data.classes)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'visualizations', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存指标
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)

    print(f"训练完成！所有结果保存在：{save_dir}")


if __name__ == '__main__':
    # Windows系统专用设置
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
        torch.multiprocessing.set_start_method('spawn', force=True)

    main()



