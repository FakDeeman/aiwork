import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import json

# 加载模型和配置
checkpoint = torch.load('training_results/20250528-174001/best_model.pth')
class_names = checkpoint['class_names']

# 定义模型结构（必须与训练时一致）
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)  # 修改输出层
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 定义预处理（无需导入训练文件）
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint['normalize_mean'],
        std=checkpoint['normalize_std']
    )
])

# 预测函数
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    pred_idx = torch.argmax(output).item()
    return class_names[pred_idx]

# 使用示例
print(predict('plantvillage_raw/color/Raspberry___healthy/0df51eb7-e701-492b-bb9f-994d30ea16c7___Mary_HL 6301.JPG'))