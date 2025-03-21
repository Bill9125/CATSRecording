import torch.nn as nn
import torch
import argparse
import os, glob, json

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)  # 短路連接
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 殘差連接
        out = torch.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ResNet32, self).__init__()
        self.initial = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # ResNet-32 主要包含 5 個 Block（共 32 層）
        self.layer1 = self.make_layer(64, 64, 5)
        self.layer2 = self.make_layer(64, 128, 5, downsample=True)
        self.layer3 = self.make_layer(128, 256, 5, downsample=True)
        self.layer4 = self.make_layer(256, 512, 5, downsample=True)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 壓縮時間維度
        self.fc = nn.Linear(512, num_classes)  # 最終分類

    def make_layer(self, in_channels, out_channels, num_blocks, downsample=False):
        layers = [ResidualBlock(in_channels, out_channels, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # 去掉最後的 1 維
        x = self.fc(x)  # 最終分類
        return x

def merge_data(folder):
    feartures = []
    delta_path = os.path.join(folder, 'filtered_delta_norm')
    delta2_path = os.path.join(folder, 'filtered_delta2_norm')
    square_path = os.path.join(folder, 'filtered_delta_square_norm')
    zscore_path = os.path.join(folder, 'filtered_zscore_norm')
    orin_path = os.path.join(folder, 'filtered_norm')
    
    
    if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
        print(f"Missing data in {folder}")
        return
    
    deltas = glob.glob(os.path.join(delta_path, '*.txt'))
    delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
    squares = glob.glob(os.path.join(square_path, '*.txt'))
    zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
    orins = glob.glob(os.path.join(orin_path, '*.txt'))
    
    data_per_ind = list(fetch(zip(deltas, delta2s, zscores, squares, orins)))  # Ensure list output
    feartures.extend(data_per_ind)
    return feartures

def fetch(uds):
    data_per_ind = []
    for ud in uds:
        parsed_data = []
        for file in ud:
            with open(file, 'r') as f:
                lines = f.read().strip().split('\n')
                parsed_data.append([list(map(float, line.split(','))) for line in lines])
        
        for num in zip(*parsed_data):
            data_per_ind.append([item for sublist in num for item in sublist])
            if len(data_per_ind) == 110:
                yield data_per_ind
                data_per_ind = []

def predict(model, fearture):
    with torch.no_grad():
        output = model(fearture)  # 獲取模型輸出
        predicted_class = torch.argmax(output, dim=1)  # 取得最大信心值的類別
        confidence_scores = torch.softmax(output, dim=1)  # 計算分類信心值
    return predicted_class, confidence_scores

def save_to_config(y_data, output_file):
    
    config_data = {
        "": y_data
    }

    # 生成 JSON 配置文件路径
    config_path = output_file

    # 保存 JSON 文件
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('dir',type=str)
parser.add_argument('--out',type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
output_file = os.path.join(out, 'Deadlift_data', 'Score.json')

data = {}
category = {'1': 'Nice lift', '2': 'Wrong feet position', '3': 'Butt fly', '4': 'Skip Knee', '5': 'Hunchback'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feartures = merge_data(dir)
for num, name in category.items():
    model = ResNet32(input_dim = 25)
    model.load_state_dict(torch.load(f"../../model/deadlift/Pscore_model/{str(num)}/ResNet32_Model.pth"), map_location=device)
    model.to(device)
    model.eval()
    for i, fearture in enumerate(feartures):
        pred, conf = predict(model, fearture)
        if pred == 1:
            print(f'Set {i} is {name}')


