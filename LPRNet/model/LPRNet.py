import torch.nn as nn
import torch
import torch.nn.functional as F

class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SmallBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        # [NEW] STN (LocNet): 이미지를 바르게 펴주는 네트워크
        self.locnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        # 128x32 입력 기준 LocNet의 FC 레이어 계산
        # MaxPool 2번 거치면 128->64->32, 32->16->8 이 됨.
        # 따라서 32채널 * 8(H) * 32(W) 크기가 됨
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 8 * 32, 32), 
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # STN 초기값 설정 (항등 행렬)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Backbone (특징 추출)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(ch_in=64, ch_out=128),    
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(ch_in=64, ch_out=256),   
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(), 
            SmallBasicBlock(ch_in=256, ch_out=256),   
            nn.BatchNorm2d(num_features=256),   
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1), 
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), 
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    # LocNet의 FC 레이어 입력 크기를 동적으로 계산하는 함수
    def stn(self, x):
        xs = self.locnet(x)
        # flatten
        xs = xs.view(xs.size(0), -1)
        
        # 만약 이미지 크기가 바뀌어서 FC layer 입력 크기가 안 맞으면 재설정 (안전장치)
        if self.fc_loc[0].in_features != xs.size(1):
            device = xs.device
            self.fc_loc = nn.Sequential(
                nn.Linear(xs.size(1), 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).to(device)
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(device))
            
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        # 입력 이미지를 먼저 STN으로 펴줌
        x = self.stn(x)

        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: 
                keep_features.append(x)

        global_context = list()
        f_pow = keep_features[-1]
        
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            
            if f.shape[2:] != f_pow.shape[2:]:
                f = F.interpolate(f, size=f_pow.shape[2:], mode='bilinear', align_corners=True)
            
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

def build_lprnet(lpr_max_len=8, phase=True, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
    return Net