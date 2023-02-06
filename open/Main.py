# Import (Baseline)


# 랜덤 임포트고
import random
import pandas as pd
import numpy as np
# 파일, 폴더 디렉토리 접근에 필요한 os
import os
# cv2 = opencv 임 이미지, 영상처리할때 필요함
import cv2

# 메인 네임스페이스. 텐서 등의 다양한 수학 함수가 포함되어져 있으며 Numpy와 유사한 구조를 가짐.
import torch

# 신경망을 구축하기 위한 다양한 데이터 구조나 레이어 등이 정의되어져 있습니다. 
# 예를 들어 RNN, LSTM과 같은 레이어, ReLU와 같은 활성화 함수, MSELoss와 같은 손실 함수들이 있습니다.
import torch.nn as nn

# 확률적 경사 하강법(Stochastic Gradient Descent, SGD)를 중심으로 한 파라미터 최적화 알고리즘이 구현되어져 있습니다.
import torch.optim as optim
# torch.nn 에 없는 함수들인듯 ?
import torch.nn.functional as F

# SGD의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수가 포함되어져 있습니다.
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 



# 파이토치에서 그거하는듯 gpu 먼저 사용하게 하고 아닐경우 cpu 사용하게하는 cuda가 gpu 임
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





# Hyperparameter Setting 

CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':4,
    'SEED':41
}




# Fixed RandomSeed

# seed 란 ? 
# random 에서 동일한 순서로 난수를 생성하게 핢

def seed_everything(seed):
    random.seed(seed)
    # print(random.seed(seed))
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    # print(os.environ['PYTHONHASHSEED'])
    
    np.random.seed(seed)
    # print(np.random.seed(seed))
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# print(CFG['SEED'])
# 41

seed_everything(CFG['SEED']) # Seed 고정







# Data Load

df = pd.read_csv('./train.csv')

# print(df.head())
#     sample_id              video_path  label
# 0  TRAIN_0000  ./train/TRAIN_0000.mp4      7
# 2  TRAIN_0002  ./train/TRAIN_0002.mp4      0
# 3  TRAIN_0003  ./train/TRAIN_0003.mp4      0
# 4  TRAIN_0004  ./train/TRAIN_0004.mp4      1






# Train / Validation Split

train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
print(train)
print(val)





# # CustomDataset

# class CustomDataset(Dataset):
#     def __init__(self, video_path_list, label_list):
#         self.video_path_list = video_path_list
#         self.label_list = label_list
        
#     def __getitem__(self, index):
#         frames = self.get_video(self.video_path_list[index])
        
#         if self.label_list is not None:
#             label = self.label_list[index]
#             return frames, label
#         else:
#             return frames
        
#     def __len__(self):
#         return len(self.video_path_list)
    
#     def get_video(self, path):
#         frames = []
#         cap = cv2.VideoCapture(path)
#         for _ in range(CFG['VIDEO_LENGTH']):
#             _, img = cap.read()
#             img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
#             img = img / 255.
#             frames.append(img)
#         return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

# train_dataset = CustomDataset(train['video_path'].values, train['label'].values)
# train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# val_dataset = CustomDataset(val['video_path'].values, val['label'].values)
# val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)




# # Model Define

# class BaseModel(nn.Module):
#     def __init__(self, num_classes=13):
#         super(BaseModel, self).__init__()
#         self.feature_extract = nn.Sequential(
#             nn.Conv3d(3, 8, (1, 3, 3)),
#             nn.ReLU(),
#             nn.BatchNorm3d(8),
#             nn.MaxPool3d(2),
#             nn.Conv3d(8, 32, (1, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, (1, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, (1, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d((3, 7, 7)),
#         )
#         self.classifier = nn.Linear(1024, num_classes)
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.feature_extract(x)
#         x = x.view(batch_size, -1)
#         x = self.classifier(x)
#         return x
    
    
    
    
# # Train

# def train(model, optimizer, train_loader, val_loader, scheduler, device):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
    
#     best_val_score = 0
#     best_model = None
    
#     for epoch in range(1, CFG['EPOCHS']+1):
#         model.train()
#         train_loss = []
#         for videos, labels in tqdm(iter(train_loader)):
#             videos = videos.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             output = model(videos)
#             loss = criterion(output, labels)
            
#             loss.backward()
#             optimizer.step()
            
#             train_loss.append(loss.item())
                    
#         _val_loss, _val_score = validation(model, criterion, val_loader, device)
#         _train_loss = np.mean(train_loss)
#         print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
#         if scheduler is not None:
#             scheduler.step(_val_score)
            
#         if best_val_score < _val_score:
#             best_val_score = _val_score
#             best_model = model
    
#     return best_model

# def validation(model, criterion, val_loader, device):
#     model.eval()
#     val_loss = []
#     preds, trues = [], []
    
#     with torch.no_grad():
#         for videos, labels in tqdm(iter(val_loader)):
#             videos = videos.to(device)
#             labels = labels.to(device)
            
#             logit = model(videos)
            
#             loss = criterion(logit, labels)
            
#             val_loss.append(loss.item())
            
#             preds += logit.argmax(1).detach().cpu().numpy().tolist()
#             trues += labels.detach().cpu().numpy().tolist()
        
#         _val_loss = np.mean(val_loss)
    
#     _val_score = f1_score(trues, preds, average='macro')
#     return _val_loss, _val_score




# # Run!!

# model = BaseModel()
# model.eval()
# optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

# infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)