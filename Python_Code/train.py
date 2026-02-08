import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import LSS
from UnityDataset import UnityDataset
import os
import time

class DiceLoss_CrossEntropy(nn.Module):
    def __init__(self, c_weight=None, d_weight = [1, 1, 1, 1], num_classes = 4):
        super(DiceLoss_CrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.crossEntropy = nn.CrossEntropyLoss(weight=c_weight)
        self.d_weight = d_weight

    def forward(self, preds, targets):
        # pred : [B, 4, 32, 64, 64]
        # target : [B, 32, 64, 64]
        ce_loss = self.crossEntropy(preds, targets) 

        pred_probs = F.softmax(preds, dim=1) # 4개 있는걸 확률값으로 변경
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(preds.device) # [B, 32, 64, 64, 4]로 변함
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float() # [B, 4, 32, 64, 64]로 변함
        
        dims = (0, 2, 3, 4)
        intersection = torch.sum(pred_probs * targets_one_hot, dims).to(preds.device) # 4
        cardinality = torch.sum(pred_probs + targets_one_hot, dims).to(preds.device) # 4

        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6) # 모두를 따로 계산하는것, 한꺼번에 하는게 아니라

        dice_loss = torch.mean((1. - dice_score) * self.d_weight.to(preds.device))

        return 0.1 * ce_loss + 0.9 * dice_loss


def calculate_iou(outputs, labels, num_classes = 4):
    preds = torch.argmax(outputs, dim=1) # [B, 32, 64, 64]
    iou_list = []
    
    for c in range(num_classes):
        intersection = ((preds == c) & (labels == c)).sum().item()
        union = ((preds == c) | (labels == c)).sum().item()

        if union == 0: iou_list.append(float('nan'))
        else: iou_list.append(intersection / union)
    return iou_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = [r'./data_1', r'./data_2', r'./data_3', r'./data_4']
model_save_path = r'./model_save'
batch_size = 8 # 16으로 하니 메모리가 꽉차서 학습이 너무 느림
lr = 0.0001
epochs = 80
save_interval = 5
num_classes = 4
name_classes = ['Empty', 'Road', 'Car', 'Obstacle']

if not os.path.exists(model_save_path): os.makedirs(model_save_path)

full_dataset = UnityDataset(root_dir)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Train Loader (학습용: 데이터 섞음)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers= 0,
    pin_memory=True,
    drop_last=True
)

# Test Loader (평가용: 데이터 섞지 않음)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers= 0,
    pin_memory=True,
    drop_last=False  # 평가는 모든 데이터를 다 봐야 하므로 버리지 않음
)

model = LSS(device).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2) # 가중치 조금씩 감소함
c_weight = torch.tensor([1.0, 5.0, 2.0, 10.0]).to(device) # 장애물에 더 큰 가중치 부여 [빈공간, 도로, 자동차, 장애물] [1.0, 10.0, 2.0, 10.0] 
d_weight = torch.tensor([1.0, 5.0, 2.0, 10.0]).to(device) # 장애물에 더 큰 가중치 부여 [빈공간, 도로, 자동차, 장애물] [1.0, 10.0, 2.0, 10.0] 
# 장애물을 10으로 주니 훨씬 많이 있다 예측함
# 장애물을 5로 낮추어봄 
# 장애물 IOU가 계속 낮게 나와 두개의 손실함수에 장애물 가중치를 높힘
#criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device=device)

criterion = DiceLoss_CrossEntropy(c_weight=c_weight, d_weight=d_weight, num_classes=4).to(device=device) # 장애물의 IOU가 계속 적게 나와 DiceLoss + CrossEntropy로 변경
# 15에포크 마다 optimizer의 학습률에 0.1을 곱함
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
# 바꿈 테스트로스가 일정하면 0.5만큼 곱하는걸로
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def train():
    best_test_loss = float('inf')
    print("학습 시작")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['imgs'].to(device)
            rots = batch['rots'].to(device)
            trans = batch['trans'].to(device)
            intrins = batch['intrinsics'].to(device)
            labels = batch['label_3d'].to(device).long() # [B, 1, 32, 64, 64]
            labels = labels.squeeze(1).long() # [B, 32, 64, 64]

            optimizer.zero_grad()
            preds = model(imgs, rots, trans, intrins) # [B, 4, 32, 64, 64]

            loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                if torch.cuda.is_available(): torch.cuda.synchronize() # gpu연산 끝날때까지 대기
                now_time = time.time()
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Time: {now_time-batch_start_time:.5f}.sec")
                batch_start_time = time.time()


                iou_list = calculate_iou(preds, labels)
                for c in range(num_classes):
                    print(f'{name_classes[c]} IOU : {iou_list[c]:.4f}')

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['imgs'].to(device)
                rots = batch['rots'].to(device)
                trans = batch['trans'].to(device)
                intrins = batch['intrinsics'].to(device)
                labels = batch['label_3d'].to(device).long()
                labels = labels.squeeze(1)

                preds = model(imgs, rots, trans, intrins)
                loss = criterion(preds, labels)
                test_loss += loss.item()


        avg_test_loss = test_loss / len(test_loader)

        # scheduler.step(avg_test_loss)
        scheduler.step()

        print(f"\n==> Epoch {epoch} 완성! Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if epoch % save_interval == 0:
            save_path = os.path.join(model_save_path, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")

        # 2. 최상 성능 모델 저장 (Best Model)
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_path = os.path.join(model_save_path, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New Best Model Saved! Loss: {best_test_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_time = time.time()
        print(f"Epoch 소요시간 : {epoch_end_time - epoch_start_time:.2f}.sec 현재 학습률: {current_lr:.8f}")
        print("-" * 50)

if __name__ == "__main__":
    train()

'''
에포크를 50까지 돌렸지만 아직도 장애물 IOU가 낮다.
먼저 카메라가 보지 못하는 부분, 예를 들어 천장 같은 부분은 당연히 예측이 안된다.
그리고 나무나 가로등같은 작은 것들을 인지하지 못한다.
'''