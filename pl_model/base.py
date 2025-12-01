import torch
from pytorch_lightning.core import LightningModule

class BasePLModel(LightningModule):
    def __init__(self):
        super(BasePLModel, self).__init__()
        self.metric = {}
        self.num_class = 2

    # [수정 1] training_step에서 loss를 리턴할 때 로깅 설정을 하는 것이 권장됩니다.
    def training_step(self, batch, batch_idx):
        # ... 모델 연산 및 loss 계산 로직 ...
        # loss = ...
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss
        pass 

    # [수정 3] Lightning 2.0+ 에서는 training_epoch_end가 제거되었습니다.
    # 위 training_step에서 log(on_epoch=True)를 사용하면 이 메서드는 삭제해도 됩니다.
    # 만약 별도의 커스텀 로직이 필요하다면 on_train_epoch_end(self)를 사용하세요.
    def on_train_epoch_end(self):
        pass

    # [수정 4] validation_epoch_end(self, outputs) -> on_validation_epoch_end(self)
    def on_validation_epoch_end(self):
        return self.on_test_epoch_end()

    def measure(self, batch, output):
        ct, mask, name = batch
        
        # output shape: (B, C, H, W, D) 등으로 가정
        output = torch.softmax(output, dim=1)[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        
        # threshold value
        output = (output > 0.4).float()

        # record values concerned with dice score
        # [참고] 배치 내 각 샘플에 대해 루프를 돌며 metric 딕셔너리에 누적
        for ib in range(len(ct)):
            pre = torch.sum(output[ib], dim=(1, 2))
            gt = torch.sum(mask[ib], dim=(1, 2))
            inter = torch.sum(torch.mul(output[ib], mask[ib]), dim=(1, 2))
            
            # name[ib]가 텐서일 경우를 대비해 item()이나 문자열 변환이 필요할 수 있습니다.
            key = name[ib] 
            
            if key not in self.metric.keys():
                self.metric[key] = torch.stack((pre, gt, inter), dim=0)
            else:
                self.metric[key] += torch.stack((pre, gt, inter), dim=0)

    # [수정 5] test_epoch_end(self, outputs) -> on_test_epoch_end(self)
    # outputs 인자가 제거됨. self.metric은 이미 멤버 변수이므로 접근 가능.
    def on_test_epoch_end(self):
        # calculate dice score
        num_class = self.num_class - 1
        
        # 디바이스 호환성을 위해 self.device 사용 권장
        scores = torch.zeros((num_class, 3), device=self.device)
        nums = torch.zeros((num_class, 1), device=self.device)
        
        for k, v in self.metric.items():
            # v: (3, ...) 형태
            dice = (2. * v[2] + 1.0) / (v[0] + v[1] + 1.0)
            voe = (2. * (v[0] - v[2])) / (v[0] + v[1] + 1e-7)
            rvd = v[0] / (v[1] + 1e-7) - 1.

            for i in range(num_class):
                # gt가 0이 아닐 때만 계산 (안전장치)
                if v[1][i].item() != 0:
                    nums[i] += 1
                    scores[i][0] += dice[i]
                    scores[i][1] += voe[i]
                    scores[i][2] += rvd[i]

        # 평균 계산
        scores = scores / (nums + 1e-7) # 0으로 나누기 방지

        for i in range(num_class):
            self.log('dice_class{}'.format(i), scores[i][0])
            self.log('voe_class{}'.format(i), scores[i][1])
            self.log('rvd_class{}'.format(i), scores[i][2])
            
            # [수정 6] print 문을 루프 안으로 이동 (기존 코드에서는 i가 루프 밖에서 호출되어 오류 가능성 있음)
            print(f'dice_class{i}: {scores[i][0].item():.5f} , '
                  f'voe_class{i}: {scores[i][1].item():.5f} , '
                  f'rvd_class{i}: {scores[i][2].item():.5f}')

        # 다음 epoch를 위해 메트릭 초기화
        self.metric = {}