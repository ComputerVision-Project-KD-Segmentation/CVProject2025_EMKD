import torch
import torch.nn.functional as F

from models import get_model
from pl_model.base import BasePLModel
from datasets.dataset import SliceDataset

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

class Dinov3SegmentationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(Dinov3SegmentationPLModel, self).__init__()
        self.save_hyperparameters(params)        
        self.net = get_model(
            self.hparams.model, 
            channels=2, 
            checkpoint_path=self.hparams.vit_checkpoint_path
        )
        self.initial_lr = self.hparams.lr
        self.vit_lr = 3e-5
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.num_epochs = self.hparams.epochs
        self.warmup_epochs = int(self.num_epochs * 0.1)
        self.train_indices = train_indices
        self.val_indices = val_indices

    def forward(self, x):
        # 모델이 (output, aux1, aux2) 형태의 튜플을 반환한다고 가정
        # output = self.net(x)
        output, _, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)  # Dice_loss Used

        # [수정 1] Loss 로깅 추가 (Lightning 2.x 권장)
        # on_epoch=True: epoch 단위 평균 자동 계산
        # prog_bar=True: 진행바에 표시
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # [수정 2] 딕셔너리가 아닌 loss 텐서 반환
        return loss

    def validation_step(self, batch, batch_idx):
        # BasePLModel의 measure 메서드를 활용하기 위해 test_step 호출
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)

        # BasePLModel에 정의된 measure 메서드로 Dice Score 등 계산
        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.train_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
        # collate_fn이 필요한 경우 주석 해제하여 사용
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True, 
            shuffle=True,
            # collate_fn=self.pad_collate_fn, 
        )

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.val_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=False
        )
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            # collate_fn=self.pad_collate_fn,
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        # Split parameters into two groups: vit and the rest.
        vit_params = []
        other_params = []
        for name, param in self.net.named_parameters():
            if 'dino_encoder' in name:
                vit_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.initial_lr, 'weight_decay': self.weight_decay},
            {'params': vit_params, 'lr': self.vit_lr, 'weight_decay': self.vit_weight_decay}
        ], betas = (0.9, 0.98))

        def lr_lambda(current_epoch):
            # Linear warmup phase.
            if current_epoch < self.warmup_epochs:
                # 0.0 ~ 1.0 까지 선형적으로 증가 (혹은 아주 작은 값부터 시작하고 싶다면 조정 가능)
                # 이렇게 하면 각 그룹은 (0 ~ 1.0) * (자기 자신의 lr) 로 동작하므로 비율이 깨지지 않음
                return float(current_epoch + 1) / float(self.warmup_epochs)
            else:
                power = 1.0
                # 전체 진행도 계산
                progress = (current_epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
                return max(0.0, (1 - progress) ** power)

        # Create the scheduler that uses the lambda function.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'name': 'learning_rate',
        }
        return [optimizer], [scheduler]