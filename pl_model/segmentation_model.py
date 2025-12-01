import torch
from models import get_model
from pl_model.base import BasePLModel
from datasets.dataset import SliceDataset

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

class SegmentationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(SegmentationPLModel, self).__init__()
        # hparams 저장
        self.save_hyperparameters(params)
        
        # 모델 로드 (channels=2는 사용하시는 설정에 맞게 유지)
        self.net = get_model(self.hparams.model, channels=2)
        
        self.train_indices = train_indices
        self.val_indices = val_indices

    def forward(self, x):
        # 모델이 튜플을 반환한다고 가정 (output, aux1, aux2 등)
        output, _, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)  # Dice_loss Used

        # [수정 1] Loss 로깅 추가
        # on_epoch=True로 설정하여 Epoch가 끝날 때 자동으로 평균 Loss를 기록합니다.
        # prog_bar=True로 설정하면 진행률 표시줄(tqdm)에 loss가 표시됩니다.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # [수정 2] 딕셔너리가 아닌 loss 텐서 자체를 반환 (Lightning 2.0 권장)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step은 값을 반환하지 않아도 됩니다 (metric 누적은 내부에서 수행)
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)

        # BasePLModel의 measure 메서드를 호출하여 Metric(Dice 등) 누적
        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.train_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
        # num_workers, pin_memory 등은 하드웨어 성능 최적화를 위해 유지
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True, 
            shuffle=True
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
            pin_memory=True
        )

    # Lightning 구버전 호환성을 위해 유지 (최신 버전에서는 val_dataloader라고 명시해도 됨)
    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.epochs, eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        return [opt], [scheduler]