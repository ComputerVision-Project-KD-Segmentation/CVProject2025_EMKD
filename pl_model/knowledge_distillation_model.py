import torch
from models import get_model
from pl_model.base import BasePLModel
from pl_model.segmentation_model import SegmentationPLModel
from datasets.dataset import SliceDataset

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss, DSDLoss
from utils.loss_functions import prediction_map_distillation, importance_maps_distillation, region_affinity_distillation

# KD loss parameters (클래스 내부나 config로 옮기는 것을 권장하지만, 일단 외부에 유지합니다)
alpha = 0.1
beta1 = 0.9
beta2 = 0.9
beta3 = 0.9
beta4 = 0.9

class KnowledgeDistillationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(KnowledgeDistillationPLModel, self).__init__()
        self.save_hyperparameters(params)
        self.alpha = self.hparams.get('alpha', alpha)
        self.beta1 = self.hparams.get('beta1', beta1)
        self.beta2 = self.hparams.get('beta2', beta2)
        self.beta3 = self.hparams.get('beta3', beta3)
        self.beta4 = self.hparams.get('beta4', beta4)

        # 1. Load and freeze teacher net
        # SegmentationPLModel도 LightningModule이므로 load_from_checkpoint 사용 가능
        # self.t_net = SegmentationPLModel.load_from_checkpoint(
        #     checkpoint_path=self.hparams.tckpt,
        #     train_indices=[],
        #     val_indices=[]
        # )
        self.t_net = SegmentationPLModel(
            params=self.hparams,
            train_indices=[],
            val_indices=[]
        )
        # 2. 체크포인트 파일 불러오기
        checkpoint = torch.load(self.hparams.tckpt, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # 3. 't_net'으로 시작하는 키 추가 (필터링)
        new_state_dict = {k: v for k, v in state_dict.items() if k.startswith('t_net.')}

        # 4. 필터링된 가중치를 모델에 적용 (strict=False 필수)
        self.t_net.load_state_dict(new_state_dict, strict=False)

        self.t_net.freeze() # 파라미터 동결 및 eval 모드 설정

        # 2. Student net
        self.net = get_model(self.hparams.smodel, channels=2)

        self.train_indices = train_indices
        self.val_indices = val_indices

        self.dsd_loss = DSDLoss(
            in_chans=2,  # Final logits
            num_classes=2,
            num_stages=3,
            cur_stage=3,
            kernel_size=3,
            interpolate=False,
            bd_include_background=False,
            hd_include_background=False,
            one_hot_target=True,
            sigmoid=False,
            softmax=True,
            tau=4,
            loss_weight=1.0,
            overall_loss_weight=1.0
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        
        # Teacher model은 항상 eval 모드여야 함 (Dropout, BatchNorm 고정)
        self.t_net.eval()
        
        # Teacher Forward
        # with torch.no_grad(): # freeze()를 했으면 자동 적용되지만 명시적으로 감싸도 됨
        t_out, t_low, t_high = self.t_net.net(ct)
        
        # Student Forward
        output, low, high = self.net(ct)

        # Loss Calculation
        loss_seg = calc_loss(output, mask)

        loss_pmd = prediction_map_distillation(output, t_out)
        loss_imd = importance_maps_distillation(low, t_low) + importance_maps_distillation(high, t_high)
        loss_rad = region_affinity_distillation(low, t_low, mask) + region_affinity_distillation(high, t_high, mask)

        loss_dict = self.dsd_loss(
            feat_student=output,
            logits_teacher=t_out.detach(),
            label=mask
        )
        
        bkd_loss = loss_dict['bkd_loss']
        hd_loss = loss_dict['hd_loss']

        loss = loss_seg + self.alpha * loss_pmd + self.beta1 * loss_imd + self.beta2 * loss_rad \
                + self.beta3 * bkd_loss + self.beta4 * hd_loss

        # [수정 1] Loss Logging (Lightning 2.x 스타일)
        # 각 컴포넌트 별 loss를 기록하면 학습 모니터링에 매우 유용합니다.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_seg', loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_pmd', loss_pmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_imd', loss_imd, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_rad', loss_rad, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_bkd', bkd_loss, on_step=False, on_epoch=True)
        self.log('train_hd', hd_loss, on_step=False, on_epoch=True)

        # [수정 2] 딕셔너리 대신 loss 텐서 반환
        return loss

    def validation_step(self, batch, batch_idx):
        # BasePLModel의 measure를 위해 test_step 로직 공유
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        
        # Student model inference
        output, low, high = self.net(ct)

        # BasePLModel의 measure 메서드 사용 (Dice 계산 등)
        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.train_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
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