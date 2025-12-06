import torch
from models import get_model
from pl_model.base import BasePLModel
from pl_model.segmentation_model import SegmentationPLModel
from utils.loss_functions import calc_loss, DSDLoss
from datasets.dataset import SliceDataset

from torch.utils.data import DataLoader


class KnowledgeDistillationPLModelEDGE(BasePLModel):
    """
    EDGE-based Knowledge Distillation Model
    - Method: EDGE (ECKD + SRKD + SAKD)
    - ECKD: Edge Constraint Knowledge Distillation
    - SRKD: Segmentation Refinement Knowledge Distillation (Hausdorff Distance)
    - SAKD: Scale Adaptation Knowledge Distillation (Multi-stage)
    """
    
    def __init__(self, params, train_indices, val_indices):
        super(KnowledgeDistillationPLModelEDGE, self).__init__() 
        self.save_hyperparameters(params)

        # 1. Load and freeze teacher net
        # Teacher는 inference만 하므로 dummy indices 사용
        self.t_net = SegmentationPLModel.load_from_checkpoint(
            checkpoint_path=self.hparams.tckpt,
            train_indices=[],  # dummy - teacher는 DataLoader 사용 안 함
            val_indices=[]     # dummy - teacher는 DataLoader 사용 안 함
        )
        self.t_net.freeze()  # 파라미터 동결 및 eval 모드 설정

        # 2. Student net
        self.net = get_model(self.hparams.smodel, channels=2)

        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # Stage : Final logits (SAKD)
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
        
        # Teacher model은 항상 eval 모드여야 함
        self.t_net.eval()
        
        # Teacher Forward (no gradient)
        with torch.no_grad():
            t_out, t_low, t_high = self.t_net.net(ct)
        
        # Student Forward
        output, low, high = self.net(ct)

        # Segmentation Loss (Focal + Dice)
        loss_seg = calc_loss(output, mask)

        # ==================== EDGE Knowledge Distillation ====================
        # Stage : Final logits (ECKD + SRKD)
        loss_dict = self.dsd_loss(
            feat_student=output,
            logits_teacher=t_out.detach(),
            label=mask
        )
        
        bkd_loss = loss_dict['bkd_loss']
        hd_loss = loss_dict['hd_loss']
        
        # Total EDGE distillation loss
        loss_edge = (
            bkd_loss + hd_loss
        )
        
        # Combined loss
        loss = loss_seg + loss_edge

        # ==================== Logging ====================
        
        # Main losses
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_seg', loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_edge', loss_edge, on_step=False, on_epoch=True, prog_bar=True)
        
        # EDGE components - Stage 3
        self.log('train_bkd3', bkd_loss, on_step=False, on_epoch=True)
        self.log('train_hd3', hd_loss, on_step=False, on_epoch=True)

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
