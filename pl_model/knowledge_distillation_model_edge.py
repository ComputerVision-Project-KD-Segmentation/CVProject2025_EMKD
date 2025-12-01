import torch
from models import get_model
from pl_model.base import BasePLModel
from pl_model.segmentation_model import SegmentationPLModel
from utils.loss_functions_edge import calc_loss, DSDLoss 
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
            #params=self.hparams,
            train_indices=[],  # dummy - teacher는 DataLoader 사용 안 함
            val_indices=[]     # dummy - teacher는 DataLoader 사용 안 함
        )
        self.t_net.freeze()  # 파라미터 동결 및 eval 모드 설정

        # 2. Student net
        self.net = get_model(self.hparams.smodel, channels=2)

        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # 3. EDGE Distillation Losses (Multi-stage)
        # Get student feature dimensions based on model type
        if self.hparams.smodel == 'enet':
            # ENet feature dimensions
            stage1_chans = 16   # Low-level features
            stage2_chans = 128  # High-level features
        elif self.hparams.smodel == 'unet':
            # U-Net feature dimensions
            stage1_chans = 64
            stage2_chans = 128
        else:
            # Default dimensions
            stage1_chans = 64
            stage2_chans = 128
        
        # Stage 1: Low-level features (SAKD)
        self.dsd_loss1 = DSDLoss(
            in_chans=stage1_chans,
            num_classes=2,
            num_stages=3,
            cur_stage=1,
            kernel_size=3,
            interpolate=False,
            bd_include_background=False,  # Exclude background for ECKD
            hd_include_background=False,  # Exclude background for SRKD
            one_hot_target=True,
            sigmoid=False,
            softmax=True,
            tau=4,
            loss_weight=1.0,
            overall_loss_weight=1.0
        )
        
        # Stage 2: High-level features (SAKD)
        self.dsd_loss2 = DSDLoss(
            in_chans=stage2_chans,
            num_classes=2,
            num_stages=3,
            cur_stage=2,
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
        
        # Stage 3: Final logits (SAKD)
        self.dsd_loss3 = DSDLoss(
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
        
        # Stage 1: Low-level features (ECKD + SRKD + SAKD)
        loss_dict1 = self.dsd_loss1(
            feat_student=low,
            logits_teacher=t_out.detach(),
            label=mask
        )
        
        # Stage 2: High-level features (ECKD + SRKD + SAKD)
        loss_dict2 = self.dsd_loss2(
            feat_student=high,
            logits_teacher=t_out.detach(),
            label=mask
        )
        
        # Stage 3: Final logits (ECKD + SRKD)
        loss_dict3 = self.dsd_loss3(
            feat_student=output,
            logits_teacher=t_out.detach(),
            label=mask
        )
        
        # Extract individual losses
        bkd_loss1 = loss_dict1['bkd_loss']  # ECKD (Edge Constraint)
        hd_loss1 = loss_dict1['hd_loss']    # SRKD (Hausdorff Distance)
        
        bkd_loss2 = loss_dict2['bkd_loss']
        hd_loss2 = loss_dict2['hd_loss']
        
        bkd_loss3 = loss_dict3['bkd_loss']
        hd_loss3 = loss_dict3['hd_loss']
        
        # Total EDGE distillation loss
        loss_edge = (
            bkd_loss1 + hd_loss1 +
            bkd_loss2 + hd_loss2 +
            bkd_loss3 + hd_loss3
        )
        
        # Combined loss
        loss = loss_seg + loss_edge

        # ==================== Logging ====================
        
        # Main losses
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_seg', loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_edge', loss_edge, on_step=False, on_epoch=True, prog_bar=True)
        
        # EDGE components - Stage 1
        self.log('train_bkd1', bkd_loss1, on_step=False, on_epoch=True)
        self.log('train_hd1', hd_loss1, on_step=False, on_epoch=True)
        
        # EDGE components - Stage 2
        self.log('train_bkd2', bkd_loss2, on_step=False, on_epoch=True)
        self.log('train_hd2', hd_loss2, on_step=False, on_epoch=True)
        
        # EDGE components - Stage 3
        self.log('train_bkd3', bkd_loss3, on_step=False, on_epoch=True)
        self.log('train_hd3', hd_loss3, on_step=False, on_epoch=True)

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