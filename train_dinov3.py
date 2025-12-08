import os
import argparse
import gc # 가비지 컬렉션(메모리 해제)을 위해 필수

from pl_model.dinov3_segmentation_model import Dinov3SegmentationPLModel
from datasets.dataset import load_case_mapping, split_train_val

from sklearn.model_selection import KFold
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser('train_dinov3')
parser.add_argument('--model', type=str, default='dinov3_vit')
parser.add_argument('--data_path', type=str, default='/data/kits/data')
parser.add_argument('--vit_checkpoint_path', type=str, default='/data/checkpoints/dinov3_vit_pretrain.pth') # 기본값 예시 수정
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--enable_progress_bar', type=bool, default=True)
parser.add_argument('--kfold', action='store_true', help='Enable 5-fold cross validation')

def get_default_indices(args):
    """Test 모드 등에서 기본 Split 인덱스를 가져오기 위한 헬퍼 함수"""
    case_mapping = load_case_mapping(args.data_path, args.task)
    return split_train_val(case_mapping, train_ratio=0.8, seed=args.seed)

def main():
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    
    case_mapping = load_case_mapping(args.data_path, args.task)
    train_indices, val_indices = split_train_val(
        case_mapping, train_ratio=0.8, seed=args.seed
    )

    model = Dinov3SegmentationPLModel(args, train_indices, val_indices)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f'checkpoint_{args.dataset}_{args.task}_{args.model}_' + '{epoch}',
        save_last=True,
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger('log', name=f'{args.dataset}_{args.task}_{args.model}')
    
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=args.enable_progress_bar, # 진행바 활성화
        logger=logger,
        log_every_n_steps=10
    )
    trainer.fit(model)

def main_k_fold():
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    
    all_cases = load_case_mapping(args.data_path, args.task)
    case_ids = np.array(sorted(all_cases.keys()))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(case_ids)):
        print(f"\n{'='*20}")
        print(f"Start Training Fold: {fold} / 4")
        print(f"{'='*20}")

        train_cases = case_ids[train_idx]
        val_cases = case_ids[val_idx]

        train_indices = []
        for case_id in train_cases:
            train_indices.extend(all_cases[case_id]['indices'])
            
        val_indices = []
        for case_id in val_cases:
            val_indices.extend(all_cases[case_id]['indices'])
        
        print(f" - Cases: Train {len(train_cases)}, Val {len(val_cases)}")
        print(f" - Slices: Train {len(train_indices)}, Val {len(val_indices)}")
        
        # 모델 초기화
        model = Dinov3SegmentationPLModel(args, train_indices, val_indices)

        # Checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, f'fold{fold}'),
            filename=f'checkpoint_{args.dataset}_{args.task}_{args.model}_fold{fold}_' + '{epoch}',
            save_last=True,
            save_top_k=5,
            monitor='dice_class0',
            mode='max',
            verbose=True
        )

        # Logger
        logger = TensorBoardLogger(
            'log', 
            name=f'{args.dataset}_{args.task}_{args.model}',
            version=f'fold_{fold}' 
        )

        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=args.epochs, 
            callbacks=[checkpoint_callback], 
            enable_progress_bar=False,
            logger=logger,
            log_every_n_steps=10
        )
        
        trainer.fit(model)
        
        # [중요] DINOv3는 무거운 모델이므로 Fold 종료 후 메모리 정리가 필수적입니다.
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

def test():
    args = parser.parse_args()
    
    # [수정] Test 시에도 __init__에 필요한 인자를 넘겨줘야 함
    train_indices, val_indices = get_default_indices(args)

    ckpt_path = os.path.join(args.checkpoint_path, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")

    # args=args를 전달하여 hparams 덮어쓰기 (특히 vit_checkpoint_path 등 경로 문제 방지)
    model = Dinov3SegmentationPLModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        params=args,
        train_indices=train_indices,
        val_indices=val_indices
    )
    
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=args.enable_progress_bar
    )
    trainer.test(model)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        if args.kfold:
            main_k_fold()
        else:
            main()
    elif args.mode == 'test':
        test()