import os
import argparse
import gc # 가비지 컬렉션을 위해 추가

from pl_model.dinov3_knowledge_distillation_model import Dinov3KnowledgeDistillationPLModel
from datasets.dataset import load_case_mapping, split_train_val

from sklearn.model_selection import KFold
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser('train_kd')
parser.add_argument('--data_path', type=str, default='/data/kits/data')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--tckpt', type=str, default='/data/checkpoints/checkpoint_kits_tumor_enet_epoch=18.ckpt', help='teacher model checkpoint path')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--smodel', type=str, default='enet') # Student Model
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--kfold', action='store_true', help='Enable 5-fold cross validation')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--beta3', type=float, default=0)
parser.add_argument('--beta4', type=float, default=0)


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

    model = Dinov3KnowledgeDistillationPLModel(args, train_indices, val_indices)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        # [수정] args.smodel 사용
        filename=f'checkpoint_{args.dataset}_{args.task}_kd_{args.smodel}_' + '{epoch}',
        save_last=True,
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger('log', name=f'{args.dataset}_{args.task}_kd_{args.smodel}')
    
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=False, # 진행상황 보이게 수정
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

    # [수정] split(all_cases) -> split(case_ids)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(case_ids)):
        print(f"\n{'='*20}")
        print(f"Start Training Fold: {fold} / 4")
        print(f"{'='*20}")

        # 인덱스를 이용해 실제 데이터 ID 리스트 추출
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
        model = Dinov3KnowledgeDistillationPLModel(args, train_indices=train_indices, val_indices=val_indices)

        # Checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, f'fold{fold}'),
            # [수정] args.model -> args.smodel (Parser에는 smodel로 정의됨)
            filename=f'checkpoint_{args.dataset}_{args.task}_kd_{args.smodel}_fold{fold}_' + '{epoch}',
            save_last=True,
            save_top_k=5,
            monitor='dice_class0',
            mode='max',
            verbose=True
        )

        # Logger
        logger = TensorBoardLogger(
            'log', 
            name=f'{args.dataset}_{args.task}_kd_{args.smodel}',
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
        
        # [추가] 메모리 정리
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

def test():
    args = parser.parse_args()

    # [수정] Test 시에도 __init__에 필요한 인자를 넘겨줘야 함
    # 저장된 체크포인트의 hparams에는 train_indices가 없을 확률이 높음
    train_indices, val_indices = get_default_indices(args)

    ckpt_path = os.path.join(args.checkpoint_path, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        # K-fold 사용시 경로가 다를 수 있음
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    
    # args=args를 전달하여 hparams 덮어쓰기 가능 (Teacher 경로 등 확보)
    model = Dinov3KnowledgeDistillationPLModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        params=args, 
        train_indices=train_indices,
        val_indices=val_indices
    )

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True
    )
    trainer.test(model)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        if args.kfold:
            main_k_fold()
        else:
            main()