import os
import argparse
import gc # 가비지 컬렉션을 위해 추가

from pl_model.segmentation_model import SegmentationPLModel
from datasets.dataset import load_case_mapping, split_train_val

from sklearn.model_selection import KFold
import numpy as np

import torch
# PyTorch Lightning 2.x에서는 lightning.pytorch를 권장하지만, 호환성을 위해 유지
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser('train')
parser.add_argument('--data_path', type=str, default='/data/kits/data')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--model', type=str, default='raunet')
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--kfold', action='store_true', help='Enable 5-fold cross validation')

def get_default_indices(args):
    """Test 모드 등에서 기본 Split 인덱스를 가져오기 위한 헬퍼 함수"""
    case_mapping = load_case_mapping(args.data_path, args.task)
    return split_train_val(case_mapping, train_ratio=0.8, seed=args.seed)

def main():
    args = parser.parse_args()
    seed_everything(args.seed, workers=True) # workers=True로 데이터 로더 시드까지 고정
    
    # 1. 데이터 Split
    case_mapping = load_case_mapping(args.data_path, args.task)
    train_indices, val_indices = split_train_val(
        case_mapping, train_ratio=0.8, seed=args.seed
    )

    # 2. 모델 초기화
    model = SegmentationPLModel(args, train_indices, val_indices)

    # 3. Checkpoint 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f'checkpoint_{args.dataset}_{args.task}_{args.model}_' + '{epoch}',
        save_last=True,
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    # 4. Logger 설정
    logger = TensorBoardLogger('log', name=f'{args.dataset}_{args.task}_{args.model}')
    
    # 5. Trainer 설정
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # GPU 개수 (필요시 args로 받도록 수정 가능)
        max_epochs=args.epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=False, # 코랩에서 프로그레스바 출력으로 과도하게 남음
        logger=logger,
        log_every_n_steps=10 # 로깅 빈도 설정
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
        
        # 모델 초기화 (현재 Fold의 인덱스 전달)
        model = SegmentationPLModel(args, train_indices=train_indices, val_indices=val_indices)

        # Checkpoint: 폴더 구조로 fold 구분
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, f'fold{fold}'),
            filename=f'checkpoint_{args.dataset}_{args.task}_{args.model}_fold{fold}_' + '{epoch}',
            save_last=True,
            save_top_k=5,
            monitor='dice_class0',
            mode='max',
            verbose=True
        )

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
        
        # 메모리 정리 (중요)
        del model, trainer
        gc.collect() 
        torch.cuda.empty_cache()

def test():
    args = parser.parse_args()
    
    # [중요 수정] load_from_checkpoint 호출 시 __init__에 필요한 인자를 넘겨줘야 함
    # 저장된 hparams에는 train_indices, val_indices 같은 대용량 리스트는 보통 포함되지 않기 때문입니다.
    # 여기서는 기본 Split(main 함수 로직)을 기준으로 인덱스를 재생성하여 전달합니다.
    # 만약 K-Fold 특정 Fold를 테스트하려면 해당 로직에 맞는 인덱스를 구해서 넣어야 합니다.
    train_indices, val_indices = get_default_indices(args)

    ckpt_path = os.path.join(args.checkpoint_path, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint from: {ckpt_path}")
    
    # args=args는 hparams에 저장된 args와 충돌할 수 있으나, 덮어쓰기 위해 전달 가능
    # 중요한 건 train_indices와 val_indices를 전달하는 것입니다.
    model = SegmentationPLModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        params=args, # 만약 __init__에서 params를 쓴다면 명시
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
    elif args.mode == 'test':
        test()