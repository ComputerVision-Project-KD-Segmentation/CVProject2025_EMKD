import os
import argparse
import gc

from pl_model.dinov3_knowledge_distillation_model import Dinov3KnowledgeDistillationPLModel
from datasets.dataset import load_case_mapping, split_train_val

from sklearn.model_selection import KFold
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def build_parser():
    parser = argparse.ArgumentParser('train_dinov3_kd')

    # 기본 경로 및 세팅
    parser.add_argument('--data_path', type=str, default='/data/kits/data')
    parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints',
                        help='디렉토리 경로 (여기에 last.ckpt 및 epoch별 ckpt 저장)')
    parser.add_argument('--tckpt', type=str,
                        default='/data/checkpoints/checkpoint_kits_tumor_enet_epoch=18.ckpt',
                        help='teacher model checkpoint path (.ckpt)')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--smodel', type=str, default='enet', help='Student model name')
    parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
    parser.add_argument('--kfold', action='store_true', help='Enable 5-fold cross validation')

    # KD loss 관련 하이퍼파라미터
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--beta3', type=float, default=0.0)
    parser.add_argument('--beta4', type=float, default=0.0)

    # last.ckpt에서 이어서 학습
    parser.add_argument('--resume_last', action='store_true',
                        help='Resume training from checkpoint_path/last.ckpt (single-split only)')
    return parser


def get_default_indices(args):
    """Test 모드 등에서 기본 Split 인덱스를 가져오기 위한 헬퍼 함수"""
    case_mapping = load_case_mapping(args.data_path, args.task)
    return split_train_val(case_mapping, train_ratio=0.8, seed=args.seed)


def run_single(args):
    """단일 train/val split에서 KD 학습"""
    seed_everything(args.seed, workers=True)

    case_mapping = load_case_mapping(args.data_path, args.task)
    train_indices, val_indices = split_train_val(
        case_mapping, train_ratio=0.8, seed=args.seed
    )

    model = Dinov3KnowledgeDistillationPLModel(args, train_indices, val_indices)

    # Checkpoint 콜백 (last.ckpt 포함)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename=f'checkpoint_{args.dataset}_{args.task}_kd_{args.smodel}_' + '{epoch}',
        save_last=True,     # 항상 last.ckpt 저장
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger(
        'log',
        name=f'{args.dataset}_{args.task}_kd_{args.smodel}'
    )

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=10
    )

    # last.ckpt에서 이어서 학습
    ckpt_path = None
    if args.resume_last:
        last_path = os.path.join(args.checkpoint_path, 'last.ckpt')
        if os.path.exists(last_path):
            print(f"[Resume] Resuming training from last checkpoint: {last_path}")
            ckpt_path = last_path
        else:
            print(f"[Warning] last.ckpt not found at: {last_path}")
            print("         새로 학습을 시작합니다.")

    trainer.fit(model, ckpt_path=ckpt_path)


def run_k_fold(args):
    """5-fold case-level cross validation"""
    seed_everything(args.seed, workers=True)

    all_cases = load_case_mapping(args.data_path, args.task)
    case_ids = np.array(sorted(all_cases.keys()))
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    if args.resume_last:
        print("[Warning] --resume_last는 k-fold 모드에서는 사용하지 않습니다. (각 fold별 fresh training)")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(case_ids)):
        print(f"\n{'=' * 20}")
        print(f"Start Training Fold: {fold} / 4")
        print(f"{'=' * 20}")

        # 인덱스를 이용해 실제 case ID 리스트 추출
        train_cases = case_ids[train_idx]
        val_cases = case_ids[val_idx]

        train_indices = []
        for case_id in train_cases:
            train_indices.extend(all_cases[case_id]['indices'])

        val_indices = []
        for case_id in val_cases:
            val_indices.extend(all_cases[case_id]['indices'])

        print(f" - Cases:  Train {len(train_cases)}, Val {len(val_cases)}")
        print(f" - Slices: Train {len(train_indices)}, Val {len(val_indices)}")

        # 모델 초기화
        model = Dinov3KnowledgeDistillationPLModel(
            args,
            train_indices=train_indices,
            val_indices=val_indices
        )

        # Fold별 체크포인트 디렉토리
        fold_ckpt_dir = os.path.join(args.checkpoint_path, f'fold{fold}')

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_ckpt_dir,
            filename=f'checkpoint_{args.dataset}_{args.task}_kd_{args.smodel}_fold{fold}_' + '{epoch}',
            save_last=True,
            save_top_k=5,
            monitor='dice_class0',
            mode='max',
            verbose=True
        )

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

        # 메모리 정리
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_test(args):
    """last.ckpt를 이용해 test 모드 실행"""
    seed_everything(args.seed, workers=True)

    train_indices, val_indices = get_default_indices(args)

    ckpt_path = os.path.join(args.checkpoint_path, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"[Warning] last.ckpt not found at: {ckpt_path}")
        return

    print(f"[Test] Loading checkpoint: {ckpt_path}")

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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'train':
        if args.kfold:
            run_k_fold(args)
        else:
            run_single(args)
    elif args.mode == 'test':
        run_test(args)


if __name__ == '__main__':
    main()
