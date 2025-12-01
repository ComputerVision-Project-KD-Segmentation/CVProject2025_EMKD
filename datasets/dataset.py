import os
import copy
import random
import numpy as np
import torch
from datasets.data_aug import window_standardize, cut_384
from torch.utils.data import Dataset
from collections import defaultdict

class SliceDataset(Dataset):
    def __init__(self, 
                 data_path:str, 
                 indices=None,
                 task='tumor',
                 dataset='kits',
                 train=True
        ):
        super(SliceDataset, self).__init__()
        assert task in ['organ', 'tumor']
        assert dataset in ['lits', 'kits']
        
        self.load_path = data_path
        self.task = task
        self.train = train
        self.dataset = dataset

        if indices is not None:
            self.indices = indices
        else:
            # 전체 파일 스캔
            all_files = sorted(os.listdir(self.load_path))
            self.indices = [int(f.replace('.npz', '')) for f in all_files if f.endswith('.npz')]

    def rotate(self, img, mask, k=None):
        """ 90도 단위 회전 증강 """
        if k is None:
            k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k, (-2, -1))
        mask = np.rot90(mask, k, (-2, -1))
        return img, mask

    def flip(self, img, mask, flip=None):
        """ 좌우/상하 반전 증강 """
        if flip is None:
            a, b = random.choice([1, -1]), random.choice([1, -1])
        else:
            a, b = flip
        
        if img.ndim == 2:
            img = img[::a, ::b]
        elif img.ndim == 3: # (C, H, W)를 가정
            img = img[:, ::a, ::b]
        mask = mask[::a, ::b] # 마스크는 (C, H, W) 또는 (H, W)
        
        return img, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # 1. 데이터 로딩
        f_name = self.indices[item]
        case = f_name.split('_')[0]
        npz_path = os.path.join(self.load_path, f_name)
        
        try:
            npz = np.load(npz_path, allow_pickle=True)
            ct = npz.get('ct')
            mask = npz.get('mask')
        except Exception as e:
            print(f"Error loading file: {npz_path}")
            raise e

        # 2. 태스크별 마스크 전처리
        if self.task == 'organ':
            mask[mask > 0] = 1
        elif self.task == 'tumor':
            mask[mask != 2] = 0 # 종양(2)이 아닌 값은 0으로
            mask[mask == 2] = 1 # 종양(2) 값은 1로

        # 3. 데이터셋별 윈도잉 ('lits' 전용)
        if self.dataset == 'lits':
            ct = window_standardize(ct, -60, 140)
        elif self.dataset == 'kits':
            ct = window_standardize(ct, -200, 300)

        # 4. 데이터 증강 (훈련 시에만)
        if self.train:
            ct, mask = self.flip(ct, mask)
            ct, mask = self.rotate(ct, mask)

        # 5. 마스크 one-hot 인코딩 (배경/전경)
        img0 = copy.deepcopy(mask)
        img0 += 1
        img0[img0 != 1] = 0
        
        mask = np.stack((img0, mask), axis=0)
        mask[mask > 0] = 1

        # 6. 텐서 변환 및 크롭
        ct = torch.from_numpy(cut_384(ct.copy())).unsqueeze(0).float()
        mask = torch.from_numpy(cut_384(mask.copy())).float()

        return ct, mask, case


def load_case_mapping(data_path, task):
    """
    data_path에서 slices.npy를 기준으로 .npz 파일들을 스캔하여 case_id를 기준으로 그룹화합니다.
    파일명 포맷은 '00000_0.npz'로 가정합니다.
    
    Args:
        data_path (str): .npz 파일들이 저장된 디렉토리 경로
        
    Returns:
        dict: {case_id: {'indices': [파일명 리스트], 'n_slices': 슬라이스 개수}}
        예: {'00000': {'indices': ['00000_0.npz', '00000_1.npz'], 'n_slices': 2}}
    """
    assert task in ['organ', 'tumor']
    # defaultdict는 키가 없을 때 자동으로 빈 리스트([])를 생성해줍니다.
    case_map_raw = defaultdict(list)
    
    try:
        all_filenames = np.load(os.path.join(data_path, '%s_slices.npy' % (task)))
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_path}")
        return {}

    # 1. 모든 .npz 파일을 순회하며 case_id를 기준으로 그룹화
    for f_name in all_filenames:
        if f_name.endswith('.npz') and not f_name.startswith('.'):
            try:
                # 파일명을 '_' 기준으로 분리 (예: '00000_0.npz')
                case_id = f_name.split('_')[0]
                # case_id를 key로, 파일명(f_name)을 value 리스트에 추가
                case_map_raw[case_id].append(f_name)
            except IndexError:
                print(f"Warning: Skipping file with unexpected format: {f_name}")
                
    if not case_map_raw:
        print(f"Warning: No .npz files matching the format found in {data_path}")
        return {}

    # 2. 최종 딕셔너리 포맷으로 변환 (n_slices 추가 및 정렬)
    final_case_map = {}
    for case_id, indices in case_map_raw.items():
        # 슬라이스 순서(예: _0, _1, _10)가 올바르게 정렬되도록 처리
        # 키: '00000_10.npz' -> '10' (int)
        try:
            sorted_indices = sorted(indices, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        except ValueError:
            print(f"Warning: Could not sort indices numerically for case {case_id}. Using string sort.")
            sorted_indices = sorted(indices)

        # 요청한 포맷으로 딕셔너리 생성
        final_case_map[case_id] = {
            'indices': sorted_indices,
            'n_slices': len(sorted_indices)
        }
    return final_case_map

def split_train_val(case_mapping, train_ratio=0.8, seed=42):
    """
    Case-level train/val split
    
    Args:
        case_mapping: Case mapping dictionary
        train_ratio: 학습 데이터 비율
        seed: Random seed
    
    Returns:
        train_indices, val_indices: 학습/검증 인덱스 리스트
    """
    # 케이스 리스트
    case_ids = sorted(case_mapping.keys())
    
    # Shuffle
    random.seed(seed)
    random.shuffle(case_ids)
    
    # Split
    n_train = int(len(case_ids) * train_ratio)
    train_cases = case_ids[:n_train]
    val_cases = case_ids[n_train:]
    
    # 인덱스 수집
    train_indices = []
    val_indices = []
    
    for case_id in train_cases:
        train_indices.extend(case_mapping[case_id]['indices'])
    
    for case_id in val_cases:
        val_indices.extend(case_mapping[case_id]['indices'])
    
    print(f"\n[Split] Case-level split 완료")
    print(f"  Train: {len(train_cases)} cases, {len(train_indices):,} slices")
    print(f"  Val:   {len(val_cases)} cases, {len(val_indices):,} slices")
    
    return train_indices, val_indices