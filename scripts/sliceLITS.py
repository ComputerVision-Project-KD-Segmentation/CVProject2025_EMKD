import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description='Slice LITS')
parser.add_argument('--in_path', type=str, default='/data/lits/data')
parser.add_argument('--out_path', type=str, default='/data/lits')
parser.add_argument('--process_num', type=int, default=2)

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    paths1 = glob(os.path.join(args.in_path, "Training Batch 1", "volume-*.nii"))
    paths2 = glob(os.path.join(args.in_path, "Training Batch 2", "volume-*.nii"))
    paths = paths1 + paths2

    pool = Pool(args.process_num)
    pool.map(make_slice, paths)


def make_slice(path):
    """
    Cut 3D lits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    case, vol, seg = read_lits(path)

    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        if ct_slice.shape != [512, 512]:
            ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        mask_slice = seg[i, ...]
        np.savez_compressed(f'{args.out_path}/{case}_{i}.npz', ct=ct_slice, mask=mask_slice)

    print(f'complete making LITS {args.mode} slices of {case}')

def read_lits(path):
    vol = nib.load(path).get_fdata()
    seg = nib.load(path.replace('volume', 'segmentation')).get_fdata().astype('int8')
    case = path.split('-')[-1].split('.')[0]
    vol = np.transpose(vol, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    return case, vol, seg


if __name__ == '__main__':
    main()