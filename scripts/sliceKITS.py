import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description='Slice KITS')
parser.add_argument('--in_path', type=str, default='/data/kits19/data')
parser.add_argument('--out_path', type=str, default='/data/kits/data')
parser.add_argument('--process_num', type=int, default=2)

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    paths = glob(os.path.join(args.in_path, "case_*/imaging*.nii.gz"))
    paths = [p for p in paths 
            if os.path.basename(os.path.dirname(p)) <= 'case_00209']

    pool = Pool(args.process_num)
    pool.map(make_slice, paths)


def make_slice(path):
    """
    Cut 3D kits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    case, vol, seg = read_kits(path)

    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        if ct_slice.shape != [512, 512]:
            ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        mask_slice = seg[i, ...]
        np.savez_compressed(f'{args.out_path}/{case}_{i}.npz', ct=ct_slice, mask=mask_slice)

    print(f'complete making KITS slices of {case}')


def read_kits(path):
    dir = os.path.dirname(path)
    vol = nib.load(path).get_fdata()
    seg = nib.load(os.path.join(dir, 'segmentation.nii.gz')).get_fdata().astype('int8')
    case = os.path.split(dir)[-1][-5:]
    return case, vol, seg


if __name__ == '__main__':
    main()