import os
import argparse
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='Collect Tumor/Organ Data')
parser.add_argument('--path', type=str, default='/data/kits19/data')
# 프로세스 개수 설정 (기본값: CPU 코어 수)
parser.add_argument('--process_num', type=int, default=cpu_count()) 

args = parser.parse_args()

def check_mask(path):
    """
    개별 파일 하나를 처리하는 함수입니다.
    process_num 개수만큼 프로세스에서 실행됩니다.
    """
    try:
        filename = os.path.basename(path)
        # 파일을 열고 닫기 위해 context manager 사용 권장, 혹은 로드 후 바로 사용
        # npz 파일 로드
        with np.load(path, allow_pickle=True) as npz:
            if 'mask' not in npz:
                return None
            mask = npz['mask']
            
            has_organ = False
            has_tumor = False
            
            # 연산 최적화: 0보다 큰게 있으면 organ, 1보다 큰게 있으면 tumor
            # np.any()는 조건 만족 시 즉시 반환하므로 빠름
            if np.any(mask > 0):
                has_organ = True
            if np.any(mask > 1):
                has_tumor = True
                
            return (filename, has_organ, has_tumor)
            
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def main():
    # 파일 리스트 가져오기
    paths = glob(os.path.join(args.path, "*_*.npz"))
    print(f"Total files to process: {len(paths)}")
    
    organ_results = []
    tumor_results = []
    
    # Multiprocessing Pool 생성
    # imap_unordered를 사용하면 처리되는 순서대로 결과를 받아옵니다 (약간 더 빠름)
    with Pool(processes=args.process_num) as pool:
        # tqdm을 사용하여 진행률 표시
        results = list(tqdm(pool.imap(check_mask, paths), total=len(paths)))

    # 결과 취합 (None인 경우 - 에러 발생 - 제외)
    for res in results:
        if res is None:
            continue
            
        filename, has_organ, has_tumor = res
        
        if has_organ:
            organ_results.append(filename)
        if has_tumor:
            tumor_results.append(filename)
    
    print("organ slices : ", len(organ_results))
    print("tumor slices :", len(tumor_results))
    
    # [수정됨] 기존 코드의 os.path.join(paths, ...)는 paths가 리스트라 에러가 납니다.
    # args.path(폴더 경로)로 수정했습니다.
    np.save(os.path.join(args.path, 'organ_slices.npy'), organ_results)
    np.save(os.path.join(args.path, 'tumor_slices.npy'), tumor_results)

if __name__ == '__main__':
    main()