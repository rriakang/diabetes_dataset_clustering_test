# data_preparation.py

# --- 모든 import ---
import os
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
from torchvision import datasets, transforms
import hashlib
import shutil

# --- (신규) Kaggle 데이터셋 로드를 위한 import ---
try:
    import kagglehub
except ImportError:
    logging.error("kagglehub 라이브러리가 필요합니다. 'pip install kagglehub'를 실행하세요.")

# Non-IID partition utility (if needed, though this example uses IID)
# from fedops.utils.fedco.datasetting import build_parts

# --- 로깅 설정 ---
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)


# --- (신규) Kaggle 데이터셋 설정 ---
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./dataset")).resolve()
KAGGLE_DATASET_SLUG = "mohankrishnathalla/diabetes-health-indicators-dataset"
ACTUAL_CSV_FILENAME = "diabetes_dataset.csv"
ACTUAL_TARGET_COLUMN = "diagnosed_diabetes"
ACTUAL_FEATURE_COUNT = 21


# --- (수정) Kaggle 데이터셋 다운로드 헬퍼 ---
def _ensure_dataset_ready() -> Path:
    """
    (수정) KaggleHub API를 사용해 데이터셋을 다운로드하고,
    로컬 DATASET_DIR로 심볼릭 링크 또는 복사 후,
    (수정) .rglob()을 사용해 CSV 파일을 재귀적으로 탐색합니다.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    slug_name = KAGGLE_DATASET_SLUG.split('/')[-1]
    
    # 최종 목표 경로 (예: ./dataset/diabetes-health-indicators-dataset)
    target_dir_path = DATASET_DIR / slug_name
    
    # 1. 파일이 이미 로컬 경로에 존재하면 즉시 반환
    #    (rglob로 탐색하여 이미 존재하는지 확인)
    if target_dir_path.exists():
        logger.info(f"Checking for CSV in existing path: {target_dir_path}")
        # ** (수정) rglob로 파일 탐색 **
        csv_files = list(target_dir_path.rglob(ACTUAL_CSV_FILENAME))
        if csv_files:
            logger.info(f"Using cached dataset: {csv_files[0]}")
            return csv_files[0] # 첫 번째 일치하는 파일 반환

    # 2. 파일이 없으면 KaggleHub 캐시에서 다운로드 시도
    logger.info(f"Downloading Kaggle dataset via KaggleHub: {KAGGLE_DATASET_SLUG}...")
    
    try:
        # kagglehub가 기본 캐시 폴더에 다운로드하고 그 경로를 반환
        src_path = Path(kagglehub.dataset_download(
            KAGGLE_DATASET_SLUG
        ))
        logger.info(f"KaggleHub cache path: {src_path}")

        # 3. 캐시 경로 -> 로컬 DATASET_DIR로 링크 또는 복사
        if not target_dir_path.exists():
            try:
                target_dir_path.symlink_to(src_path, target_is_directory=True)
                logger.info(f"Linked {src_path} -> {target_dir_path}")
            except Exception as e:
                logger.warning(f"Symlink failed ({e}), copying instead...")
                shutil.copytree(src_path, target_dir_path)
                logger.info(f"Copied {src_path} -> {target_dir_path}")
        
        # 4. (수정) 로컬 경로에서 CSV 파일을 재귀적으로 탐색
        logger.info(f"Searching for '{ACTUAL_CSV_FILENAME}' in {target_dir_path}...")
        csv_files = list(target_dir_path.rglob(ACTUAL_CSV_FILENAME))
        
        if csv_files:
            final_csv_path = csv_files[0]
            logger.info(f"Found CSV file: {final_csv_path}")
            return final_csv_path
        else:
            # (오류 수정) 이 부분이 실제 오류 원인이었습니다.
            logger.error(f"KaggleHub 데이터셋 처리 실패: CSV file not found in target path: {target_dir_path}")
            # 디버깅을 위해 폴더 내용 일부를 로깅
            contents = [str(p.relative_to(target_dir_path)) for p in target_dir_path.rglob('*')]
            logger.error(f"Directory contents (max 10): {contents[:10]}")
            return None
            
    except Exception as e:
        logger.error(f"KaggleHub 데이터셋 처리 실패: {e}")
        logger.error("Kaggle API 토큰이 올바른지 확인하세요 (e.g., ~/.kaggle/kaggle.json)")
        logger.error("또는 'pip install kagglehub'가 실행되었는지 확인하세요.")
        return None


# --- (수정) 데이터 로드 및 분리 헬퍼 함수 ---
def _load_and_split_diabetes_data(seed):
    csv_path = _ensure_dataset_ready()
    if csv_path is None:
        return None, None, None, None  # ← 리턴 4개로 변경

    df = pd.read_csv(csv_path)

    TARGET_COLUMN = ACTUAL_TARGET_COLUMN  # "Diabetes_binary"
    if TARGET_COLUMN not in df.columns:
        logger.error(f"타겟 컬럼 '{TARGET_COLUMN}'을 데이터에서 찾을 수 없습니다.")
        return None, None, None, None

    # 1) object 컬럼만 원-핫
    obj_cols = df.dtypes[df.dtypes == object].index.tolist()
    df_encoded = pd.get_dummies(df, columns=obj_cols, dtype=int)

    # 2) 전부 숫자형 강제 + NaN/inf 정리 (안전)
    for c in df_encoded.columns:
        if c != TARGET_COLUMN:
            df_encoded[c] = pd.to_numeric(df_encoded[c], errors="coerce")
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True)).fillna(0)

    # 3) 피처/타깃 분리 및 공통 컬럼 리스트
    feature_cols = [c for c in df_encoded.columns if c != TARGET_COLUMN]

    # 4) 80/20 split (stratify)
    TEST_SPLIT_RATIO = 0.2
    train_val_df, test_df = train_test_split(
        df_encoded,
        test_size=TEST_SPLIT_RATIO,
        random_state=seed,
        stratify=df_encoded[TARGET_COLUMN],
    )

    # 5) (선택) 모델이 입력 차원을 알도록 env에 기록
    os.environ["FEDOPS_INPUT_DIM"] = str(len(feature_cols))

    logger.info(f"데이터 로드 완료: {len(train_val_df)} (Train/Val), {len(test_df)} (Test), input_dim={len(feature_cols)}")

    return train_val_df, test_df, TARGET_COLUMN, feature_cols  # ← feature_cols 추가



# --- 클라이언트 파티션 로드 함수 (수정) ---
# (이 함수는 _load_and_split_diabetes_data를 호출하므로, 코드는 동일하게 유지)
def load_partition(dataset, validation_split, batch_size) :
    # ... (이하 코드는 이전과 동일) ...
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    num_clients = int(os.getenv("FEDOPS_NUM_CLIENTS", "3"))
    client_id   = int(os.getenv("FEDOPS_CLIENT_ID", "1")) # 1-based ID
    seed        = int(os.getenv("FEDOPS_SEED", "42"))
    
    logging.info(f"[partition] num_clients={num_clients}, client_id={client_id}, seed={seed}")

    train_val_df, test_df, TARGET_COLUMN, feature_cols = _load_and_split_diabetes_data(seed)
    if train_val_df is None:
        return None, None, None

    # IID 파티셔닝
    df_shuffled = train_val_df.sample(frac=1, random_state=seed)
    client_dfs = np.array_split(df_shuffled, num_clients)
    client_df = client_dfs[client_id - 1]

    # ★ 항상 동일한 피처셋/순서 보장
    X_client_df = client_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_client = client_df[TARGET_COLUMN].astype(int).values

    # 안전: float32로 보장
    X_client = X_client_df.to_numpy(dtype=np.float32)

    can_stratify = np.min(np.bincount(y_client)) > 1 if len(y_client) > 0 else False
    X_train, X_val, y_train, y_val = train_test_split(
        X_client, y_client,
        test_size=validation_split,
        random_state=seed,
        stratify=y_client if can_stratify else None
    )

    # 텐서 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.int64)

    trainloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(TensorDataset(X_val_tensor,   y_val_tensor),   batch_size=batch_size, shuffle=False)

    # ★ 테스트도 동일 컬럼셋으로 reindex
    X_test_df = test_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_test = test_df[TARGET_COLUMN].astype(int).values
    X_test = X_test_df.to_numpy(dtype=np.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    testloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader


# --- 글로벌 검증 함수 (수정) ---
# (이 함수는 _load_and_split_diabetes_data를 호출하므로, 코드는 동일하게 유지)
def gl_model_torch_validation(batch_size):
    """
    (서버용) 글로벌 모델 검증 로더.
    _load_and_split_diabetes_data가 4개를 반환하므로 여기도 4개를 받는다.
    """
    seed = int(os.getenv("FEDOPS_SEED", "42"))

    # ⬇⬇⬇ 여기 4개로 변경
    _, test_df, TARGET_COLUMN, feature_cols = _load_and_split_diabetes_data(seed)
    if test_df is None:
        logging.error("글로벌 테스트 데이터 로드 실패.")
        return None

    # 항상 동일한 컬럼셋/순서로 정렬 (학습과 동일)
    X_test_df = test_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_test = test_df[TARGET_COLUMN].astype(int).values

    # 안전: float32로 변환
    X_test = X_test_df.to_numpy(dtype=np.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    gl_val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"글로벌 검증 로더 생성 완료: {len(test_df)}개 샘플.")
    return gl_val_loader

