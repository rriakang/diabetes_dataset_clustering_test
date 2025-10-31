# data_preparation.py

# --- 모든 import ---
import os
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
# torchvision은 여기선 사용 안 함 (이미지 아님)

# Non-IID partition utility (FedOps 제공 유틸)
from fedops.utils.fedco.datasetting import build_parts  # ← keep as-is

# --- 로깅 설정 ---
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger("data_preparation")


# --- Kaggle 데이터셋 설정/옵션 ---
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./dataset")).resolve()
KAGGLE_DATASET_SLUG = "mohankrishnathalla/diabetes-health-indicators-dataset"
ACTUAL_CSV_FILENAME = "diabetes_dataset.csv"

# 기본 타깃 후보들(순차로 탐색)
TARGET_CANDIDATES = [
    os.environ.get("ACTUAL_TARGET_COLUMN", "").strip() or "",
    "Diabetes_binary",          # Kaggle 원본
    "diagnosed_diabetes",       # 사용자가 만든 CSV일 가능성 대비
    "Outcome"                   # Pima 계열 대비
]

# kagglehub 사용 가능 여부
try:
    import kagglehub
    _HAS_KAGGLEHUB = True
except Exception:
    _HAS_KAGGLEHUB = False
    logger.warning("kagglehub 미설치: 로컬 CSV를 사용하거나 직접 경로를 제공하세요.")


# =============================
# 파티션 모드 파싱 (IID/Non-IID)
# =============================
def _resolve_mode_from_env() -> str:
    """
    FEDOPS_PARTITION_CODE:
      "0"=iid | "1"=dirichlet | "2"=label_skew | "3"=qty_skew
    추가 파라미터:
      FEDOPS_DIRICHLET_ALPHA (default 0.05)
      FEDOPS_LABELS_PER_CLIENT (default 2)
      FEDOPS_QTY_BETA (default 0.5)
    """
    code = os.getenv("FEDOPS_PARTITION_CODE", "1").strip()
    if code == "0":
        return "iid"
    elif code == "1":
        alpha = os.getenv("FEDOPS_DIRICHLET_ALPHA", "0.05").strip()
        return f"dirichlet:{alpha}"
    elif code == "2":
        n_labels = os.getenv("FEDOPS_LABELS_PER_CLIENT", "2").strip()
        return f"label_skew:{n_labels}"
    elif code == "3":
        beta = os.getenv("FEDOPS_QTY_BETA", "0.5").strip()
        return f"qty_skew:beta{beta}"
    else:
        logger.warning(f"[partition] Unknown FEDOPS_PARTITION_CODE={code}, fallback to iid")
        return "iid"


# =========================
# Kaggle/로컬 CSV 보장하기
# =========================
def _ensure_dataset_ready() -> Path:
    """
    ./dataset/<slug>/diabetes_dataset.csv 경로 확보.
    - 이미 있으면 그대로 사용
    - 없으면 kagglehub로 다운로드(가능한 경우), 심볼릭링크 or 복사
    - 실패 시 명확한 예외 발생
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    slug_name = KAGGLE_DATASET_SLUG.split("/")[-1]
    target_dir_path = DATASET_DIR / slug_name
    local_csv = target_dir_path / ACTUAL_CSV_FILENAME

    # 로컬 파일이 이미 있으면 사용
    if local_csv.exists():
        logger.info(f"file: {local_csv}")
        return local_csv

    # kagglehub 없으면 중단
    if not _HAS_KAGGLEHUB:
        raise RuntimeError(
            "kagglehub 미설치. 다음 중 하나 필요:\n"
            " - `pip install kagglehub && ~/.kaggle/kaggle.json` 설정\n"
            f" - 또는 CSV를 여기에 두세요: {local_csv}"
        )

    # kagglehub 다운로드
    logger.info(f"Downloading Kaggle dataset via KaggleHub: {KAGGLE_DATASET_SLUG}...")
    src_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG)).resolve()
    logger.info(f"KaggleHub cache path: {src_path}")

    if not target_dir_path.exists():
        try:
            target_dir_path.symlink_to(src_path, target_is_directory=True)
            logger.info(f"Linked {src_path} -> {target_dir_path}")
        except Exception as e:
            logger.warning(f"Symlink failed ({e}), copying instead...")
            shutil.copytree(src_path, target_dir_path)
            logger.info(f"Copied {src_path} -> {target_dir_path}")

    # CSV 찾기
    csv_files = list(target_dir_path.rglob(ACTUAL_CSV_FILENAME))
    if not csv_files:
        raise FileNotFoundError(
            f"CSV '{ACTUAL_CSV_FILENAME}' 미발견. 다음 경로를 확인/배치하세요: {target_dir_path}"
        )
    return csv_files[0]


# =========================
# 데이터 로드/원핫/분할
# =========================
def _pick_target_column(df: pd.DataFrame) -> str:
    for cand in TARGET_CANDIDATES:
        if cand and cand in df.columns:
            return cand
    raise KeyError(
        f"타깃 컬럼을 찾지 못했습니다. 후보: {TARGET_CANDIDATES}. "
        f"실제 컬럼: {df.columns.tolist()[:10]} ... (총 {len(df.columns)})"
    )


def _numericize_and_clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """object 컬럼 one-hot, 숫자형 강제, NaN/inf 처리."""
    obj_cols = df.dtypes[df.dtypes == object].index.tolist()
    df_encoded = pd.get_dummies(df, columns=obj_cols, dtype=int)

    for c in df_encoded.columns:
        if c != target_col:
            df_encoded[c] = pd.to_numeric(df_encoded[c], errors="coerce")

    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
    # 숫자형 중앙값 → 0 순으로 메움
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True)).fillna(0)
    return df_encoded


def _load_and_split_diabetes_data(seed: int):
    """
    CSV 확보 → DataFrame 로드 → 타깃 결정 → 인코딩 → 80/20 split → feature_cols 반환
    """
    csv_path = _ensure_dataset_ready()
    df = pd.read_csv(csv_path)

    target_col = _pick_target_column(df)
    df_encoded = _numericize_and_clean(df, target_col)

    feature_cols = [c for c in df_encoded.columns if c != target_col]

    TEST_SPLIT_RATIO = 0.2
    train_val_df, test_df = train_test_split(
        df_encoded,
        test_size=TEST_SPLIT_RATIO,
        random_state=seed,
        stratify=df_encoded[target_col],
    )

    # 모델이 입력 차원을 읽을 수 있도록 환경변수로 기록
    os.environ["FEDOPS_INPUT_DIM"] = str(len(feature_cols))

    logger.info(
        f"데이터 로드 완료: {len(train_val_df)} (Train/Val), {len(test_df)} (Test), input_dim={len(feature_cols)}"
    )

    return train_val_df, test_df, target_col, feature_cols


# =========================
# Public: 클라이언트 로더
# =========================
def load_partition(dataset, validation_split, batch_size):
    """
    Returns: trainloader, valloader, testloader
    - Non-IID은 build_parts로 클라이언트별 인덱스를 생성 (train_val_df 기준)
    - 글로벌 테스트셋은 모든 클라이언트 공통 사용
    """
    # 메타 로그
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'FL_Task - {json.dumps({"dataset": dataset, "start_execution_time": now})}')

    # 환경변수
    num_clients = int(os.getenv("FEDOPS_NUM_CLIENTS", "3"))
    client_id_1based = int(os.getenv("FEDOPS_CLIENT_ID", "1"))  # 1-based
    client_idx = client_id_1based - 1
    seed = int(os.getenv("FEDOPS_SEED", "42"))
    mode_str = _resolve_mode_from_env()

    if not (0 <= client_idx < num_clients):
        raise ValueError(f"[partition] client_id must be 1..{num_clients}, got {client_id_1based}")

    logger.info(f"[partition] mode={mode_str}, num_clients={num_clients}, client_id(1-based)={client_id_1based}, seed={seed}")

    # 로드/전처리/분할
    train_val_df, test_df, target_col, feature_cols = _load_and_split_diabetes_data(seed)

    # ===== Non-IID 인덱스 생성 =====
    labels_np = train_val_df[target_col].astype(int).to_numpy()
    parts = build_parts(labels_np, num_clients=num_clients, mode_str=mode_str, seed=seed)

    client_indices = parts[client_idx]
    if len(client_indices) == 0:
        logger.warning(f"[partition] client {client_id_1based} has 0 samples (mode={mode_str})")

    # 이 클라이언트의 데이터프레임
    client_df = train_val_df.iloc[client_indices].reset_index(drop=True)

    # 동일 컬럼셋/순서 보장
    X_client_df = client_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_client = client_df[target_col].astype(int).values
    X_client = X_client_df.to_numpy(dtype=np.float32)

    # Train/Val 분리
    can_stratify = np.min(np.bincount(y_client)) > 1 if len(y_client) > 0 else False
    X_train, X_val, y_train, y_val = train_test_split(
        X_client, y_client,
        test_size=validation_split,
        random_state=seed,
        stratify=y_client if can_stratify else None
    )

    # 텐서/로더
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.int64)

    trainloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(TensorDataset(X_val_tensor,   y_val_tensor),   batch_size=batch_size, shuffle=False)

    # 글로벌 테스트 (공통)
    X_test_df = test_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_test = test_df[target_col].astype(int).values
    X_test = X_test_df.to_numpy(dtype=np.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    testloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # 간단한 히스토그램(검증용)
    def _count_labels(arr):
        if arr is None or len(arr) == 0:
            return {}
        hist = dict(Counter(arr.tolist()))
        return hist

    logger.info(f"[partition] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"[partition] train_label_hist={_count_labels(y_train)}")

    return trainloader, valloader, testloader


# =========================
# Public: 서버 글로벌 검증 로더
# =========================
def gl_model_torch_validation(batch_size):
    """
    서버 측 글로벌 평가를 위한 전체 테스트셋 로더
    """
    seed = int(os.getenv("FEDOPS_SEED", "42"))
    _, test_df, target_col, feature_cols = _load_and_split_diabetes_data(seed)

    X_test_df = test_df.reindex(columns=feature_cols, fill_value=0).copy()
    y_test = test_df[target_col].astype(int).values
    X_test = X_test_df.to_numpy(dtype=np.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    gl_val_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    logger.info(f"글로벌 검증 로더 생성 완료: {len(test_df)}개 샘플.")
    return gl_val_loader
