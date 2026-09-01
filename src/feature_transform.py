## feature_transform.py는 npz 파일에 저장된 feature들의 변환(transform) 및 역변환(inverse transform)을 수행하는 코드입니다.
## 명령어는 python feature_transform.py --dataset <dataset_name> 형태로 실행할 수 있습니다.
import numpy as np


LOG1P_FEATURES = [
    "Muon_pt",
    "Muon_pfRelIso04_all",
]

SIGNED_LOG1P_FEATURES = [
    "Muon_dxy",
    "Muon_dz",
]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log1p(x):
    return np.sign(x) * np.expm1(np.abs(x))


def transform_features(X, features):
    X_new = X.copy()

    for i, feature in enumerate(features):
        if feature in LOG1P_FEATURES:
            X_new[..., i] = np.log1p(np.maximum(X_new[..., i], 0.0))

        elif feature in SIGNED_LOG1P_FEATURES:
            X_new[..., i] = signed_log1p(X_new[..., i])

    return X_new


def inverse_transform_features(X, features):
    X_new = X.copy()

    for i, feature in enumerate(features):
        if feature in LOG1P_FEATURES:
            X_new[..., i] = np.expm1(X_new[..., i])

        elif feature in SIGNED_LOG1P_FEATURES:
            X_new[..., i] = inverse_signed_log1p(X_new[..., i])

    return X_new