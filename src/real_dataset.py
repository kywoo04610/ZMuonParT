## real_dataset.py
## 실제 CMS SingleMuon 데이터를 모델 입력용 npz로 변환하는 코드입니다.
##
## 출력:
##   X       : (N, 8, 6)
##   mask    : (N, 8)
##   m_mumu  : (N,) leading OS muon pair invariant mass
##   features: feature name list
##
## 주의:
##   m_mumu는 검증용으로만 저장합니다.
##   모델 입력 X에는 절대 넣지 않습니다.
##
## 실행 예:
## python real_dataset.py \
##     --input-dir ../data/Real/SingleMuon \
##     --output ../processed/real_singlemuon_dataset.npz

import argparse
import glob
import os

import awkward as ak
import numpy as np
import uproot


MAX_MUONS = 8
MUON_MASS_GEV = 0.105658

FEATURES = np.array(
    [
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_pfRelIso04_all",
        "Muon_dxy",
        "Muon_dz",
    ]
)


BRANCHES = [
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_pfRelIso04_all",
    "Muon_dxy",
    "Muon_dz",
    "Muon_charge",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="../data/Real/SingleMuon",
        help="Directory containing real SingleMuon ROOT files.",
    )
    parser.add_argument(
        "--output",
        default="../processed/real_singlemuon_dataset.npz",
        help="Output npz path.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of ROOT files to process. Useful for quick tests.",
    )
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help="uproot iterate step size.",
    )
    return parser.parse_args()


def get_root_files(input_dir, max_files=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.root")))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {input_dir}")

    return files


def build_feature_array(events):
    """
    Build padded muon feature tensor.

    Returns
    -------
    X:
        shape (num_events, MAX_MUONS, 6)

    mask:
        shape (num_events, MAX_MUONS)
    """

    pt = events["Muon_pt"]
    eta = events["Muon_eta"]
    phi = events["Muon_phi"]
    iso = events["Muon_pfRelIso04_all"]
    dxy = events["Muon_dxy"]
    dz = events["Muon_dz"]

    # Sort muons by descending pt.
    order = ak.argsort(pt, ascending=False)

    pt = pt[order]
    eta = eta[order]
    phi = phi[order]
    iso = iso[order]
    dxy = dxy[order]
    dz = dz[order]

    muons = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "iso": iso,
            "dxy": dxy,
            "dz": dz,
        }
    )

    muons = muons[:, :MAX_MUONS]

    n_muons = ak.num(muons.pt)
    mask = ak.local_index(muons.pt, axis=1) < n_muons

    padded = ak.pad_none(muons, MAX_MUONS, axis=1, clip=True)

    X = np.stack(
        [
            ak.to_numpy(ak.fill_none(padded.pt, 0.0)),
            ak.to_numpy(ak.fill_none(padded.eta, 0.0)),
            ak.to_numpy(ak.fill_none(padded.phi, 0.0)),
            ak.to_numpy(ak.fill_none(padded.iso, 0.0)),
            ak.to_numpy(ak.fill_none(padded.dxy, 0.0)),
            ak.to_numpy(ak.fill_none(padded.dz, 0.0)),
        ],
        axis=-1,
    )

    mask = ak.to_numpy(ak.fill_none(ak.pad_none(mask, MAX_MUONS, axis=1, clip=True), False))

    return X.astype(np.float32), mask.astype(bool)


def compute_pair_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    """
    Compute invariant mass of two muons using pt, eta, phi, and fixed muon mass.
    """

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + MUON_MASS_GEV**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + MUON_MASS_GEV**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass2 = e**2 - px**2 - py**2 - pz**2
    mass2 = np.maximum(mass2, 0.0)

    return np.sqrt(mass2)


def compute_leading_os_mumu_mass(events):
    """
    Compute invariant mass of leading opposite-sign muon pair.

    Pair choice:
        Among opposite-sign muon pairs, choose the pair with largest pt1 + pt2.

    This avoids choosing the pair closest to Z mass, which would bias the validation.
    """

    pt = events["Muon_pt"]
    eta = events["Muon_eta"]
    phi = events["Muon_phi"]
    charge = events["Muon_charge"]

    order = ak.argsort(pt, ascending=False)

    pt = pt[order]
    eta = eta[order]
    phi = phi[order]
    charge = charge[order]

    muons = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "charge": charge,
        }
    )

    pairs = ak.combinations(muons, 2, fields=["a", "b"])
    os_pairs = pairs[pairs.a.charge * pairs.b.charge < 0]

    has_os_pair = ak.num(os_pairs) > 0

    pt_sum = os_pairs.a.pt + os_pairs.b.pt

    best_index = ak.argmax(pt_sum, axis=1, keepdims=True)
    best_pair = os_pairs[best_index]

    best_pair = ak.firsts(best_pair)

    pt1 = ak.to_numpy(ak.fill_none(best_pair.a.pt, np.nan))
    eta1 = ak.to_numpy(ak.fill_none(best_pair.a.eta, np.nan))
    phi1 = ak.to_numpy(ak.fill_none(best_pair.a.phi, np.nan))

    pt2 = ak.to_numpy(ak.fill_none(best_pair.b.pt, np.nan))
    eta2 = ak.to_numpy(ak.fill_none(best_pair.b.eta, np.nan))
    phi2 = ak.to_numpy(ak.fill_none(best_pair.b.phi, np.nan))

    mass = compute_pair_mass(pt1, eta1, phi1, pt2, eta2, phi2)

    has_os_pair = ak.to_numpy(has_os_pair)
    mass[~has_os_pair] = np.nan

    return mass.astype(np.float32), has_os_pair.astype(bool)


def process_files(files, step_size):
    X_list = []
    mask_list = []
    mass_list = []
    has_os_pair_list = []

    total_chunks = 0
    total_events = 0

    for arrays in uproot.iterate(
        [f + ":Events" for f in files],
        BRANCHES,
        step_size=step_size,
        library="ak",
    ):
        total_chunks += 1

        X, mask = build_feature_array(arrays)
        m_mumu, has_os_pair = compute_leading_os_mumu_mass(arrays)

        X_list.append(X)
        mask_list.append(mask)
        mass_list.append(m_mumu)
        has_os_pair_list.append(has_os_pair)

        total_events += len(X)

        print(
            f"Chunk {total_chunks:04d} | "
            f"events {len(X):8d} | "
            f"total {total_events:10d}"
        )

    X = np.concatenate(X_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    m_mumu = np.concatenate(mass_list, axis=0)
    has_os_pair = np.concatenate(has_os_pair_list, axis=0)

    return X, mask, m_mumu, has_os_pair


def main():
    args = parse_args()

    files = get_root_files(args.input_dir, args.max_files)

    print("Input dir:", args.input_dir)
    print("Number of ROOT files:", len(files))
    print("Output:", args.output)

    X, mask, m_mumu, has_os_pair = process_files(files, args.step_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        X=X,
        mask=mask,
        m_mumu=m_mumu,
        has_os_pair=has_os_pair,
        features=FEATURES,
    )

    print("\nSaved:", args.output)
    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("m_mumu shape:", m_mumu.shape)
    print("Events with OS pair:", int(has_os_pair.sum()))
    print("Total events:", len(has_os_pair))


if __name__ == "__main__":
    main()