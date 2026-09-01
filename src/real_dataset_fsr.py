## real_dataset_fsr.py
## 실제 CMS SingleMuon 데이터를 모델 입력용 npz로 변환하고,
## uncorrected dimuon mass와 FSR corrected dimuon mass를 함께 저장합니다.
##
## 출력:
##   X          : (N, 8, 6)
##   mask       : (N, 8)
##   m_mumu     : leading OS muon pair invariant mass
##   m_mumu_fsr : FSR corrected leading OS muon pair invariant mass
##   has_os_pair
##   features
##
## 실행 예:
## python real_dataset_fsr.py \
##     --input-dir ../data/Real/SingleMuon \
##     --output ../processed/real_singlemuon_dataset_fsr.npz

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
    "Muon_fsrPhotonIdx",
    "FsrPhoton_pt",
    "FsrPhoton_eta",
    "FsrPhoton_phi",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/Real/SingleMuon")
    parser.add_argument("--output", default="../processed/real_singlemuon_dataset_fsr.npz")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--step-size", default="100 MB")
    return parser.parse_args()


def get_root_files(input_dir, max_files=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.root")))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {input_dir}")

    return files


def build_feature_array(events):
    pt = events["Muon_pt"]
    eta = events["Muon_eta"]
    phi = events["Muon_phi"]
    iso = events["Muon_pfRelIso04_all"]
    dxy = events["Muon_dxy"]
    dz = events["Muon_dz"]

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

    mask = ak.to_numpy(
        ak.fill_none(
            ak.pad_none(mask, MAX_MUONS, axis=1, clip=True),
            False,
        )
    )

    return X.astype(np.float32), mask.astype(bool)


def four_vector_from_pt_eta_phi_mass(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return e, px, py, pz


def photon_four_vector(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)
    return e, px, py, pz


def invariant_mass(e, px, py, pz):
    mass2 = e**2 - px**2 - py**2 - pz**2
    mass2 = np.maximum(mass2, 0.0)
    return np.sqrt(mass2)


def get_fsr_photon_value(values, photon_idx):
    """
    Pick per-event FSR photon value using Muon_fsrPhotonIdx.

    values:
        jagged array, e.g. FsrPhoton_pt

    photon_idx:
        one index per event. -1 means no photon.

    Returns:
        numpy array with selected photon value.
        Missing photons are returned as 0.
    """

    photon_idx = ak.fill_none(photon_idx, -1)
    photon_idx = ak.values_astype(photon_idx, np.int64)

    local_idx = ak.local_index(values, axis=1)
    selected = values[local_idx == photon_idx[:, None]]
    selected = ak.firsts(selected)

    return ak.to_numpy(ak.fill_none(selected, 0.0)).astype(np.float32)


def compute_leading_os_mumu_masses(events):
    """
    Compute both uncorrected and FSR-corrected invariant mass
    of the leading opposite-sign muon pair.

    Pair choice:
        Among opposite-sign muon pairs, choose the pair with largest pt1 + pt2.

    This avoids choosing the pair closest to the Z mass.
    """

    pt = events["Muon_pt"]
    eta = events["Muon_eta"]
    phi = events["Muon_phi"]
    charge = events["Muon_charge"]
    fsr_idx = events["Muon_fsrPhotonIdx"]

    fsr_pt = events["FsrPhoton_pt"]
    fsr_eta = events["FsrPhoton_eta"]
    fsr_phi = events["FsrPhoton_phi"]

    order = ak.argsort(pt, ascending=False)

    pt = pt[order]
    eta = eta[order]
    phi = phi[order]
    charge = charge[order]
    fsr_idx = fsr_idx[order]

    muons = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "charge": charge,
            "fsr_idx": fsr_idx,
        }
    )

    pairs = ak.combinations(muons, 2, fields=["a", "b"])
    os_pairs = pairs[pairs.a.charge * pairs.b.charge < 0]

    has_os_pair = ak.num(os_pairs) > 0

    pt_sum = os_pairs.a.pt + os_pairs.b.pt
    best_index = ak.argmax(pt_sum, axis=1, keepdims=True)

    best_pair = os_pairs[best_index]
    best_pair = ak.firsts(best_pair)

    pt1 = ak.to_numpy(ak.fill_none(best_pair.a.pt, np.nan)).astype(np.float32)
    eta1 = ak.to_numpy(ak.fill_none(best_pair.a.eta, np.nan)).astype(np.float32)
    phi1 = ak.to_numpy(ak.fill_none(best_pair.a.phi, np.nan)).astype(np.float32)

    pt2 = ak.to_numpy(ak.fill_none(best_pair.b.pt, np.nan)).astype(np.float32)
    eta2 = ak.to_numpy(ak.fill_none(best_pair.b.eta, np.nan)).astype(np.float32)
    phi2 = ak.to_numpy(ak.fill_none(best_pair.b.phi, np.nan)).astype(np.float32)

    idx1 = ak.fill_none(best_pair.a.fsr_idx, -1)
    idx2 = ak.fill_none(best_pair.b.fsr_idx, -1)

    e1, px1, py1, pz1 = four_vector_from_pt_eta_phi_mass(
        pt1, eta1, phi1, MUON_MASS_GEV
    )
    e2, px2, py2, pz2 = four_vector_from_pt_eta_phi_mass(
        pt2, eta2, phi2, MUON_MASS_GEV
    )

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m_mumu = invariant_mass(e, px, py, pz).astype(np.float32)

    # FSR photon for muon 1
    g1_pt = get_fsr_photon_value(fsr_pt, idx1)
    g1_eta = get_fsr_photon_value(fsr_eta, idx1)
    g1_phi = get_fsr_photon_value(fsr_phi, idx1)

    # FSR photon for muon 2
    g2_pt = get_fsr_photon_value(fsr_pt, idx2)
    g2_eta = get_fsr_photon_value(fsr_eta, idx2)
    g2_phi = get_fsr_photon_value(fsr_phi, idx2)

    eg1, pxg1, pyg1, pzg1 = photon_four_vector(g1_pt, g1_eta, g1_phi)
    eg2, pxg2, pyg2, pzg2 = photon_four_vector(g2_pt, g2_eta, g2_phi)

    idx1_np = ak.to_numpy(ak.fill_none(idx1, -1)).astype(np.int64)
    idx2_np = ak.to_numpy(ak.fill_none(idx2, -1)).astype(np.int64)

    same_fsr = (idx1_np >= 0) & (idx1_np == idx2_np)

    e_fsr = e + eg1 + eg2
    px_fsr = px + pxg1 + pxg2
    py_fsr = py + pyg1 + pyg2
    pz_fsr = pz + pzg1 + pzg2

    # If both muons point to the same FSR photon, subtract one copy.
    e_fsr[same_fsr] -= eg2[same_fsr]
    px_fsr[same_fsr] -= pxg2[same_fsr]
    py_fsr[same_fsr] -= pyg2[same_fsr]
    pz_fsr[same_fsr] -= pzg2[same_fsr]

    m_mumu_fsr = invariant_mass(e_fsr, px_fsr, py_fsr, pz_fsr).astype(np.float32)

    has_os_pair_np = ak.to_numpy(has_os_pair).astype(bool)

    m_mumu[~has_os_pair_np] = np.nan
    m_mumu_fsr[~has_os_pair_np] = np.nan

    return m_mumu, m_mumu_fsr, has_os_pair_np


def process_files(files, step_size):
    X_list = []
    mask_list = []
    mass_list = []
    mass_fsr_list = []
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
        m_mumu, m_mumu_fsr, has_os_pair = compute_leading_os_mumu_masses(arrays)

        X_list.append(X)
        mask_list.append(mask)
        mass_list.append(m_mumu)
        mass_fsr_list.append(m_mumu_fsr)
        has_os_pair_list.append(has_os_pair)

        total_events += len(X)

        print(
            f"Chunk {total_chunks:04d} | "
            f"events {len(X):8d} | "
            f"total {total_events:10d} | "
            f"OS pairs {int(has_os_pair.sum()):8d}"
        )

    X = np.concatenate(X_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    m_mumu = np.concatenate(mass_list, axis=0)
    m_mumu_fsr = np.concatenate(mass_fsr_list, axis=0)
    has_os_pair = np.concatenate(has_os_pair_list, axis=0)

    return X, mask, m_mumu, m_mumu_fsr, has_os_pair


def main():
    args = parse_args()

    files = get_root_files(args.input_dir, args.max_files)

    print("Input dir:", args.input_dir)
    print("Number of ROOT files:", len(files))
    print("Output:", args.output)

    X, mask, m_mumu, m_mumu_fsr, has_os_pair = process_files(
        files,
        args.step_size,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        X=X,
        mask=mask,
        m_mumu=m_mumu,
        m_mumu_fsr=m_mumu_fsr,
        has_os_pair=has_os_pair,
        features=FEATURES,
    )

    print("\nSaved:", args.output)
    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("m_mumu shape:", m_mumu.shape)
    print("m_mumu_fsr shape:", m_mumu_fsr.shape)
    print("Events with OS pair:", int(has_os_pair.sum()))
    print("Total events:", len(has_os_pair))


if __name__ == "__main__":
    main()