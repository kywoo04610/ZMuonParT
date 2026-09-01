## truth.py는 reco muon과 truth muon의 매칭을 확인하는 코드입니다.
## 명령어는 python truth.py --sample <sample_name> --n-events <number_of_events> 형태로 실행할 수 있습니다.

import awkward as ak
import numpy as np


INVALID = -1
DEFAULT = -999999


def safe_take(gen_field, indices, default=DEFAULT):
    safe_indices = ak.where(indices >= 0, indices, 0)
    values = gen_field[safe_indices]
    return ak.where(indices >= 0, values, default)


def get_truth_info(arrays, max_depth=30):
    gen_idx = arrays["Muon_genPartIdx"]

    gen_pdgid = arrays["GenPart_pdgId"]
    gen_mother_idx = arrays["GenPart_genPartIdxMother"]

    matched_pdgid = safe_take(gen_pdgid, gen_idx)
    is_gen_matched = gen_idx >= 0
    is_matched_muon = is_gen_matched & (np.abs(matched_pdgid) == 13)

    current = ak.where(is_matched_muon, gen_idx, INVALID)
    z_index = ak.full_like(gen_idx, INVALID)

    for _ in range(max_depth):
        active = current >= 0

        current_pdgid = safe_take(gen_pdgid, current)
        found_z = active & (current_pdgid == 23)

        z_index = ak.where((z_index < 0) & found_z, current, z_index)

        mother = safe_take(gen_mother_idx, current, default=INVALID)
        current = ak.where(active & (~found_z), mother, INVALID)

    has_z_ancestor = z_index >= 0

    return ak.zip(
        {
            "gen_idx": gen_idx,
            "matched_pdgid": matched_pdgid,
            "is_gen_matched": is_gen_matched,
            "is_matched_muon": is_matched_muon,
            "z_index": z_index,
            "has_z_ancestor": has_z_ancestor,
        }
    )


def find_z_ancestor_indices(arrays, max_depth=30):
    return get_truth_info(arrays, max_depth=max_depth).z_index


def has_truth_z_mumu_pair(arrays, max_depth=30):
    truth = get_truth_info(arrays, max_depth=max_depth)

    muons = ak.zip(
        {
            "charge": arrays["Muon_charge"],
            "z_index": truth.z_index,
            "is_good_z_muon": truth.is_matched_muon & truth.has_z_ancestor,
        }
    )

    pairs = ak.combinations(muons, 2, fields=["mu1", "mu2"])

    opposite_charge = pairs.mu1.charge * pairs.mu2.charge < 0
    both_z_muons = pairs.mu1.is_good_z_muon & pairs.mu2.is_good_z_muon
    same_z = pairs.mu1.z_index == pairs.mu2.z_index

    return ak.any(opposite_charge & both_z_muons & same_z, axis=1)


def find_z_ancestor_indices_for_event(
    muon_gen_indices,
    gen_pdgids,
    gen_mother_indices,
    max_depth=30,
):
    z_indices = []

    for gen_idx in muon_gen_indices:
        if gen_idx < 0:
            z_indices.append(-1)
            continue

        if abs(gen_pdgids[gen_idx]) != 13:
            z_indices.append(-1)
            continue

        current = gen_idx
        z_index = -1

        for _ in range(max_depth):
            if current < 0:
                break

            if gen_pdgids[current] == 23:
                z_index = current
                break

            current = gen_mother_indices[current]

        z_indices.append(z_index)

    return z_indices