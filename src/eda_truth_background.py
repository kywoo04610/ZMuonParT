## eda_truth_background.py는 reco preselection을 통과했지만 truth Z->mumu 라벨을 실패한 이벤트를 분석하는 코드입니다.
## 명령어는 python eda_truth_background.py --sample <sample_name> --max-files <number_of_files> 형태로 실행할 수 있습니다.

import argparse
import glob
import os
from collections import Counter

import awkward as ak
import uproot

from config import SAMPLES, TREE_NAME, STEP_SIZE
from truth import has_truth_z_mumu_pair, find_z_ancestor_indices


BRANCHES = [
    "nMuon",
    "Muon_charge",
    "Muon_genPartIdx",
    "GenPart_pdgId",
    "GenPart_genPartIdxMother",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study events that pass reco preselection but fail truth Z->mumu label."
    )
    parser.add_argument("--sample", default="DYJetsToLL", choices=SAMPLES.keys())
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def get_root_files(sample, max_files=None):
    data_dir = SAMPLES[sample]["data_dir"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.root")))

    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")

    if max_files is not None:
        files = files[:max_files]

    return files


def has_opposite_charge_pair(charges):
    pairs = ak.combinations(charges, 2, fields=["q1", "q2"])
    opposite = pairs.q1 * pairs.q2 < 0
    return ak.any(opposite, axis=1)


def classify_truth_failure_event(charges, gen_indices, z_indices):
    """
    Classify why this event is not truth Z->mumu.
    This is for interpretation only.
    """

    if len(charges) < 2:
        return "less_than_2_muons"

    has_os_pair = False
    has_same_z_pair = False
    has_any_z_muon = False
    has_unmatched = False
    has_matched_non_z = False

    for i in range(len(charges)):
        if gen_indices[i] < 0:
            has_unmatched = True
        elif z_indices[i] < 0:
            has_matched_non_z = True
        else:
            has_any_z_muon = True

        for j in range(i + 1, len(charges)):
            if charges[i] * charges[j] < 0:
                has_os_pair = True

                if z_indices[i] >= 0 and z_indices[i] == z_indices[j]:
                    has_same_z_pair = True

    if has_same_z_pair:
        return "should_be_truth_z"

    if has_any_z_muon and has_os_pair:
        return "z_muon_exists_but_no_same_z_os_pair"

    if has_matched_non_z:
        return "matched_muons_but_no_z_ancestor"

    if has_unmatched:
        return "unmatched_muons"

    if not has_os_pair:
        return "no_opposite_charge_pair"

    return "other"


def main():
    args = parse_args()
    root_files = get_root_files(args.sample, args.max_files)

    total_events = 0
    selected_events = 0
    truth_z_events = 0
    truth_non_z_events = 0

    reason_counter = Counter()
    nmuon_counter = Counter()
    gen_pdgid_counter = Counter()
    mother_pdgid_counter = Counter()

    print("Sample:", args.sample)
    print("Number of ROOT files:", len(root_files))

    for root_file in root_files:
        print("\nProcessing:", root_file)

        file = uproot.open(root_file)
        events = file[TREE_NAME]

        for arrays in events.iterate(BRANCHES, step_size=STEP_SIZE):
            total_events += len(arrays["nMuon"])

            reco_mask = has_opposite_charge_pair(arrays["Muon_charge"])
            selected = arrays[reco_mask]

            if len(selected["nMuon"]) == 0:
                continue

            truth_label = has_truth_z_mumu_pair(selected)
            z_indices = find_z_ancestor_indices(selected)

            truth_z = selected[truth_label]
            truth_non_z = selected[~truth_label]
            z_non_z = z_indices[~truth_label]

            selected_events += len(selected["nMuon"])
            truth_z_events += len(truth_z["nMuon"])
            truth_non_z_events += len(truth_non_z["nMuon"])

            for n in ak.to_list(truth_non_z["nMuon"]):
                nmuon_counter[int(n)] += 1

            for iev in range(len(truth_non_z["nMuon"])):
                charges = truth_non_z["Muon_charge"][iev]
                gen_indices = truth_non_z["Muon_genPartIdx"][iev]
                z_event_indices = z_non_z[iev]

                reason = classify_truth_failure_event(
                    charges,
                    gen_indices,
                    z_event_indices,
                )
                reason_counter[reason] += 1

                for gen_idx in gen_indices:
                    if gen_idx < 0:
                        gen_pdgid_counter["unmatched"] += 1
                        mother_pdgid_counter["unmatched"] += 1
                        continue

                    gen_pdgid = truth_non_z["GenPart_pdgId"][iev][gen_idx]
                    mother_idx = truth_non_z["GenPart_genPartIdxMother"][iev][gen_idx]

                    gen_pdgid_counter[int(gen_pdgid)] += 1

                    if mother_idx >= 0:
                        mother_pdgid = truth_non_z["GenPart_pdgId"][iev][mother_idx]
                        mother_pdgid_counter[int(mother_pdgid)] += 1
                    else:
                        mother_pdgid_counter["no_mother"] += 1

    print("\nSummary")
    print("Total events:", total_events)
    print("Reco preselected events:", selected_events)
    print("Truth Z->mumu events:", truth_z_events)
    print("Truth non-Z events:", truth_non_z_events)

    if selected_events > 0:
        print("Truth non-Z fraction:", truth_non_z_events / selected_events)

    print("\nTruth non-Z failure reasons")
    for reason, count in reason_counter.most_common():
        frac = count / truth_non_z_events if truth_non_z_events > 0 else 0
        print(f"{reason}: {count} ({frac:.4f})")

    print("\nMuon multiplicity in truth non-Z events")
    for n, c in sorted(nmuon_counter.items()):
        print(f"nMuon = {n}: {c}")

    print("\nMatched GenPart pdgId counts in truth non-Z events")
    for pdgid, c in gen_pdgid_counter.most_common():
        print(f"{pdgid}: {c}")

    print("\nMother pdgId counts in truth non-Z events")
    for pdgid, c in mother_pdgid_counter.most_common():
        print(f"{pdgid}: {c}")


if __name__ == "__main__":
    main()