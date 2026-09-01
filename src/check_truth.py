## check_truth.py는 reco muon과 truth muon의 매칭을 확인하는 코드입니다.
## 명령어는 python check_truth.py --sample <sample_name> --n-events <number_of_events> 형태로 실행할 수 있습니다.
import argparse
import glob
import os

import uproot

from config import SAMPLES, TREE_NAME
from truth import has_truth_z_mumu_pair, find_z_ancestor_indices_for_event


BRANCHES = [
    "nMuon",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_charge",
    "Muon_genPartIdx",
    "GenPart_pdgId",
    "GenPart_genPartIdxMother",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Check truth matching for reco muons.")
    parser.add_argument("--sample", required=True, choices=SAMPLES.keys())
    parser.add_argument("--n-events", type=int, default=20)
    return parser.parse_args()


def get_first_root_file(sample):
    data_dir = SAMPLES[sample]["data_dir"]
    root_files = sorted(glob.glob(os.path.join(data_dir, "*.root")))

    if len(root_files) == 0:
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")

    return root_files[0]


def main():
    args = parse_args()

    root_file = get_first_root_file(args.sample)

    print("Sample:", args.sample)
    print("ROOT file:", root_file)

    file = uproot.open(root_file)
    events = file[TREE_NAME]

    arrays = events.arrays(BRANCHES, entry_stop=args.n_events)

    truth_labels = has_truth_z_mumu_pair(arrays)

    for iev in range(len(arrays["nMuon"])):
        print("\n" + "=" * 60)
        print(f"Event {iev}")
        print("nMuon:", arrays["nMuon"][iev])
        print("truth Z->mumu label:", bool(truth_labels[iev]))

        z_indices = find_z_ancestor_indices_for_event(
            arrays["Muon_genPartIdx"][iev],
            arrays["GenPart_pdgId"][iev],
            arrays["GenPart_genPartIdxMother"][iev],
        )

        for imu in range(arrays["nMuon"][iev]):
            gen_idx = arrays["Muon_genPartIdx"][iev][imu]

            print(f"\n  Reco muon {imu}")
            print("    pt:", arrays["Muon_pt"][iev][imu])
            print("    eta:", arrays["Muon_eta"][iev][imu])
            print("    phi:", arrays["Muon_phi"][iev][imu])
            print("    charge:", arrays["Muon_charge"][iev][imu])
            print("    Muon_genPartIdx:", gen_idx)
            print("    Z ancestor index:", z_indices[imu])

            if gen_idx < 0:
                print("    No gen match")
                continue

            gen_pdgid = arrays["GenPart_pdgId"][iev][gen_idx]
            mother_idx = arrays["GenPart_genPartIdxMother"][iev][gen_idx]

            print("    matched GenPart pdgId:", gen_pdgid)
            print("    matched GenPart mother index:", mother_idx)

            if mother_idx < 0:
                print("    No mother particle")
                continue

            mother_pdgid = arrays["GenPart_pdgId"][iev][mother_idx]

            print("    mother pdgId:", mother_pdgid)

            if z_indices[imu] >= 0:
                print("    ==> has Z ancestor")
            elif abs(gen_pdgid) == 13:
                print("    ==> matched to muon, but no Z ancestor found")
            else:
                print("    ==> gen match is not a muon")


if __name__ == "__main__":
    main()