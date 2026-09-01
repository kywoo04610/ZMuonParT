## check_root.py는 ROOT 파일의 브랜치를 확인하는 코드입니다.
## 명령어는 python check_root.py --sample <sample_name> 형태로 실행할 수 있습니다.
import argparse
import glob
import os

import uproot

from config import SAMPLES, TREE_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect ROOT branches."
    )

    parser.add_argument(
        "--sample",
        required=True,
        choices=SAMPLES.keys(),
        help="Sample name",
    )

    return parser.parse_args()


def get_first_root_file(sample):

    data_dir = SAMPLES[sample]["data_dir"]

    root_files = sorted(
        glob.glob(os.path.join(data_dir, "*.root"))
    )

    if len(root_files) == 0:
        raise FileNotFoundError(
            f"No ROOT files found in {data_dir}"
        )

    return root_files[0]


def main():

    args = parse_args()

    root_file = get_first_root_file(args.sample)

    print("=" * 70)
    print("Sample :", args.sample)
    print("ROOT file :", root_file)
    print("=" * 70)

    file = uproot.open(root_file)
    events = file[TREE_NAME]

    branches = events.keys()

    print(f"\nTotal branches : {len(branches)}")

    print("\n================ Muon =================")
    for branch in branches:
        if branch.startswith("Muon"):
            print(branch)

    print("\n================ GenPart =================")
    for branch in branches:
        if branch.startswith("GenPart"):
            print(branch)

    print("\n================ gen =================")
    for branch in branches:
        if "gen" in branch.lower():
            print(branch)

    print("\n================ mother =================")
    for branch in branches:
        if "mother" in branch.lower():
            print(branch)

    print("\n================ status =================")
    for branch in branches:
        if "status" in branch.lower():
            print(branch)


if __name__ == "__main__":
    main()