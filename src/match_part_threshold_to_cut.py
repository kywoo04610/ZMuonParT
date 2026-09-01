## match_part_threshold_to_cut.py
##
## Cut-based와 동일한 이벤트 수를 선택하도록
## Particle Transformer score threshold를 계산합니다.
##
## 실행 예:
##
## python match_part_threshold_to_cut.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --target-events 4545858
##
## 또는
##
## python match_part_threshold_to_cut.py \
##     --input ../processed/real_scores_particle_transformer_v4_fsr.npz \
##     --target-fraction 0.03

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../processed/real_scores_particle_transformer_v4_fsr.npz",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--target-events",
        type=int,
        help="Number of events to keep.",
    )

    group.add_argument(
        "--target-fraction",
        type=float,
        help="Fraction of all events to keep (0~1).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    data = np.load(args.input)

    score = data["score"]

    total_events = len(score)

    if args.target_events is not None:

        target_events = args.target_events

    else:

        target_events = int(total_events * args.target_fraction)

    if target_events <= 0:
        raise ValueError("target_events must be positive.")

    if target_events >= total_events:
        raise ValueError("target_events must be smaller than total events.")

    print()

    print("Input:", args.input)
    print("Total events:", total_events)
    print("Target events:", target_events)

    # kth largest score
    threshold = np.partition(score, -target_events)[-target_events]

    selected = score >= threshold

    n_selected = int(np.sum(selected))

    print()
    print("Matched threshold:", threshold)
    print("Selected events:", n_selected)
    print("Difference:", n_selected - target_events)

    print()
    print("Selection fraction:",
          n_selected / total_events)

    print()
    print("Suggested command")
    print("-----------------")

    print(
        "python analysis_real_mass_fsr.py "
        f"--input {args.input} "
        f"--thresholds {threshold:.8f}"
    )

    print()

    print(
        "python fit_real_mass_fsr_v2.py "
        f"--input {args.input} "
        f"--thresholds {threshold:.8f}"
    )


if __name__ == "__main__":
    main()