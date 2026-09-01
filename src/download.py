## download.py는 xrdcp를 이용해 ROOT 파일을 다운로드하는 코드입니다.
## 명령어는 python download.py --sample <sample_name> --max-files <max_files> 형태로 실행할 수 있습니다.

import argparse
import glob
import os
import subprocess
from config import SAMPLES


def parse_args():
    parser = argparse.ArgumentParser(description="Download ROOT files with xrdcp.")
    parser.add_argument(
        "--sample",
        required=True,
        choices=SAMPLES.keys(),
        help="Sample name to download.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of ROOT files to download. If omitted, download all.",
    )
    return parser.parse_args()


def read_urls(sample):
    txt_files = sorted(glob.glob(SAMPLES[sample]["url_pattern"]))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No URL text files found for sample: {sample}")

    urls = []

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            for line in f:
                url = line.strip()

                if url == "":
                    continue

                if not url.startswith("root://"):
                    continue

                urls.append(url)

    return urls


def download_file(url, output_dir):
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"[SKIP] {filename} already exists")
        return

    print(f"[DOWNLOAD] {filename}")

    command = ["xrdcp", url, output_path]
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"[FAILED] {filename}")
    else:
        print(f"[DONE] {filename}")


def main():
    args = parse_args()

    sample = args.sample
    output_dir = SAMPLES[sample]["data_dir"]

    os.makedirs(output_dir, exist_ok=True)

    urls = read_urls(sample)

    if args.max_files is not None:
        urls = urls[: args.max_files]

    print("Sample:", sample)
    print("Number of files to download:", len(urls))
    print("Output directory:", output_dir)

    n_files = len(urls)

    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{n_files}]")

        download_file(
            url=url,
            output_dir=output_dir,
        )

    print("\n========================")
    print("Download completed.")
    print("Sample :", sample)
    print("Files  :", n_files)
    print("Output :", output_dir)
    print("========================")


if __name__ == "__main__":
    main()