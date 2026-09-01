## download_real.py
## Real SingleMuon ROOT 파일을 다운로드하는 코드입니다.
##
## 실행 예:
## python download_real.py \
##     --url-files ../dataset/SingleMuon_url1.txt ../dataset/SingleMuon_url2.txt ../dataset/SingleMuon_url3.txt ../dataset/SingleMuon_url4.txt ../dataset/SingleMuon_url5.txt \
##     --output-dir ../data/Real/SingleMuon

import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url-files",
        nargs="+",
        required=True,
        help="Text files containing ROOT file URLs.",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/Real/SingleMuon",
        help="Directory where ROOT files will be saved.",
    )
    return parser.parse_args()


def read_urls(url_files):
    urls = []

    for url_file in url_files:
        with open(url_file, "r") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                if line.startswith("#"):
                    continue

                urls.append(line)

    return urls


def download_file(url, output_dir):
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Already exists, skip: {output_path}")
        return

    print(f"Downloading: {url}")
    print(f"Output: {output_path}")

    if url.startswith("root://"):
        command = [
            "xrdcp",
            "-f",
            url,
            output_path,
        ]
    else:
        command = [
            "wget",
            "-O",
            output_path,
            url,
        ]

    subprocess.run(command, check=True)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    urls = read_urls(args.url_files)

    print("URL files:")
    for url_file in args.url_files:
        print(" ", url_file)

    print("Output dir:", args.output_dir)
    print("Number of files:", len(urls))

    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{len(urls)}]")
        download_file(url, args.output_dir)

    print("\nDownload finished.")


if __name__ == "__main__":
    main()