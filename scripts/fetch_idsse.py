"""Download the IDSSE open Bundesliga matches and check them against committed checksums.

    python scripts/fetch_idsse.py            # download what is missing, then check every file
    python scripts/fetch_idsse.py --check    # check only; exit 1 naming any missing or changed file

The IDSSE dataset: Bassek, Weber, Rein & Memmert (2025), "An integrated dataset of spatiotemporal
and event data in elite soccer", Scientific Data; CC BY 4.0; hosted on figshare. Seven matches of
the 2022/23 Bundesliga and 2. Bundesliga, about 2.6 GB of XML, into data/external/idsse/<match>/
(gitignored, like all data). The file ids are those kloppy's sportec.load_open_tracking_data uses.

Files come through figshare's API download host (ndownloader.figshare.com), which redirects to the
storage bucket. Standard library only.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data", "external", "idsse")
CHECKSUMS = os.path.join(REPO, "scripts", "idsse_checksums.json")
URL = "https://ndownloader.figshare.com/files/{}"

FILE_IDS = {
    "J03WMX": {"meta": 51643475, "event": 51643493, "tracking": 51643514},
    "J03WN1": {"meta": 51643472, "event": 51643496, "tracking": 51643517},
    "J03WOH": {"meta": 51643478, "event": 51643499, "tracking": 51643520},
    "J03WOY": {"meta": 51643481, "event": 51643502, "tracking": 51643523},
    "J03WPY": {"meta": 51643487, "event": 51643505, "tracking": 51643526},
    "J03WQQ": {"meta": 51643484, "event": 51643508, "tracking": 51643529},
    "J03WR9": {"meta": 51643490, "event": 51643511, "tracking": 51643532},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(match, kind):
    return "{}/{}.xml".format(match, kind)


def download(data_dir=DATA_DIR, say=print):
    for match, files in FILE_IDS.items():
        for kind, file_id in files.items():
            path = os.path.join(data_dir, relative(match, kind))
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            say("downloading {}".format(relative(match, kind)))
            request = urllib.request.Request(URL.format(file_id),
                                             headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(request) as response, open(path + ".part", "wb") as out:
                shutil.copyfileobj(response, out)
            os.replace(path + ".part", path)


def check(data_dir=DATA_DIR):
    """[problem] for every file that is missing or differs from the committed checksum."""
    with open(CHECKSUMS) as handle:
        expected = json.load(handle)
    problems = []
    for name, digest in sorted(expected.items()):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            problems.append("missing: {}".format(name))
        elif sha256(path) != digest:
            problems.append("differs from the committed checksum: {}".format(name))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="check only, download nothing")
    args = parser.parse_args(argv)
    if not args.check:
        download()
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("IDSSE data OK: {} files match the committed checksums".format(
        sum(len(f) for f in FILE_IDS.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
