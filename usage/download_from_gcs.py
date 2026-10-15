"""Download datasets from a GCP bucket to this machine.

Files are copied into ~/Datasets (beside ~/repos) by default, named after
the last part of the bucket path, e.g.

    gs://aircheck-ml/pgk2            ->  ~/Datasets/pgk2/...
    gs://aircheck-ml/pgk2/train.csv  ->  ~/Datasets/train.csv

Uses the `gcloud storage` CLI, so the VM's service account (or your
`gcloud auth login` account) must have read access to the bucket.

Examples:
    # See what is in a bucket folder
    python download_from_gcs.py gs://aircheck-ml/pgk2 --list

    # Download a folder (only new or changed files are copied on re-runs)
    python download_from_gcs.py gs://aircheck-ml/pgk2

    # Download a single file
    python download_from_gcs.py gs://aircheck-ml/pgk2/train.csv

    # Show what would be copied without copying
    python download_from_gcs.py gs://aircheck-ml/pgk2 --dry-run
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DEST = Path.home() / "Datasets"


def run(cmd):
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def is_folder(gcs_path):
    """True if the path is a bucket or folder rather than a single object."""
    if gcs_path.endswith("/"):
        return True
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"Cannot access {gcs_path}:\n{result.stderr.strip()}")
    # A single object lists as exactly itself; a folder lists its contents.
    return result.stdout.strip() != gcs_path


def local_path_for(gcs_path, dest_root):
    """gs://bucket/a/b -> <dest_root>/b"""
    return Path(dest_root) / gcs_path.rstrip("/").rsplit("/", 1)[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Download files from a GCP bucket into ~/Datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:" + __doc__.split("Examples:")[1],
    )
    parser.add_argument("source", help="gs:// path to a bucket, folder or file")
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Local root folder (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the source and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be copied"
    )
    args = parser.parse_args()

    if shutil.which("gcloud") is None:
        sys.exit("gcloud CLI not found. Install the Google Cloud SDK first.")
    if not args.source.startswith("gs://"):
        sys.exit("Source must start with gs://")

    if args.list:
        run(["gcloud", "storage", "ls", "-l", "-r", args.source])
        return

    target = local_path_for(args.source, args.dest)

    if is_folder(args.source):
        # rsync only copies new or changed files, so re-runs are cheap.
        cmd = ["gcloud", "storage", "rsync", "-r"]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd += [args.source.rstrip("/"), str(target)]
        folder_to_create = target
    else:
        if args.dry_run:
            print(f"[dry-run] would copy {args.source} -> {target}")
            return
        cmd = ["gcloud", "storage", "cp", args.source, str(target)]
        folder_to_create = target.parent

    if not args.dry_run:
        folder_to_create.mkdir(parents=True, exist_ok=True)

    run(cmd)
    if not args.dry_run:
        print(f"\nDone. Files are in: {target}")


if __name__ == "__main__":
    main()
