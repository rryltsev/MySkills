#!/usr/bin/env bash
# Usage: bash enumerate.sh fresh-output-dir
set -eu
set -o pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: bash enumerate.sh fresh-output-dir" >&2
    exit 2
fi
source_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -- "$1"
cd -- "$1"
cp "$source_dir/lat.in" .
cp "$source_dir/enumerate.sh" job.sh
sha256sum "$(command -v genstr)" "$(command -v corrdump)" > binaries.sha256
genstr -l=lat.in -n=12 -sig=14 > enum.out
corrdump -l=lat.in -2=1.90 -3=1.90 -clus -sig=14
corrdump -l=lat.in -c -cf=clusters.out -s=enum.out -sig=14 > correlations.out
tar -czf enum.tar.gz enum.out lat.in clusters.out sym.out correlations.out binaries.sha256 job.sh
