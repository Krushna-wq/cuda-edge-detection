#!/usr/bin/env bash

set -euo pipefail

make build

if [[ $# -eq 0 ]]; then
	mkdir -p output
	set -- input/test.pgm output/edges.pgm
elif [[ $# -eq 6 && "$1" == "--input" && "$3" == "--output" && "$5" == "--count" ]]; then
	mkdir -p "$4"
fi

exec ./edge_detection "$@"
