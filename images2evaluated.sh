#!/bin/bash

set -e -o pipefail

[ $# != 0 ]
images=("$@")

log=tmp/images2evaluated.log
zip_in=tmp/images2evaluated.input.zip
json_out=tmp/images2evaluated.output.json

rm -f "$zip_in"
zip "$zip_in" "${images[@]}"

args_curl=(
	-X POST
	-H "X-Style: abstraktsiya"
	-F zip_in=@"$zip_in"
	http://localhost:5000/evaluate
	#http://localhost:5000/forward_batch
)

curl "${args_curl[@]}" >"$json_out"
