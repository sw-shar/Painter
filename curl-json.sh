#!/bin/bash

set -e -o pipefail

[ $# = 1 ]
image=$1

args_curl=(
	-X POST
       	-H 'Content-Type: application/json' 
	--data @-
	http://localhost:5000/
)

./image2json.sh <"$image" |
	curl "${args_curl[@]}" |
       	jq -r '.image' |
	base64 -d - | 
	display -
