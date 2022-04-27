#!/bin/bash

set -e -o pipefail

[ $# = 0 ]

args=(
	-X POST
       	-H 'Content-Type: application/json' 
	--data @-
	http://localhost:5000/
)
curl "${args[@]}" | display -
