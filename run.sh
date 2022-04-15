#!/bin/bash

set -eu -o pipefail

[ $# = 2 ]
image=$1
container=$2

args=(
        -it
        --rm
	-p 5000:5000
	-v "$PWD":/app
        --name="$container"
        "$image"
        #bash
)
        
set -x
docker run "${args[@]}"
