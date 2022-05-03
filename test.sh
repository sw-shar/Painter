#!/bin/bash

set -ex -o pipefail

[ $# = 0 ]

./curl-json.sh tmp/moskva.jpg 200

./images2evaluated.sh tmp/moskva.jpg styleimages/abstraktsiya.jpg
less tmp/images2evaluated.output.json
