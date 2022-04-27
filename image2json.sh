#!/bin/bash

set -e -o pipefail

[ $# = 0 ]

base64 | 
	tr '\n' ' ' | 
	perl -nle '
		s/ /\\n/g;
	       	print qq({"style": "abstraktsiya", "image": "$_"});
		'
