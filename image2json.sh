#!/bin/bash

set -e -o pipefail

[ $# = 1 ]
image=$1

key=${KEY_IMAGE:-image}

base64 <"$image" | 
	tr '\n' ' ' | 
	perl -nle '
		s/ /\\n/g;
		$text .= $_;

		END {
	       	    print qq({"style": "abstraktsiya", "'$key'": "$text"});
	        }
		'
