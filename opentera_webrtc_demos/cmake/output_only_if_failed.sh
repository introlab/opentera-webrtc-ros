#!/bin/bash

output=$("$@" 2>&1)
return_code=$?

if [ $return_code -ne 0 ]; then
    echo -e "$output" >&2
fi

exit $return_code
