#!/bin/sh
docker run --rm --name "$(whoami)_nnlo" -it -v ${PWD}:/aitp -v /Users/wulff/cern/ai/data:/data aitp bash