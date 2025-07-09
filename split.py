#!/usr/bin/env python
#=========================================================================
# This is OPEN SOURCE SOFTWARE governed by the Gnu General Public
# License (GPL) version 3, as described at www.opensource.org.
# Copyright (C)2021 William H. Majoros <bmajoros@alumni.duke.edu>
#=========================================================================
from __future__ import (absolute_import, division, print_function, 
   unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
   chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
# The above imports should allow this program to run in both Python 2 and
# Python 3.  You might need to update your version of module "future".
import sys
import ProgramName
import gzip

#=========================================================================
# main()
#=========================================================================
if(len(sys.argv)!=8):
    exit(ProgramName.get()+" <all-regions> <num-train> <num-valid> <num-test> <out:train> <out:valid> <out:test>\n")
(infile,numTrain,numValid,numTest,outTrain,outValid,outTest)=sys.argv[1:]
numTrain=int(numTrain); numValid=int(numValid); numTest=int(numTest)

TRAIN=gzip.open(outTrain,"wt")
VALID=gzip.open(outValid,"wt")
TEST=gzip.open(outTest,"wt")
with open(infile,"rt") as IN:
    lines=IN.readlines()
    header=lines[0].rstrip()
    print(header,file=TRAIN)
    print(header,file=VALID)
    print(header,file=TEST)
    train=lines[1:numTrain]
    valid=lines[numTrain:numTrain+numValid]
    test=lines[numTrain+numValid:]
    print("".join(train),file=TRAIN)
    print("".join(valid),file=VALID)
    print("".join(test),file=TEST)
TRAIN.close(); VALID.close(); TEST.close()



