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

MIN_LEN=10

KEEP=["gene_id","gene_name","strand","transcript_id","gene_region_identifier",
      "amino_sequence","theta","theta_upper","theta_lower",
      "missense","lof","total_variants",
      "homologous_superfamily","homologous_superfamily_name",
      "topology","topology_name","architecture","architecture_name",
      "class","class_name"]
newHeader="\t".join(KEEP)

def indexHeader(header):
    index=dict()
    i=0
    for term in header:
        index[term]=i
        i+=1
    return index

def subsetLine(fields,index):
    kept=[]
    for term in KEEP:
        #print(term,index[term],fields[index[term]],sep="\t")
        kept.append(fields[index[term]])
    return kept

#=========================================================================
# main()
#=========================================================================
if(len(sys.argv)!=3):
    exit(ProgramName.get()+" <infile> <outfile>\n")
(infile,outfile)=sys.argv[1:]

index=None
OUT=open(outfile,"wt")
print(newHeader,file=OUT)
with open(infile,"rt") as IN:
    header=IN.readline().rstrip().split("\t")
    index=indexHeader(header)
    for line in IN:
        line=line.rstrip().split("\t")
        if(line[index["theta"]]=="NA"): continue
        #print(len(line[index["amino_sequence"]]),MIN_LEN)
        if(len(line[index["amino_sequence"]])<MIN_LEN): continue
        keep=subsetLine(line,index)
        print("\t".join(keep),file=OUT)
    
    
    


