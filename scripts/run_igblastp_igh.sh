#!/bin/bash
FILENAME="$1";


# From https://ncbi.github.io/igblast/cook/examples.html :
# "The parameters are similar to those of igblastn except it does not search germline D database or germline J database. The optional file is not needed."

# We removed these parts of the igblastn command:
# -germline_db_D human_gl_D -germline_db_J human_gl_J -num_alignments_D 1 -num_alignments_J 1 \
# -auxiliary_data human_gl.aux \
# -germline_db_D_seqidlist human_gl_igh_seqidlist \
# -germline_db_J_seqidlist human_gl_igh_seqidlist \

./igblastp \
-ig_seqtype Ig -germline_db_V human_gl_V \
-organism human -domain_system imgt -num_threads 1 -num_alignments_V 1 \
-outfmt "7 qseqid sseqid" \
-germline_db_V_seqidlist human_gl_igh_seqidlist \
-num_alignments 1 \
-query "${FILENAME}" > "${FILENAME}.parse.txt";
