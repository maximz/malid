#!/bin/bash
# Written by Krish Roskin

FILENAME="$1";

./igblastn \
-ig_seqtype Ig -germline_db_V human_gl_V -germline_db_D human_gl_D -germline_db_J human_gl_J \
-organism human -domain_system imgt -num_threads 1 -num_alignments_V 1 -num_alignments_D 1 -num_alignments_J 1 \
-auxiliary_data human_gl.aux -outfmt "3" \
-germline_db_V_seqidlist human_gl_igh_seqidlist \
-germline_db_D_seqidlist human_gl_igh_seqidlist \
-germline_db_J_seqidlist human_gl_igh_seqidlist \
-num_alignments 1 \
-query "${FILENAME}" > "${FILENAME}.parse.txt";
