#!/bin/bash

set -euxo pipefail

patient_id=$1
repertoire_id=$2
fname=$3

model_header='/data/maxim/briney_healthy/model_header.txt'
table_name='ireceptor_data.briney_healthy_sequences'
patient_id_column_name='patient_id'
repertoire_id_column_name='repertoire_id'

# confirm header matches
f=$(mktemp)
head -n 1 $fname > $f
diff -q $f $model_header
rm $f

# load $fname into $table_name in a transaction. we list out column names because patient_id not included.
# set any null $patient_id_column_name values to $patient_id
# commit transaction

f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

begin;

\copy $table_name ( "seq_id",
"uid",
"chain",
"productive",
"v_full",
"v_gene",
"d_full",
"d_gene",
"j_full",
"j_gene",
"cdr3_length",
"cdr3_nt",
"cdr3_aa",
"v_start",
"vdj_nt",
"vj_aa",
"var_muts_nt",
"var_muts_aa",
"var_identity_nt",
"var_identity_aa",
"var_mut_count_nt",
"var_mut_count_aa",
"var_ins",
"var_del",
"isotype",
"raw_input")
from '$fname' with (FORMAT CSV, HEADER);

update $table_name set $patient_id_column_name = '$patient_id' where $patient_id_column_name is null;
update $table_name set $repertoire_id_column_name = '$repertoire_id' where $repertoire_id_column_name is null;

commit;

HERE
psql boydlab --echo-all --file $f
rm $f
