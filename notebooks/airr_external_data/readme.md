# Order

1. Load and process raw data as below
2. `metadata_per_patient.ipynb`
3. `assign_clone_ids.ipynb`

Then run embeddings for external cohorts using `run_embedding_external_cohorts.ipynb` and `run_embedding_external_cohorts.covid_early_time_points.ipynb` in parent directory. (See main runbook in parent directory.)

Below, we load downloaded external cohort data into Postgres, in a schema well-separated from the rest of our database:

- Create some new tables in a separate schema within postgres
- Import downloaded sequences and patient metadata into those new tables
- Query the metadata to select repertoires of interest
- Export sequences of interest from those repertoires to a fasta file
- Run igblast from our standard internal pipeline, with some small tweaks to get in and out of igblast
- Write igblast output back to another table in this separate schema
- Join the original sequence and metadata tables with the igblast output table to filter and export a combined “mini part table”

TODO: consider SQLite instead in future, since this is a self-contained analysis.

See schema explanation: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

---

# Import Covid AIRR-seq data from iReceptor Gateway download into Postgres

## Prep

```bash
# Create Postgres schema. And be sure to give rights to backup job runner.

$ psql -U postgres
    \c boydlab
    CREATE SCHEMA IF NOT EXISTS ireceptor_data AUTHORIZATION maxim;
    GRANT ALL PRIVILEGES ON SCHEMA ireceptor_data TO boydlab_administrator;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ireceptor_data TO boydlab_administrator;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ireceptor_data TO boydlab_administrator;
    ALTER DEFAULT PRIVILEGES IN SCHEMA ireceptor_data GRANT ALL PRIVILEGES ON TABLES TO boydlab_administrator;
    \dn+
    \q

$ pip install --upgrade pandas csvkit flatten_json
```

## metadata

```bash
# convert json to tsv
cat /data/maxim/airr-covid-19-1-metadata.json | python airr_metadata_to_tsv.py > /data/maxim/airr_covid19_metadata.tsv
cat /data/maxim/vdjserver-metadata.json | python airr_metadata_to_tsv.py > /data/maxim/vdjserver_metadata.tsv

# combine
python -c "import sys, pandas as pd; dfs = [pd.read_csv('/data/maxim/airr_covid19_metadata.tsv', sep='\t'), pd.read_csv('/data/maxim/vdjserver_metadata.tsv', sep='\t')]; df = pd.concat(dfs, axis=0); df.to_csv(sys.stdout, index=None, sep='\t')" > /data/maxim/combined_metadata.tsv

echo 'drop table if exists ireceptor_data.covid_metadata' | psql boydlab --echo-all;

# create table appropriately
# note with --blanks we aren't coercing "none" to NaN in csvsql. let's stick with original text
cat /data/maxim/combined_metadata.tsv | csvsql --no-constraints -i postgresql --db-schema ireceptor_data --tables covid_metadata --blanks | sed 's/DECIMAL/NUMERIC/' | sed 's/VARCHAR/TEXT/' | psql boydlab --echo-all

# load into table. each row is unique.
psql boydlab
    $ \copy ireceptor_data.covid_metadata from '/data/maxim/combined_metadata.tsv' WITH CSV DELIMITER E'\t' HEADER;
    $ select count(repertoire_id) from ireceptor_data.covid_metadata;
    $ select count(distinct (repertoire_id)) from ireceptor_data.covid_metadata;
    $ select count(*) from ireceptor_data.covid_metadata;

```

Metadata: our (Boydlab) study is `study.study_id = PRJNA628125`. See https://docs.google.com/spreadsheets/d/1AlaGc02n7aj6DQossWJv-YVm0i-zrxTaUHW_cerp2gI/edit#gid=667720007

## Import sequence data

### create table

Schema from https://docs.airr-community.org/en/stable/datarep/rearrangements.html (click download as TSV).

```bash
psql boydlab
    $ drop table if exists ireceptor_data.covid_sequences;

cat airr_schema_official.sql | sed 's/TABLENAME/ireceptor_data.covid_sequences/' | psql boydlab --echo-all;
```

### add relationships/indexes

```
$ psql boydlab

ALTER TABLE ireceptor_data.covid_metadata
ADD PRIMARY KEY (repertoire_id);

ALTER TABLE ireceptor_data.covid_sequences
ADD CONSTRAINT fk_metadata
FOREIGN KEY (repertoire_id)
REFERENCES ireceptor_data.covid_metadata (repertoire_id);
```

### load data

Don't just run `\copy` like `\copy ireceptor_data.covid_sequences from '/data/maxim/airr-covid-19-1.tsv' WITH CSV DELIMITER E'\t' HEADER;`. Instead, specify column order, based on header row (note that airr schema is in a different order than the actual airr downloads, and vdjserver has deprecated columns missing): see https://docs.google.com/spreadsheets/d/17XMCXbiIzSXhQB5uEZLekkEKHEqMy70kbBNYP98jG3Y/edit#gid=137476848 :

Run in tmux to import AIRR data:

```bash
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\copy ireceptor_data.covid_sequences ("sequence_id",
    "sequence",
    "sequence_aa",
    "rev_comp",
    "productive",
    "vj_in_frame",
    "stop_codon",
    "complete_vdj",
    "locus",
    "v_call",
    "d_call",
    "d2_call",
    "j_call",
    "c_call",
    "sequence_alignment",
    "sequence_alignment_aa",
    "germline_alignment",
    "germline_alignment_aa",
    "junction",
    "junction_aa",
    "np1",
    "np1_aa",
    "np2",
    "np2_aa",
    "np3",
    "np3_aa",
    "cdr1",
    "cdr1_aa",
    "cdr2",
    "cdr2_aa",
    "cdr3",
    "cdr3_aa",
    "fwr1",
    "fwr1_aa",
    "fwr2",
    "fwr2_aa",
    "fwr3",
    "fwr3_aa",
    "fwr4",
    "fwr4_aa",
    "v_score",
    "v_identity",
    "v_support",
    "v_cigar",
    "d_score",
    "d_identity",
    "d_support",
    "d_cigar",
    "d2_score",
    "d2_identity",
    "d2_support",
    "d2_cigar",
    "j_score",
    "j_identity",
    "j_support",
    "j_cigar",
    "c_score",
    "c_identity",
    "c_support",
    "c_cigar",
    "v_sequence_start",
    "v_sequence_end",
    "v_germline_start",
    "v_germline_end",
    "v_alignment_start",
    "v_alignment_end",
    "d_sequence_start",
    "d_sequence_end",
    "d_germline_start",
    "d_germline_end",
    "d_alignment_start",
    "d_alignment_end",
    "d2_sequence_start",
    "d2_sequence_end",
    "d2_germline_start",
    "d2_germline_end",
    "d2_alignment_start",
    "d2_alignment_end",
    "j_sequence_start",
    "j_sequence_end",
    "j_germline_start",
    "j_germline_end",
    "j_alignment_start",
    "j_alignment_end",
    "cdr1_start",
    "cdr1_end",
    "cdr2_start",
    "cdr2_end",
    "cdr3_start",
    "cdr3_end",
    "fwr1_start",
    "fwr1_end",
    "fwr2_start",
    "fwr2_end",
    "fwr3_start",
    "fwr3_end",
    "fwr4_start",
    "fwr4_end",
    "v_sequence_alignment",
    "v_sequence_alignment_aa",
    "d_sequence_alignment",
    "d_sequence_alignment_aa",
    "d2_sequence_alignment",
    "d2_sequence_alignment_aa",
    "j_sequence_alignment",
    "j_sequence_alignment_aa",
    "c_sequence_alignment",
    "c_sequence_alignment_aa",
    "v_germline_alignment",
    "v_germline_alignment_aa",
    "d_germline_alignment",
    "d_germline_alignment_aa",
    "d2_germline_alignment",
    "d2_germline_alignment_aa",
    "j_germline_alignment",
    "j_germline_alignment_aa",
    "c_germline_alignment",
    "c_germline_alignment_aa",
    "junction_length",
    "junction_aa_length",
    "np1_length",
    "np2_length",
    "np3_length",
    "n1_length",
    "n2_length",
    "n3_length",
    "p3v_length",
    "p5d_length",
    "p3d_length",
    "p5d2_length",
    "p3d2_length",
    "p5j_length",
    "consensus_count",
    "duplicate_count",
    "cell_id",
    "clone_id",
    "rearrangement_id",
    "repertoire_id",
    "sample_processing_id",
    "data_processing_id",
    "germline_database",
    "rearrangement_set_id")
from '/data/maxim/airr-covid-19-1.tsv'
with (FORMAT CSV, DELIMITER E'\t', HEADER);
HERE
psql boydlab --echo-all --file $f
rm $f
```

### eval

```sql
boydlab=> select count(1) from ireceptor_data.covid_sequences;
  count
----------
 27003541
(1 row)

boydlab=> select count(distinct (sequence_id)) from ireceptor_data.covid_sequences;
  count
----------
 27003541
(1 row)
```

Maybe sequence_id only unique when grouped with patient ID (repertoire_id)?

Evaluate the join on `repertoire_id`: how many repertoire_id are defined? All of them, good:

```sql
boydlab=> select count(*) from ireceptor_data.covid_sequences where "repertoire_id" is NULL;
  count
----------
 0
(1 row)
```

How many clone_ids are defined?

```sql
boydlab=> select count(*) from ireceptor_data.covid_sequences where "clone_id" is NULL;
  count
----------
 22281775
(1 row)
```

So we will need to do per-patient clone IDs ourselves.

Choose repertoires of interest in `airr_query.sql`.

### load vdjserver data

NOTE: Montague et al. study from VDJserver is IgG only, so we don't yet use it for evaluation.

VDJserver data has header row printed 56 times throughout the file:

```bash
$ grep -c 'repertoire_id' /data/maxim/vdjserver.tsv
# 56

$ grep -c 'repertoire_id' /data/maxim/airr-covid-19-1.tsv
# 1
```

It also looks like there are 4 more repertoires in the metadata than there are in the vdjserver sequences. That's probably because `vdjserver.tsv (incomplete, expected 18121948 sequences): 17031009 sequences (46.3 GB)`. But the repeated header row definitely a bug.

Let's manually fixing the header row duplication to load vdjserver (Montague et al) data. Let's import the 17031009/18121948 = 94% of sequences that we managed to download.

```bash
$ grep -v 'repertoire_id' /data/maxim/vdjserver.tsv > /data/maxim/vdjserver.no_header_rows.tsv

$ wc -l /data/maxim/vdjserver.tsv /data/maxim/vdjserver.no_header_rows.tsv
  #  17031010 /data/maxim/vdjserver.tsv
  #  17030954 /data/maxim/vdjserver.no_header_rows.tsv
  #  34061964 total
```

Perfect: 56 lines removed.

Now load sequences from this new file, but without header row, in a tmux:

```bash
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\copy ireceptor_data.covid_sequences ("sequence_id",
    "sequence",
    "rev_comp",
    "productive",
    "v_call",
    "d_call",
    "j_call",
    "sequence_alignment",
    "germline_alignment",
    "junction",
    "junction_aa",
    "v_cigar",
    "d_cigar",
    "j_cigar",
    "sequence_aa",
    "vj_in_frame",
    "stop_codon",
    "complete_vdj",
    "locus",
    "d2_call",
    "c_call",
    "sequence_alignment_aa",
    "germline_alignment_aa",
    "np1",
    "np1_aa",
    "np2",
    "np2_aa",
    "np3",
    "np3_aa",
    "cdr1",
    "cdr1_aa",
    "cdr2",
    "cdr2_aa",
    "cdr3",
    "cdr3_aa",
    "fwr1",
    "fwr1_aa",
    "fwr2",
    "fwr2_aa",
    "fwr3",
    "fwr3_aa",
    "fwr4",
    "fwr4_aa",
    "v_score",
    "v_identity",
    "v_support",
    "d_score",
    "d_identity",
    "d_support",
    "d2_score",
    "d2_identity",
    "d2_support",
    "d2_cigar",
    "j_score",
    "j_identity",
    "j_support",
    "c_score",
    "c_identity",
    "c_support",
    "c_cigar",
    "v_sequence_start",
    "v_sequence_end",
    "v_germline_start",
    "v_germline_end",
    "v_alignment_start",
    "v_alignment_end",
    "d_sequence_start",
    "d_sequence_end",
    "d_germline_start",
    "d_germline_end",
    "d_alignment_start",
    "d_alignment_end",
    "d2_sequence_start",
    "d2_sequence_end",
    "d2_germline_start",
    "d2_germline_end",
    "d2_alignment_start",
    "d2_alignment_end",
    "j_sequence_start",
    "j_sequence_end",
    "j_germline_start",
    "j_germline_end",
    "j_alignment_start",
    "j_alignment_end",
    "cdr1_start",
    "cdr1_end",
    "cdr2_start",
    "cdr2_end",
    "cdr3_start",
    "cdr3_end",
    "fwr1_start",
    "fwr1_end",
    "fwr2_start",
    "fwr2_end",
    "fwr3_start",
    "fwr3_end",
    "fwr4_start",
    "fwr4_end",
    "v_sequence_alignment",
    "v_sequence_alignment_aa",
    "d_sequence_alignment",
    "d_sequence_alignment_aa",
    "d2_sequence_alignment",
    "d2_sequence_alignment_aa",
    "j_sequence_alignment",
    "j_sequence_alignment_aa",
    "c_sequence_alignment",
    "c_sequence_alignment_aa",
    "v_germline_alignment",
    "v_germline_alignment_aa",
    "d_germline_alignment",
    "d_germline_alignment_aa",
    "d2_germline_alignment",
    "d2_germline_alignment_aa",
    "j_germline_alignment",
    "j_germline_alignment_aa",
    "c_germline_alignment",
    "c_germline_alignment_aa",
    "junction_length",
    "junction_aa_length",
    "np1_length",
    "np2_length",
    "np3_length",
    "n1_length",
    "n2_length",
    "n3_length",
    "p3v_length",
    "p5d_length",
    "p3d_length",
    "p5d2_length",
    "p3d2_length",
    "p5j_length",
    "consensus_count",
    "duplicate_count",
    "cell_id",
    "clone_id",
    "repertoire_id",
    "sample_processing_id",
    "data_processing_id",
    "germline_database",
    "rearrangement_id")
from '/data/maxim/vdjserver.no_header_rows.tsv'
with (FORMAT CSV, DELIMITER E'\t');
HERE
psql boydlab --echo-all --file $f
rm $f
```

What info is available? Count non-nulls of:

- sequence
- cdr's
- c_call
- productive
- v_identity
- id (created by `alter table` command below)

using this query:

```sql
select
    count(*) as count_all,
    count(seq.repertoire_id) as count_repertoire_id,
    count(seq.sequence) as count_sequence,
    count(seq.sequence_id) as count_sequence_id,
    count(seq.cdr3) as count_cdr3,
    count(seq.cdr3_aa) as count_cdr3_aa,
    count(seq.cdr1) as count_cdr1,
    count(seq.cdr1_aa) as count_cdr1_aa,
    count(seq.c_call) as count_c_call,
    count(seq.productive) as count_productive,
    count(seq.v_identity) as count_v_identity,
    count(seq.locus) as count_locus
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    meta."study.study_id" = 'PRJNA645245';

--  count_all | count_repertoire_id | count_sequence | count_sequence_id | count_cdr3 | count_cdr3_aa | count_cdr1 | count_cdr1_aa | count_c_call | count_productive | count_v_identity | count_locus
-- -----------+---------------------+----------------+-------------------+------------+---------------+------------+---------------+--------------+------------------+------------------+-------------
--   17030948 |            17030948 |       17030948 |          17027948 |   16939425 |      16939340 |   16736834 |      16736337 |            0 |         17009732 |         17030925 |    17030924
-- (1 row)
```

## Run Igblast to re-annotate the sequences and compute SHM levels

See `airr_query.sql` for how we chose these repertoires. Including peak and non-peak here.

No v_mut information available:

```sql
select
    count(*) - count(seq.v_identity) as count_nulls,
    count(seq.v_identity) as count_not_nulls
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    seq.productive = TRUE
    AND seq.locus = 'IGH'
    AND seq.cdr1 is not NULL
    AND seq.cdr1_aa is not NULL
    AND seq.cdr2 is not NULL
    AND seq.cdr2_aa is not NULL
    AND seq.cdr3 is not NULL
    AND seq.cdr3_aa is not NULL
    AND seq.repertoire_id = '5f21e814e1adeb2edc12613d'
    AND meta."study.study_id" = 'PRJNA648677'
    AND seq.c_call LIKE 'IGH%';

--  count_nulls | count_not_nulls
-- -------------+-----------------
--        240012 |               0
-- (1 row)
```

Also, we don't trust the igblast annotations that came from the external studies, since they were run with different igblast versions and reference annotation sets.

Will need to reprocess from `seq.sequence`.

Index on patient_id: `CREATE INDEX ireceptor_data_covid_sequences_repertoire_id ON ireceptor_data.covid_sequences (repertoire_id);`

Create unique sequence ID: `ALTER TABLE ireceptor_data.covid_sequences ADD COLUMN id SERIAL PRIMARY KEY;`

Export the matching filtered ones, including peak (seropositive) time points and earlier time points:

```bash
# use a here-doc to output sql, and also delete all newlines in the sql query so that \COPY is a one-liner
# then pass output through sed to convert `\n` into actual newlines for the fasta
f=$(mktemp)
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\COPY (
    select
        ('>' || seq.id || E'\n' || seq.sequence)
    from
        ireceptor_data.covid_sequences seq
        INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
    where
        seq.productive = TRUE
        AND seq.locus = 'IGH'
        AND seq.sequence is not NULL
        AND (
            (
                meta."study.study_id" = 'PRJNA648677'
                AND seq.c_call LIKE 'IGH%'
            )
            OR (
                meta."study.study_id" = 'PRJNA645245'
            )
        )
) TO STDOUT WITH (FORMAT TEXT);

HERE
psql boydlab --quiet --file $f | sed 's/\\n/\n/g' > /data/maxim/covid_external_all_sequences.fasta
rm $f
```

Copy necessary igblast files:

```bash
mkdir /data/maxim/covid_external_ireceptor
cd /data/maxim/covid_external_ireceptor
cp $HOME/boydlab/pipeline/run_igblast_command.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;
```

Chunk the fasta file. Use gnu split command because we know the fasta we generated always has each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)

```bash
cd /data/maxim/covid_external_ireceptor
mkdir splits
split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" /data/maxim/covid_external_all_sequences.fasta splits/covid_external_all_sequences.fasta.part

wc -l splits/* | grep -v 10000
#       1492 splits/covid_external_all_sequences.fasta.part0000003673.fasta
#   36721492 total

wc -l /data/maxim/covid_external_all_sequences.fasta
#   36721492 covid_external_all_sequences.fasta
```

Run the igblast jobs (we can run this on yellowhammer instead if needed):

```bash
cd /data/maxim/covid_external_ireceptor
find splits -name "covid_external_all_sequences.fasta.part*.fasta" | xargs -I {} -n 1 -P 55 sh -c "./run_igblast_command.sh {}"
```

Monitor progress:

```bash
cd /data/maxim/covid_external_ireceptor
ls splits/*.part*.fasta | wc -l
# 3673
ls splits/*parse* | wc -l
# 3673
```

Import igblast output back to database in a new table:

```bash
echo 'drop table if exists ireceptor_data.covid_sequences_igblast' | psql boydlab --echo-all;

cd ~/boydlab/pipeline
conda deactivate
source ~/boydlab/pyenv/activate
./load_igblast_parse.ireceptor_data.py --locus IgH --schema ireceptor_data covid_sequences_igblast /data/maxim/covid_external_ireceptor/splits/*.parse.txt
# INFO:root:creating parse table covid_sequences_igblast
# INFO:root:processing file /data/maxim/covid_external_ireceptor/splits/covid_external_all_sequences.fasta.part0000000001.fasta.parse.txt
# [...]
# INFO:root:processing file /data/maxim/covid_external_ireceptor/splits/covid_external_all_sequences.fasta.part0000003671.fasta.parse.txt
# INFO:root:processing file /data/maxim/covid_external_ireceptor/splits/covid_external_all_sequences.fasta.part0000003672.fasta.parse.txt
# INFO:root:processing file /data/maxim/covid_external_ireceptor/splits/covid_external_all_sequences.fasta.part0000003673.fasta.parse.txt
# INFO:root:processed 18360746 reads
# INFO:root:found hits for 18360745 reads
# INFO:root:found VH hits for 18360741 reads
# INFO:root:found 0 reverse complement hits
# INFO:root:found V and J segments for 18356236 reads
# INFO:root:0 V-segments were set to none because of name map
# INFO:root:0 D-segments were set to none because of name map
# INFO:root:0 J-segments were set to none because of name map
# INFO:root:looked for a CDR3 in 18356079 reads
# INFO:root:found CDR3 motif in 17631736 reads
```

## Export joined data

Export sequences with our own igblast annotations (i.e. ignore most igblast anontation fields on covid_sequences table, except for `c_call` which is the isotype).

This igblast script does not produce a `v_identity` calculation, so we will derive that from `v_sequence`. However, `v_sequence` is usually produced later in the internal pipeline, by the sort script. To reproduce that ourselves, we'll need all nucleotide segments, as included in the query below.

Recall that the `id` column is an auto-incrementing integer primary key in `ireceptor_data.covid_sequences`. We used it to label sequences in the fasta input to igblast, so we join that column with `id` in `ireceptor_data.covid_sequences_igblast`. On the other hand, `ireceptor_data.covid_sequences` also has a separate `sequence_id` column that comes from the source data -- not to be confused with the custom `id` column we have added here to join the two tables!

```bash
rm -r /data/maxim/covid_external_as_part_tables/
mkdir -p /data/maxim/covid_external_as_part_tables

# get repertoire IDs
# https://stackoverflow.com/a/45364469/130164
for repertoire_id in $(psql boydlab -qAtX -c "select \"repertoire_id\" from ireceptor_data.covid_metadata where (\"study.study_id\" = 'PRJNA645245' or \"study.study_id\" = 'PRJNA648677');")
do
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';
\COPY (
  select
    seq.id,
    seq.repertoire_id,
    seq.sequence_id,
    igblast.v_segment,
    seq.c_call,
    igblast.j_segment,
    igblast.cdr1_seq_aa_q,
    igblast.cdr2_seq_aa_q,
    igblast.cdr3_seq_aa_q,
    igblast.pre_seq_nt_q,
    igblast.fr1_seq_nt_q,
    igblast.cdr1_seq_nt_q,
    igblast.fr2_seq_nt_q,
    igblast.cdr2_seq_nt_q,
    igblast.fr3_seq_nt_q,
    igblast.cdr3_seq_nt_q,
    igblast.post_seq_nt_q,
    igblast.pre_seq_nt_v,
    igblast.fr1_seq_nt_v,
    igblast.cdr1_seq_nt_v,
    igblast.fr2_seq_nt_v,
    igblast.cdr2_seq_nt_v,
    igblast.fr3_seq_nt_v,
    igblast.cdr3_seq_nt_v,
    igblast.post_seq_nt_v,
    igblast.pre_seq_nt_d,
    igblast.fr1_seq_nt_d,
    igblast.cdr1_seq_nt_d,
    igblast.fr2_seq_nt_d,
    igblast.cdr2_seq_nt_d,
    igblast.fr3_seq_nt_d,
    igblast.cdr3_seq_nt_d,
    igblast.post_seq_nt_d,
    igblast.pre_seq_nt_j,
    igblast.fr1_seq_nt_j,
    igblast.cdr1_seq_nt_j,
    igblast.fr2_seq_nt_j,
    igblast.cdr2_seq_nt_j,
    igblast.fr3_seq_nt_j,
    igblast.cdr3_seq_nt_j,
    igblast.post_seq_nt_j,
    igblast.productive,
    seq.consensus_count,
    seq.duplicate_count
  from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
    inner join ireceptor_data.covid_sequences_igblast igblast on igblast.id = seq.id
  where
    seq.locus = 'IGH'
    AND igblast.productive = TRUE
    AND igblast.cdr1_seq_nt_q IS NOT NULL
    AND igblast.cdr1_seq_aa_q IS NOT NULL
    AND igblast.cdr2_seq_nt_q IS NOT NULL
    AND igblast.cdr2_seq_aa_q IS NOT NULL
    AND igblast.cdr3_seq_nt_q IS NOT NULL
    AND igblast.cdr3_seq_aa_q IS NOT NULL
    AND seq.repertoire_id = '$repertoire_id'
    AND (
      (meta."study.study_id" = 'PRJNA648677' AND seq.c_call LIKE 'IGH%')
      OR (meta."study.study_id" = 'PRJNA645245')
    )
)
TO STDOUT WITH (FORMAT CSV, DELIMITER E'\t', HEADER);
HERE
echo "exported.part_table.$repertoire_id.tsv"
psql boydlab --quiet --file $f > /data/maxim/covid_external_as_part_tables/exported.part_table.$repertoire_id.tsv
rm $f
done
```

And metadata:

```bash
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\COPY (
    select * from ireceptor_data.covid_metadata where "study.study_id" IN (
        'PRJNA645245',
        'PRJNA648677'
    )
  )
TO STDOUT WITH (FORMAT CSV, DELIMITER E'\t', HEADER);
HERE
psql boydlab --quiet --file $f > /data/maxim/covid_external_as_part_tables/exported.metadata.tsv
rm $f
```

## Exclude patients with very few sequences

Get number of sequences per file, and filter to files with no sequences (i.e. only header row) -- the awk magic is to remove the spaces that `wc` prepends to each line:

```bash
wc -l /data/maxim/covid_external_as_part_tables/* | awk '{$1=$1; print}' | grep -P '^1 '
# 1 /data/maxim/covid_external_as_part_tables/exported.part_table.6089320523505013226-242ac116-0001-012.tsv
# 1 /data/maxim/covid_external_as_part_tables/exported.part_table.6103966361984373226-242ac116-0001-012.tsv
# 1 /data/maxim/covid_external_as_part_tables/exported.part_table.6119428244249973226-242ac116-0001-012.tsv
# 1 /data/maxim/covid_external_as_part_tables/exported.part_table.6135190774226293226-242ac116-0001-012.tsv
```

Remove part tables and corresponding metadata entries that have no sequences: (TODO just make this ok downstream instead?)

```bash
# fill this in from above:
BROKEN_REPERTOIRE_IDS=("6089320523505013226-242ac116-0001-012" "6103966361984373226-242ac116-0001-012" "6119428244249973226-242ac116-0001-012" "6135190774226293226-242ac116-0001-012")

wc -l /data/maxim/covid_external_as_part_tables/exported.metadata.tsv
# 101

for BROKEN_REPERTOIRE_ID in "${BROKEN_REPERTOIRE_IDS[@]}"
do
rm "/data/maxim/covid_external_as_part_tables/exported.part_table.$BROKEN_REPERTOIRE_ID.tsv"
grep -v "$BROKEN_REPERTOIRE_ID" /data/maxim/covid_external_as_part_tables/exported.metadata.tsv > /data/maxim/covid_external_as_part_tables/exported.metadata.tsv2
mv /data/maxim/covid_external_as_part_tables/exported.metadata.tsv2 /data/maxim/covid_external_as_part_tables/exported.metadata.tsv
done

wc -l /data/maxim/covid_external_as_part_tables/exported.metadata.tsv
# 97

wc -l /data/maxim/covid_external_as_part_tables/* | awk '{$1=$1; print}' | grep -P '^1 '
# silent / empty
```

Next up: export healthy data below, then cluster clones (by nucleotide) within each patient's data (using `metadata_per_patient.ipynb` followed by `assign_clone_ids.ipynb`)

---

# healthy data from https://github.com/briney/grp_paper

Download all data to `maxim@yellowblade:/data/maxim/briney_healthy`:

```bash
mkdir /data/maxim/briney_healthy
cd /data/maxim/briney_healthy
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/316188_HNCHNBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326650_HCGCYBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326737_HNKVKBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326780_HLH7KBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326797_HCGNLBCXY%2BHJLN5BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326907_HLT33BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/327059_HCGTCBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/D103_HCGCLBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
```

Untar:

```bash
cd /data/maxim/briney_healthy
mkdir -p 316188
tar -zxvf 316188_HNCHNBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 316188
mkdir -p 326650
tar -zxvf 326650_HCGCYBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 326650
mkdir -p 326737
tar -zxvf 326737_HNKVKBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 326737
mkdir -p 326780
tar -zxvf 326780_HLH7KBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 326780
mkdir -p 326797
tar -zxvf 326797_HCGNLBCXY+HJLN5BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 326797
mkdir -p 326907
tar -zxvf 326907_HLT33BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 326907
mkdir -p 327059
tar -zxvf 327059_HCGTCBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C 327059
mkdir -p D103
tar -zxvf D103_HCGCLBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz -C D103
```

Review contents of one file - shows that we have isotype info included:

```bash
head -n 1 /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | sed 's/,/\n/g'
# seq_id
# uid
# chain
# productive
# v_full
# v_gene
# d_full
# d_gene
# j_full
# j_gene
# cdr3_length
# cdr3_nt
# cdr3_aa
# v_start
# vdj_nt
# vj_aa
# var_muts_nt
# var_muts_aa
# var_identity_nt
# var_identity_aa
# var_mut_count_nt
# var_mut_count_aa
# var_ins
# var_del
# isotype
# raw_input

cat /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f3 | sort -u
# chain
# heavy
# lambda

cat /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f4 | sort -u
# no
# productive
# yes

cat /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f25 | sort -u
# IgA1
# IgA2
# IgD
# IgE
# IgG1
# IgG2
# IgG3
# IgG4
# IgM
# isotype
# unknown
```

But we still don't have cdr1 and cdr2 in here, so still need to run through igblast.

We see files `1_consensus.txt`, `2_consensus.txt`, etc. Explanation:

> Samples 1-6 are biological replicates. Samples 7-12 and 13-18 are technical replicates of samples 1-6.
>
> Biological replicates refer to different aliquots of peripheral blood monomuclear cells (PBMCs), from which total RNA was separately isolated and processed. Thus, sequences or clonotypes found in multiple biological replicates are assumed to have independently occurred in different cells

So just load 1 through 6.

Is the patient ID in the csv or do we need to insert it as a separate column as above? We decided to insert it separately.

Looks like all the csv headers are the same, so create a table from one of them:

```bash
echo 'drop table if exists ireceptor_data.briney_healthy_sequences' | psql boydlab --echo-all;

head -n 20 /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | csvsql --no-constraints -i postgresql --db-schema ireceptor_data --tables briney_healthy_sequences --blanks | sed 's/DECIMAL/NUMERIC/' | sed 's/VARCHAR/TEXT/' | psql boydlab --echo-all;

echo 'ALTER TABLE ireceptor_data.briney_healthy_sequences ADD patient_id TEXT' | psql boydlab --echo-all;
echo 'ALTER TABLE ireceptor_data.briney_healthy_sequences ADD repertoire_id TEXT' | psql boydlab --echo-all;
```

then load them all in (don't worry, this confirms that the headers are the same):

```bash
head -n 1 /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt > /data/maxim/briney_healthy/model_header.txt;

# ./load_briney.sh PatientID RepertoireID FNAME;

./load_briney.sh D103 D103_1 /data/maxim/briney_healthy/D103/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 326780 326780_1 /data/maxim/briney_healthy/326780/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 326650 326650_1 /data/maxim/briney_healthy/326650/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 326737 326737_1 /data/maxim/briney_healthy/326737/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 327059 327059_1 /data/maxim/briney_healthy/327059/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 326907 326907_1 /data/maxim/briney_healthy/326907/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 316188 316188_1 /data/maxim/briney_healthy/316188/consensus-cdr3nt-90_minimal/1_consensus.txt;
./load_briney.sh 326797 326797_1 /data/maxim/briney_healthy/326797/consensus-cdr3nt-90_minimal/1_consensus.txt;
```

TODO: add other biological replicates.

Sanity check: get counts by each patient:

```sql
boydlab=> select patient_id, count(*) from ireceptor_data.briney_healthy_sequences group by patient_id order by 2 desc;

--  patient_id |  count
-- ------------+---------
--  327059     | 4680470
--  326780     | 3975202
--  326650     | 3226732
--  326797     | 2463151
--  326737     | 1797595
--  D103       |  758380
--  316188     |  202345
--  326907     |    5030
-- (8 rows)
```

Then we will filter to heavy locus with isotype call available (any Ig* isotype). Get counts by each patient:

```sql
boydlab=> select patient_id, count(*) from ireceptor_data.briney_healthy_sequences
where chain = 'heavy'
and isotype like 'Ig%'
group by patient_id order by 2 desc;

--  patient_id |  count
-- ------------+---------
--  327059     | 4678541
--  326780     | 3973615
--  326650     | 3225999
--  326797     | 2462510
--  326737     | 1795880
--  D103       |  757983
--  316188     |  201929
--  326907     |    5025
-- (8 rows)
```

We will run igblast on the `raw_input` column.

Add indices on `patient_id` and `repertoire_id`:

```sql
CREATE INDEX ireceptor_data_briney_healthy_sequences_patient_id ON ireceptor_data.briney_healthy_sequences (patient_id);
CREATE INDEX ireceptor_data_briney_healthy_sequences_repertoire_id ON ireceptor_data.briney_healthy_sequences (repertoire_id);
```

Create unique sequence ID: `ALTER TABLE ireceptor_data.briney_healthy_sequences ADD COLUMN id SERIAL PRIMARY KEY;`

Export the matching filtered ones (pass through sed to convert `\n` into actual newlines)

```bash
mkdir /data/maxim/briney_healthy/igblast

# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
f=$(mktemp)
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\COPY (
  select ('>' || id || E'\n' || raw_input)
  from ireceptor_data.briney_healthy_sequences
  where chain = 'heavy'
  and isotype like 'Ig%'
) TO STDOUT WITH (FORMAT TEXT);
HERE
psql boydlab --quiet --file $f | sed 's/\\n/\n/g' > /data/maxim/briney_healthy/igblast/all_sequences.fasta
rm $f
```

Copy necessary igblast files:

```bash
cd /data/maxim/briney_healthy/igblast
cp $HOME/boydlab/pipeline/run_igblast_command.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;
```

Chunk the fasta file. Use gnu split command because we know the fasta we generated always has each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)

```bash
cd /data/maxim/briney_healthy/igblast
mkdir splits
split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" /data/maxim/briney_healthy/igblast/all_sequences.fasta splits/all_sequences.fasta.part

wc -l splits/* | grep -v 10000
  #     2964 splits/all_sequences.fasta.part0000003421.fasta
  # 34202964 total

wc -l all_sequences.fasta
  # 34202964 all_sequences.fasta
```

Run the igblast jobs:

```bash
cd /data/maxim/briney_healthy/igblast
find splits -name "all_sequences.fasta.part*.fasta" | xargs -I {} -n 1 -P 55 sh -c "./run_igblast_command.sh {}"
```

Monitor progress:

```bash
cd /data/maxim/briney_healthy/igblast
ls splits/*.part*.fasta | wc -l
# 3421
ls splits/*parse* | wc -l
# 3421
```

Import igblast output back to database in a new table:

```bash
echo 'drop table if exists ireceptor_data.briney_healthy_sequences_igblast' | psql boydlab --echo-all;

cd ~/boydlab/pipeline
conda deactivate
source ~/boydlab/pyenv/activate
./load_igblast_parse.ireceptor_data.py --locus IgH --schema ireceptor_data briney_healthy_sequences_igblast /data/maxim/briney_healthy/igblast/splits/*.parse.txt
# INFO:root:creating parse table briney_healthy_sequences_igblast
# INFO:root:processing file /data/maxim/briney_healthy/igblast/splits/all_sequences.fasta.part0000000001.fasta.parse.txt
# [...]
# INFO:root:processing file /data/maxim/briney_healthy/igblast/splits/all_sequences.fasta.part0000003418.fasta.parse.txt
# INFO:root:processing file /data/maxim/briney_healthy/igblast/splits/all_sequences.fasta.part0000003419.fasta.parse.txt
# INFO:root:processing file /data/maxim/briney_healthy/igblast/splits/all_sequences.fasta.part0000003420.fasta.parse.txt
# INFO:root:processed 17100000 parses
# INFO:root:processing file /data/maxim/briney_healthy/igblast/splits/all_sequences.fasta.part0000003421.fasta.parse.txt
# INFO:root:processed 17101482 reads
# INFO:root:found hits for 17101482 reads
# INFO:root:found VH hits for 17101466 reads
# INFO:root:found 17101451 reverse complement hits
# INFO:root:found V and J segments for 17089516 reads
# INFO:root:0 V-segments were set to none because of name map
# INFO:root:0 D-segments were set to none because of name map
# INFO:root:0 J-segments were set to none because of name map
# INFO:root:looked for a CDR3 in 17048013 reads
# INFO:root:found CDR3 motif in 16527689 reads
```

## Export joined data

Export sequences with our own igblast annotations (except isotype call from original data).

This igblast script does not produce a `v_identity` calculation, so we will derive that from `v_sequence`. However, `v_sequence` is usually produced later in the internal pipeline, by the sort script. To reproduce that ourselves, we'll need all nucleotide segments, as included in the query below.

Recall that the `id` column is an auto-incrementing integer primary key in `ireceptor_data.briney_healthy_sequences`. We used it to label sequences in the fasta input to igblast, so we join that column with `id` in `ireceptor_data.briney_healthy_sequences_igblast`.

```bash
rm -r /data/maxim/briney_healthy_as_part_tables/
mkdir -p /data/maxim/briney_healthy_as_part_tables

for repertoire_id in $(psql boydlab -qAtX -c "select distinct(\"repertoire_id\") from ireceptor_data.briney_healthy_sequences;")
do
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';
\COPY (
  select
    seq.id,
    seq.patient_id as participant_label,
    seq.repertoire_id,
    seq.isotype as c_call,
    igblast.v_segment,
    igblast.j_segment,
    igblast.cdr1_seq_aa_q,
    igblast.cdr2_seq_aa_q,
    igblast.cdr3_seq_aa_q,
    igblast.pre_seq_nt_q,
    igblast.fr1_seq_nt_q,
    igblast.cdr1_seq_nt_q,
    igblast.fr2_seq_nt_q,
    igblast.cdr2_seq_nt_q,
    igblast.fr3_seq_nt_q,
    igblast.cdr3_seq_nt_q,
    igblast.post_seq_nt_q,
    igblast.pre_seq_nt_v,
    igblast.fr1_seq_nt_v,
    igblast.cdr1_seq_nt_v,
    igblast.fr2_seq_nt_v,
    igblast.cdr2_seq_nt_v,
    igblast.fr3_seq_nt_v,
    igblast.cdr3_seq_nt_v,
    igblast.post_seq_nt_v,
    igblast.pre_seq_nt_d,
    igblast.fr1_seq_nt_d,
    igblast.cdr1_seq_nt_d,
    igblast.fr2_seq_nt_d,
    igblast.cdr2_seq_nt_d,
    igblast.fr3_seq_nt_d,
    igblast.cdr3_seq_nt_d,
    igblast.post_seq_nt_d,
    igblast.pre_seq_nt_j,
    igblast.fr1_seq_nt_j,
    igblast.cdr1_seq_nt_j,
    igblast.fr2_seq_nt_j,
    igblast.cdr2_seq_nt_j,
    igblast.fr3_seq_nt_j,
    igblast.cdr3_seq_nt_j,
    igblast.post_seq_nt_j,
    igblast.productive
  from
    ireceptor_data.briney_healthy_sequences seq
    inner join ireceptor_data.briney_healthy_sequences_igblast igblast on igblast.id = seq.id
  where
    seq.chain = 'heavy'
    AND igblast.productive = TRUE
    AND igblast.cdr1_seq_nt_q IS NOT NULL
    AND igblast.cdr1_seq_aa_q IS NOT NULL
    AND igblast.cdr2_seq_nt_q IS NOT NULL
    AND igblast.cdr2_seq_aa_q IS NOT NULL
    AND igblast.cdr3_seq_nt_q IS NOT NULL
    AND igblast.cdr3_seq_aa_q IS NOT NULL
    AND seq.repertoire_id = '$repertoire_id'
    AND seq.isotype like 'Ig%'
)
TO STDOUT WITH (FORMAT CSV, DELIMITER E'\t', HEADER);
HERE
psql boydlab --quiet --file $f > /data/maxim/briney_healthy_as_part_tables/exported.part_table.$repertoire_id.tsv
rm $f
done
```

And metadata:

```bash
f=$(mktemp)
# use a here-doc to output sql, and also delete all newlines, so that \COPY is a one-liner
cat << HERE | tr '\n' ' ' >$f
SET SESSION enable_nestloop TO 'off';

\COPY (
    select distinct patient_id as participant_label, repertoire_id, 'Briney' as study_name
    from ireceptor_data.briney_healthy_sequences
  )
TO STDOUT WITH (FORMAT CSV, DELIMITER E'\t', HEADER);
HERE
psql boydlab --quiet --file $f > /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv
rm $f
```

## Exclude patients with very few sequences

Get number of sequences per file:

```bash
$ wc -l /data/maxim/briney_healthy_as_part_tables/*
          9 /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv
     167776 /data/maxim/briney_healthy_as_part_tables/exported.part_table.316188_1.tsv
    3005562 /data/maxim/briney_healthy_as_part_tables/exported.part_table.326650_1.tsv
    1497207 /data/maxim/briney_healthy_as_part_tables/exported.part_table.326737_1.tsv
    3498946 /data/maxim/briney_healthy_as_part_tables/exported.part_table.326780_1.tsv
    2233483 /data/maxim/briney_healthy_as_part_tables/exported.part_table.326797_1.tsv
       4576 /data/maxim/briney_healthy_as_part_tables/exported.part_table.326907_1.tsv
    4242002 /data/maxim/briney_healthy_as_part_tables/exported.part_table.327059_1.tsv
     681578 /data/maxim/briney_healthy_as_part_tables/exported.part_table.D103_1.tsv
   15331139 total
```

Remove part tables and corresponding metadata entries that have little or no sequences: (TODO just make this ok downstream instead?)

```bash
# fill this in from above:
BROKEN_REPERTOIRE_IDS=("326907_1")

wc -l /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv
# 9

for BROKEN_REPERTOIRE_ID in "${BROKEN_REPERTOIRE_IDS[@]}"
do
rm "/data/maxim/briney_healthy_as_part_tables/exported.part_table.$BROKEN_REPERTOIRE_ID.tsv"
grep -v "$BROKEN_REPERTOIRE_ID" /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv > /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv2
mv /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv2 /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv
done

wc -l /data/maxim/briney_healthy_as_part_tables/exported.metadata.tsv
# 8
```

Next up: cluster clones (by nucleotide) within each patient's data (using `metadata_per_patient.ipynb` followed by `assign_clone_ids.ipynb`)

---


# Shomuradova (Covid19 TCRB)

```bash
# Cut on repertoire_id column 150
head -n 1 data/external_cohorts/raw_data/shomuradova/airr-covid-19.tsv | cut -f150 # repertoire_id

# save all specimen labels
cut -f150 data/external_cohorts/raw_data/shomuradova/airr-covid-19.tsv | sort -u | grep -v repertoire_id > data/external_cohorts/raw_data/shomuradova/specimen_labels.txt

# cut by specimen label
for specimen_label in $(cat data/external_cohorts/raw_data/shomuradova/specimen_labels.txt)
do
  # first we need to put the header row in each file
  head -n 1 data/external_cohorts/raw_data/shomuradova/airr-covid-19.tsv > "data/external_cohorts/raw_data/shomuradova/split.$specimen_label.tsv"

  # now append the rest of the data
  # awk -F '\t' '$150 == "$MYSPECIMENLABEL" { print }' airr-covid-19.tsv
  awk -F '\t' "\$150 == \"$specimen_label\" { print }" data/external_cohorts/raw_data/shomuradova/airr-covid-19.tsv >> "data/external_cohorts/raw_data/shomuradova/split.$specimen_label.tsv"

  # export to fasta
  # note special escape for tab delimter character
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/shomuradova/split.$specimen_label.tsv" \
    --output "data/external_cohorts/raw_data/shomuradova/split.$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator $'\t' \
    --column "sequence";

  echo "split.$specimen_label.tsv" "split.$specimen_label.fasta"
done

# Move these to yellowblade:
mkdir -p /data/maxim/shomuradova
cd /data/maxim/shomuradova
scp 'maximz@nandi:code/boyd-immune-repertoire-classification/data/external_cohorts/raw_data/shomuradova/*.fasta' .

# Chunk the fasta files.
# Use gnu split command because we know the fastas we generated always have each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)
cd /data/maxim/shomuradova
rm -r splits
mkdir -p splits
for fname in *.fasta; do
  split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" "$fname" "splits/$fname.part"
done

# Run igblast
cp $HOME/boydlab/pipeline/run_igblast_command_tcr.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;
# IN TMUX:
find splits -name "*.part*.fasta" | xargs -I {} -n 1 -P 55 sh -c "./run_igblast_command_tcr.sh {}"

# Monitor
find splits -name "*.part*.fasta" | wc -l
find splits -name "*.parse.txt" | wc -l

# Parse to file
# IN TMUX:
conda deactivate
source ~/boydlab/pyenv/activate
# $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus TCRB splits/*.parse.txt
# parallelize in chunk size of 50 parses x 40 processes:
find splits -name "*.parse.txt" | xargs -x -n 50 -P 40 $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus "TCRB"

# Monitor
find splits -name "*.parse.txt" | wc -l
find splits -name "*.parse.txt.parsed.tsv" | wc -l

# We will then join IgBlast parsed output to the original data in a notebook.

# Here's how to transfer this giant set of parses. Standard scp can fail with "argument list too long", but this works:
rsync -a --include='*.parse.txt.parsed.tsv' --exclude='*' splits/ maximz@nandi:code/boyd-immune-repertoire-classification/data/external_cohorts/raw_data/shomuradova/igblast_splits/
```

# Britanova (healthy control TCRB)

Papers: https://www.jimmunol.org/content/192/6/2689 and https://www.jimmunol.org/content/196/12/5005

Data: https://zenodo.org/record/826447#.Yy0tbezMLAw

Sequences not available, have to use their V gene calls directly. I already see that they use 12-4 where we use 12-3, and 6-3 where we use 6-2. Added rename logic on import to make our datasets consistent.

```bash
pip install zenodo_get
mkdir -p data/external_cohorts/raw_data/chudakov_aging/
cd data/external_cohorts/raw_data/chudakov_aging/
zenodo_get 826447
```

---

# Adaptive TCR cohorts

Run them through our IgBlast, without importing to Postgres.

We run IgBlast ourselves using the "rearrangement" field (raw sequence detected in the assay). (Note there's also an "extended_rearrangement" field, which is an inferred full rearrangement: “The full length TCR imputed via algorithm for the Rearrangement; includes the full CDR1, CDR2 and CDR3 region.”)

Our IgBlast gives some different V gene calls, but generally doesn't provide CDR3 calls for these short sequences. That's because our parser looks for the location of our primers. We'll use the V/J gene and productive calls from our IgBlast, while using Adaptive's CDR3 call.

```bash
## Export Adaptive sequences to FASTA files
mkdir -p "data/external_cohorts/raw_data/adaptive_emerson_and_immunecode"

# Healthy TCR specimens
for fname in $(ls data/external_cohorts/raw_data/emerson/*.tsv)
do
  # extract file name without the path and then without the extension
  # https://stackoverflow.com/a/965072/130164
  fname_without_folder=$(basename -- "$fname");
  specimen_label="${fname_without_folder%.*}";
  # export to fasta
  # note special escape for tab delimter character
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/emerson/$specimen_label.tsv" \
    --output "data/external_cohorts/raw_data/adaptive_emerson_and_immunecode/split.$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator $'\t' \
    --column "rearrangement";

  echo "$specimen_label.tsv" "split.$specimen_label.fasta"
done

# Covid TCR specimens
# Cut metadata list - get column 2: specimen_label
head -n 1 metadata/generated.external_cohorts.adaptive_covid_tcr.specimens.tsv | cut -f2 # specimen_label
for specimen_label in $(cat metadata/generated.external_cohorts.adaptive_covid_tcr.specimens.tsv | cut -f2)
do
  # export to fasta
  # note special escape for tab delimter character
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/immunecode/reps/ImmuneCODE-Review-002/$specimen_label.tsv" \
    --output "data/external_cohorts/raw_data/adaptive_emerson_and_immunecode/split.$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator $'\t' \
    --column "rearrangement";

  echo "$specimen_label.tsv" "split.$specimen_label.fasta"
done

###

# Move these to yellowblade:
mkdir -p /data/maxim/adaptive_emerson_and_immunecode
cd /data/maxim/adaptive_emerson_and_immunecode
scp 'maximz@nandi:code/boyd-immune-repertoire-classification/data/external_cohorts/raw_data/adaptive_emerson_and_immunecode/*.fasta' .

# Chunk the fasta files.
# Use gnu split command because we know the fastas we generated always have each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)
cd /data/maxim/adaptive_emerson_and_immunecode
rm -r splits
mkdir -p splits
for fname in exported.*.fasta; do
  split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" "$fname" "splits/$fname.part"
done

# Run igblast
cp $HOME/boydlab/pipeline/run_igblast_command_tcr.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;
# IN TMUX:
find splits -name "*.part*.fasta" | xargs -I {} -n 1 -P 55 sh -c "./run_igblast_command_tcr.sh {}"

# Monitor
find splits -name "*.part*.fasta" | wc -l
find splits -name "*.parse.txt" | wc -l

# Parse to file
# IN TMUX:
conda deactivate
source ~/boydlab/pyenv/activate
# $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus TCRB splits/*.parse.txt
# parallelize in chunk size of 50 parses x 40 processes:
find splits -name "*.parse.txt" | xargs -x -n 50 -P 40 $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus "TCRB"

# Monitor
find splits -name "*.parse.txt" | wc -l
find splits -name "*.parse.txt.parsed.tsv" | wc -l

# We will then join IgBlast parsed output to the original Adaptive data inside assign_clone_ids.ipynb.

# Here's how to transfer this giant set of parses. Standard scp fails with "argument list too long", but this works:
rsync -a --include='*.parse.txt.parsed.tsv' --exclude='*' splits/ maximz@nandi:code/boyd-immune-repertoire-classification/data/external_cohorts/raw_data/adaptive_emerson_and_immunecode/igblast_splits/
```
