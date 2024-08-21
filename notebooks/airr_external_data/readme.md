# Import downloaded samples and run IgBlast to annotate the sequences consistently

For each study, add an example of the data format to our snapshot tests in `tests/test_etl.py`.

When compiling metadata in upcoming notebooks, make sure the study names match the data locations on disk.

## Kim et al., Covid19 BCR

Collect metadata from the iReceptor Gateway export:

```bash
# convert json to tsv
cat data/external_cohorts/raw_data/Kim/airr-covid-19-1-metadata.json | python notebooks/airr_external_data/airr_metadata_to_tsv.py > data/external_cohorts/raw_data/Kim/airr_covid19_metadata.tsv
```

Avoid our own Boydlab study marked as `study.study_id = PRJNA628125`.

Reviewing the metadata for study `PRJNA648677`, we see 16 repertoires from 7 patients. Choosing the peak time points only (day 14-21 -- since seroconverting around day 11), we have:

* Sample `A_d17`: male, age 55, 17 days since symptom onset, with Extensive Pneumonic infiltrates
* Sample `B_d19`: male, age 55, 19 days since symptom onset, with Limited Pneumonic infiltrates
* Sample `C_d15`: female, age 53, 15 days since symptom onset, with Limited Pneumonic infiltrates
* Sample `D_d28`: male, age 24, 28 days since symptom onset, with Limited Pneumonic infiltrates
* Sample `E_d23`: male, age 48, 23 days since symptom onset, with Extensive Pneumonic infiltrates
* Sample `F_d14`: female, age 40, 14 days since symptom onset, with Limited Pneumonic infiltrates
* Sample `G_d22`: female, age 59, 22 days since symptom onset, with Limited Pneumonic infiltrates

The columns we used were:

* `study.study_id`
* `repertoire_id`
* `sample.0.sample_id`
* `subject.sex`
* `subject.age_min`
* `subject.race`
* `subject.diagnosis.0.disease_length`
* `subject.diagnosis.0.disease_stage`

Process those samples:

```bash
# All of those samples have a repertoire ID we can use to find their sequences.
# Use a here-doc to store our list of friendly sample names and repertoire IDs
f=$(mktemp)
cat << HERE > $f
specimen_label,repertoire_id
A_d17,5f21e814e1adeb2edc12613d
B_d19,5f21e815e1adeb2edc126140
C_d15,5f21e816e1adeb2edc126142
D_d28,5f21e817e1adeb2edc126144
E_d23,5f21e817e1adeb2edc126145
F_d14,5f21e818e1adeb2edc126148
G_d22,5f21e819e1adeb2edc12614b
HERE

# Find column number for "repertoire_id" column in the sequence file: https://unix.stackexchange.com/a/304320
# Insert it in the awk command below. Awk column numbers start at 1.
sed 's/\t/\n/g;q' data/external_cohorts/raw_data/Kim/airr-covid-19-1.tsv | nl -ba | grep "repertoire_id" # Result: 138 repertoire_id

# Loop over the friendly sample label and the repertoire ID columns together (skip header row when loading)
while IFS=$',' read -r specimen_label repertoire_id
do
  echo "Processing specimen label $specimen_label with repertoire ID $repertoire_id"

  # Make a separate file with only this specimen's data.
  # first we need to put the header row in each file
  head -n 1 data/external_cohorts/raw_data/Kim/airr-covid-19-1.tsv > "data/external_cohorts/raw_data/Kim/$specimen_label.tsv"

  # now append the rest of the data
  # insert the column number here for the 'repertoire_id' column (see above)
  # awk -F '\t' '$138 == "$repertoire_id" { print }' airr-covid-19-1.tsv
  awk -F '\t' "\$138 == \"$repertoire_id\" { print }" data/external_cohorts/raw_data/Kim/airr-covid-19-1.tsv >> "data/external_cohorts/raw_data/Kim/$specimen_label.tsv"

  # Export to fasta.
  # note special escape for tab delimter character
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/Kim/$specimen_label.tsv" \
    --output "data/external_cohorts/raw_data/Kim/$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator $'\t' \
    --column "sequence";

  # Chunk the fasta file.
  # sample_name.fasta --> sample_name.fasta.part_001.fasta
  seqkit split2 "data/external_cohorts/raw_data/Kim/$specimen_label.fasta" -O "data/external_cohorts/raw_data/Kim" --by-size 10000 --by-size-prefix "$specimen_label.fasta.part_"
  echo "$specimen_label.tsv" "$specimen_label.fasta"

done < <(tail -n +2 $f)
rm $f

# Run igblast:
# data/external_cohorts/raw_data/Kim/sample_name.fasta.part_001.fasta -> data/external_cohorts/raw_data/Kim/sample_name.fasta.part_001.fasta.parse.txt
tmpdir_igblast=$(mktemp -d)
echo "$tmpdir_igblast"
cp scripts/run_igblast_command.sh "$tmpdir_igblast";
cp igblast/igblastn "$tmpdir_igblast";
cp igblast/human_gl* "$tmpdir_igblast";
cp -r igblast/internal_data/ "$tmpdir_igblast";
workdir=$(pwd) # mark current directory
pushd "$tmpdir_igblast" # switch to new directory

num_processors=50

# use -print0 and -0 to handle spaces in filenames
# _ is a dummy value for $0 (the script name)
# $1 in the sh -c command will be the filename
find $workdir/data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblast_command.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta" | wc -l
find data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta.parse.txt" | wc -l

# Parse to file with: scripts/parse_igblastn.py --locus "IgH" splits/*.parse.txt
# But parallelize in chunk size of 50 parses x 40 processes:
num_processors=40
# use -print0 and -0 to handle spaces in filenames
find data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta.parse.txt" -print0 | xargs -0 -x -n 50 -P "$num_processors" scripts/parse_igblastn.py --locus "IgH"
echo $?

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/external_cohorts/raw_data/Kim/ -name "*.part_*.fasta.parse.txt.parsed.IgH.tsv" | wc -l
```

We will then join IgBlast parsed output to the original data inside `etl.ipynb`.

---

## Briney et al., healthy BCR

Download data:

```bash
# Data from https://github.com/briney/grp_paper
mkdir data/external_cohorts/raw_data/Briney/
pushd data/external_cohorts/raw_data/Briney/
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/316188_HNCHNBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326650_HCGCYBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326737_HNKVKBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326780_HLH7KBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326797_HCGNLBCXY%2BHJLN5BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/326907_HLT33BCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/327059_HCGTCBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz
wget http://burtonlab.s3.amazonaws.com/sequencing-data/hiseq_2016-supplement/D103_HCGCLBCXY_consensus_UID18-cdr3nt-90_minimal_071817.tar.gz

# Untar
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
popd
```

Review contents of one file - shows that we have isotype info included, but notice that patient ID is not included:

```bash
head -n 1 data/external_cohorts/raw_data/Briney/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | sed 's/,/\n/g'
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

cat data/external_cohorts/raw_data/Briney/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f3 | sort -u
# chain
# heavy
# lambda

cat data/external_cohorts/raw_data/Briney/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f4 | sort -u
# no
# productive
# yes

cat data/external_cohorts/raw_data/Briney/316188/consensus-cdr3nt-90_minimal/1_consensus.txt | cut -d ',' -f25 | sort -u
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

We see files `1_consensus.txt`, `2_consensus.txt`, etc. Explanation:

> Samples 1-6 are biological replicates. Samples 7-12 and 13-18 are technical replicates of samples 1-6.
>
> Biological replicates refer to different aliquots of peripheral blood monomuclear cells (PBMCs), from which total RNA was separately isolated and processed. Thus, sequences or clonotypes found in multiple biological replicates are assumed to have independently occurred in different cells

So just load 1 through 6. For now we actually only load 1. TODO: load more biological replicates.

```bash
pushd data/external_cohorts/raw_data/Briney
mv D103/consensus-cdr3nt-90_minimal/1_consensus.txt D103_1.csv;
mv 326780/consensus-cdr3nt-90_minimal/1_consensus.txt 326780_1.csv;
mv 326650/consensus-cdr3nt-90_minimal/1_consensus.txt 326650_1.csv;
mv 326737/consensus-cdr3nt-90_minimal/1_consensus.txt 326737_1.csv;
mv 327059/consensus-cdr3nt-90_minimal/1_consensus.txt 327059_1.csv;
mv 326907/consensus-cdr3nt-90_minimal/1_consensus.txt 326907_1.csv;
mv 316188/consensus-cdr3nt-90_minimal/1_consensus.txt 316188_1.csv;
mv 326797/consensus-cdr3nt-90_minimal/1_consensus.txt 326797_1.csv;
popd

for specimen_label in 'D103_1' '326780_1' '326650_1' '326737_1' '327059_1' '326907_1' '316188_1' '326797_1'
do

  # Export to fasta.
  # Comma delimeted
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/Briney/$specimen_label.csv" \
    --output "data/external_cohorts/raw_data/Briney/$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator ',' \
    --column "raw_input";

  # Chunk the fasta file.
  # sample_name.fasta --> sample_name.fasta.part_001.fasta
  seqkit split2 "data/external_cohorts/raw_data/Briney/$specimen_label.fasta" -O "data/external_cohorts/raw_data/Briney" --by-size 10000 --by-size-prefix "$specimen_label.fasta.part_"
  echo "$specimen_label.csv" "$specimen_label.fasta"

done

# Run igblast:
# data/external_cohorts/raw_data/Briney/sample_name.fasta.part_001.fasta -> data/external_cohorts/raw_data/Briney/sample_name.fasta.part_001.fasta.parse.txt
tmpdir_igblast=$(mktemp -d)
echo "$tmpdir_igblast"
cp scripts/run_igblast_command.sh "$tmpdir_igblast";
cp igblast/igblastn "$tmpdir_igblast";
cp igblast/human_gl* "$tmpdir_igblast";
cp -r igblast/internal_data/ "$tmpdir_igblast";
workdir=$(pwd) # mark current directory
pushd "$tmpdir_igblast" # switch to new directory

num_processors=50

# use -print0 and -0 to handle spaces in filenames
# _ is a dummy value for $0 (the script name)
# $1 in the sh -c command will be the filename
find $workdir/data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblast_command.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta" | wc -l
find data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta.parse.txt" | wc -l

# Parse to file with: scripts/parse_igblastn.py --locus "IgH" splits/*.parse.txt
# But parallelize in chunk size of 50 parses x 40 processes:
num_processors=40
# use -print0 and -0 to handle spaces in filenames
find data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta.parse.txt" -print0 | xargs -0 -x -n 50 -P "$num_processors" scripts/parse_igblastn.py --locus "IgH"
echo $?

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/external_cohorts/raw_data/Briney/ -name "*.part_*.fasta.parse.txt.parsed.IgH.tsv" | wc -l
```

We will then join IgBlast parsed output to the original data inside `etl.ipynb`.

---


## Shomuradova (Covid19 TCRB)

```bash
# Find column number for "repertoire_id" column in the sequence file: https://unix.stackexchange.com/a/304320
# Insert it in the awk command below. Awk column numbers start at 1.
sed 's/\t/\n/g;q' data/external_cohorts/raw_data/Shomuradova/airr-covid-19.tsv | nl -ba | grep "repertoire_id" # Result: 150 repertoire_id
# Extra check
head -n 1 data/external_cohorts/raw_data/Shomuradova/airr-covid-19.tsv | cut -f150 # repertoire_id
# So we will cut on repertoire_id column 150

# save all specimen labels
cut -f150 data/external_cohorts/raw_data/Shomuradova/airr-covid-19.tsv | sort -u | grep -v repertoire_id > data/external_cohorts/raw_data/Shomuradova/specimen_labels.txt

# review as sanity check
cat data/external_cohorts/raw_data/Shomuradova/specimen_labels.txt

# cut by specimen label
for specimen_label in $(cat data/external_cohorts/raw_data/Shomuradova/specimen_labels.txt)
do
  # Make a separate file with only this specimen's data.
  # first we need to put the header row in each file
  head -n 1 data/external_cohorts/raw_data/Shomuradova/airr-covid-19.tsv > "data/external_cohorts/raw_data/Shomuradova/$specimen_label.tsv"

  # now append the rest of the data
  # awk -F '\t' '$150 == "$MYSPECIMENLABEL" { print }' airr-covid-19.tsv
  awk -F '\t' "\$150 == \"$specimen_label\" { print }" data/external_cohorts/raw_data/Shomuradova/airr-covid-19.tsv >> "data/external_cohorts/raw_data/Shomuradova/$specimen_label.tsv"

  # Export to fasta.
  # note special escape for tab delimter character
  python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/Shomuradova/$specimen_label.tsv" \
    --output "data/external_cohorts/raw_data/Shomuradova/$specimen_label.fasta" \
    --name "$specimen_label" \
    --separator $'\t' \
    --column "sequence";

  # Chunk the fasta file.
  # sample_name.fasta --> sample_name.fasta.part_001.fasta
  seqkit split2 "data/external_cohorts/raw_data/Shomuradova/$specimen_label.fasta" -O "data/external_cohorts/raw_data/Shomuradova" --by-size 10000 --by-size-prefix "$specimen_label.fasta.part_"
  echo "$specimen_label.tsv" "$specimen_label.fasta"
done

# Run igblast:
# data/external_cohorts/raw_data/Shomuradova/sample_name.fasta.part_001.fasta -> data/external_cohorts/raw_data/Shomuradova/sample_name.fasta.part_001.fasta.parse.txt
tmpdir_igblast=$(mktemp -d)
echo "$tmpdir_igblast"
cp scripts/run_igblast_command_tcr.sh "$tmpdir_igblast";
cp igblast/igblastn "$tmpdir_igblast";
cp igblast/human_gl* "$tmpdir_igblast";
cp -r igblast/internal_data/ "$tmpdir_igblast";
workdir=$(pwd) # mark current directory
pushd "$tmpdir_igblast" # switch to new directory

num_processors=50

# use -print0 and -0 to handle spaces in filenames
# _ is a dummy value for $0 (the script name)
# $1 in the sh -c command will be the filename
find $workdir/data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblast_command_tcr.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta" | wc -l
find data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta.parse.txt" | wc -l

# Parse to file with: scripts/parse_igblastn.py --locus TCRB splits/*.parse.txt
# But parallelize in chunk size of 50 parses x 40 processes:
num_processors=40
# use -print0 and -0 to handle spaces in filenames
find data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta.parse.txt" -print0 | xargs -0 -x -n 50 -P "$num_processors" scripts/parse_igblastn.py --locus "TCRB"
echo $?

# Monitor: these numbers must match
find data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/external_cohorts/raw_data/Shomuradova/ -name "*.part_*.fasta.parse.txt.parsed.TCRB.tsv" | wc -l
```

We will then join IgBlast parsed output to the original data inside `etl.ipynb`.

---

## Britanova (healthy control TCRB)

Papers: https://www.jimmunol.org/content/192/6/2689 and https://www.jimmunol.org/content/196/12/5005

Data: https://zenodo.org/record/826447#.Yy0tbezMLAw

Sequences not available, have to use their V gene calls directly. I already see that they use 12-4 where we use 12-3, and 6-3 where we use 6-2. Added rename logic on import to make our datasets consistent.

```bash
pip install zenodo_get
mkdir -p data/external_cohorts/raw_data/Britanova/
cd data/external_cohorts/raw_data/Britanova/
zenodo_get 826447
```
