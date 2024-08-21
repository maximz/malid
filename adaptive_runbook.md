# Adaptive TCR cohorts

Download and run them through our IgBlast, using the "rearrangement" field or similar (raw sequence detected in the assay — not to be confused with the "extended_rearrangement" field that is sometimes present, which is an inferred full-length TCR).

Our IgBlast gives some different V gene calls, but generally doesn't provide CDR3 calls for these short sequences. That's because our parser looks for the location of our primers. We'll use the V/J gene and productive calls from our IgBlast, along with as many sequence regions as IgBlast can reasonably give us, while using Adaptive's CDR3 call. That logic is in etl.py.

Here, we run IgBlast.

```bash
# Metadata
./run_notebooks.sh \
  notebooks/airr_external_data/covid_tcr_immunecode_metadata.ipynb \
  notebooks/airr_external_data/adaptive_cohorts.metadata.ipynb;

# Copy the subset of immunecode we'll use.
mkdir -p data/external_cohorts/raw_data/adaptive_immuneaccess/immunecode;
tail -n +2 metadata/adaptive/generated.immunecode_covid_tcr.specimens.tsv | cut -f2 | while read specimen_label; do
  cp "data/external_cohorts/raw_data/immunecode_all/reps/ImmuneCODE-Review-002/$specimen_label.tsv" "data/external_cohorts/raw_data/adaptive_immuneaccess/immunecode/";
done

# Extract zips
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -type f -name "sampleExport*.zip" -execdir ls '{}' \; # dry run
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -type f -name "sampleExport*.zip" -execdir unzip '{}' \; # real thing

# Follow instructions in adaptive_cohorts.metadata.ipynb for how to get the Emerson dataset.
# After unzipping, split into emerson-2017-natgen_train and emerson-2017-natgen_validation folders:
mkdir data/external_cohorts/raw_data/adaptive_immuneaccess/emerson-2017-natgen_validation;
mv data/external_cohorts/raw_data/adaptive_immuneaccess/emerson-2017-natgen/Keck* data/external_cohorts/raw_data/adaptive_immuneaccess/emerson-2017-natgen_validation/;
mv data/external_cohorts/raw_data/adaptive_immuneaccess/emerson-2017-natgen data/external_cohorts/raw_data/adaptive_immuneaccess/emerson-2017-natgen_train;

# Confirm all files exist.
# cut metadata list to get study name and sample name (skip header row when cutting)
# loop over the two columns to assemble the file name
# if instead were to loop over single column, could just do: for specimen_label in $(cat file.tsv | cut -f2) do ... done
while IFS=$'\t' read -r study_name sample_name
do
  filepath="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name/$sample_name.tsv"
  # echo $filepath
  if [ ! -e "$filepath" ]; then
    echo "File $filepath does not exist."
  fi
done < <(tail -n +2 metadata/adaptive/generated.adaptive_external_cohorts.tsv | cut -f1,2)


# What tsv header rows do we have to support?
while IFS=$'\t' read -r study_name sample_name
do
  filepath="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name/$sample_name.tsv"
  echo $(head -n 1 "$filepath")
done < <(tail -n +2 metadata/adaptive/generated.adaptive_external_cohorts.tsv | cut -f1,2) | sort -u


# Export to fasta.
# Supply possible column names for the detected VDJ rearrangement sequence, per the header row investigation above.
while IFS=$'\t' read -r study_name sample_name
do
  filepath_in="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name/$sample_name.tsv"
  filepath_out="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name/$sample_name.fasta"
  # echo $filepath
  # note special escape for tab delimter character
  # it will try each column name in order.
  python scripts/export_sequences_to_fasta.py \
    --input "$filepath_in" \
    --output "$filepath_out" \
    --name "$sample_name" \
    --separator $'\t' \
    --column "nucleotide" \
    --column "rearrangement";

  echo "$filepath_out"

done < <(tail -n +2 metadata/adaptive/generated.adaptive_external_cohorts.tsv | cut -f1,2)


# Confirm all fasta files exist, following the example above.
while IFS=$'\t' read -r study_name sample_name
do
  filepath="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name/$sample_name.fasta"
  # echo $filepath
  if [ ! -e "$filepath" ]; then
    echo "File $filepath does not exist."
  fi
done < <(tail -n +2 metadata/adaptive/generated.adaptive_external_cohorts.tsv | cut -f1,2)


# Chunk the fasta files.
# Old: Use gnu split command because we know the fastas we generated always have each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)
# New: use seqkit split, which is the right tool for the job
mamba install -c bioconda seqkit -y;

while IFS=$'\t' read -r study_name sample_name
do
  foldername_out="data/external_cohorts/raw_data/adaptive_immuneaccess/$study_name"
  filename_in="$sample_name.fasta"
  # OLD (gnu split):
  # sample_name.fasta --> sample_name.fasta.part0000000001.fasta
  # split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" "$filename_in" "splits/$filename_in.part"

  # NEW (seqkit split):
  # sample_name.fasta --> sample_name.fasta.part_001.fasta
  seqkit split2 "$foldername_out/$filename_in" -O "$foldername_out" --by-size 10000 --by-size-prefix "$filename_in.part_"
done < <(tail -n +2 metadata/adaptive/generated.adaptive_external_cohorts.tsv | cut -f1,2)





# Run igblast:
# data/external_cohorts/raw_data/adaptive_immuneaccess/study_name/sample_name.fasta.part_001.fasta -> data/external_cohorts/raw_data/adaptive_immuneaccess/study_name/sample_name.fasta.part_001.fasta.parse.txt
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
find $workdir/data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblast_command_tcr.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta" | wc -l
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta.parse.txt" | wc -l

# Parse to file with: scripts/parse_igblastn.py --locus TCRB splits/*.parse.txt
# But parallelize in chunk size of 50 parses x num_processors=40 processes:
num_processors=40
# use -print0 and -0 to handle spaces in filenames
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta.parse.txt" -print0 | xargs -0 -x -n 50 -P "$num_processors" scripts/parse_igblastn.py --locus "TCRB"
echo $?

# Monitor: these numbers must match
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/external_cohorts/raw_data/adaptive_immuneaccess/ -name "*.part_*.fasta.parse.txt.parsed.TCRB.tsv" | wc -l
```

We will then join IgBlast parsed output to the original Adaptive data inside `etl.ipynb`.
