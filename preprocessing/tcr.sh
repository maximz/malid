# TCRB data preprocessing
# These steps are followed for all TCRB runs.
# Here we are using M493 as an example. M491 is the associated IgH run.

## Step 1. Run FLASH v1.2.11
flash --compress --threads=8 --max-overlap=1000 \
    M493_S1_L001_R1_001.fastq.gz \
    M493_S1_L001_R2_001.fastq.gz > flash_output.txt;

# flash_output.txt starts with:
# [FLASH] Starting FLASH v1.2.11
# [FLASH] Fast Length Adjustment of SHort reads
# [FLASH]
# [FLASH] Input files:
# [FLASH]     M493_S1_L001_R1_001.fastq.gz
# [FLASH]     M493_S1_L001_R2_001.fastq.gz
# [FLASH]
# [FLASH] Output files:
# [FLASH]     ./out.extendedFrags.fastq.gz
# [FLASH]     ./out.notCombined_1.fastq.gz
# [FLASH]     ./out.notCombined_2.fastq.gz
# [FLASH]     ./out.hist
# [FLASH]     ./out.histogram
# [FLASH]
# [FLASH] Parameters:
# [FLASH]     Min overlap:           10
# [FLASH]     Max overlap:           1000
# [FLASH]     Max mismatch density:  0.250000
# [FLASH]     Allow "outie" pairs:   false
# [FLASH]     Cap mismatch quals:    false
# [FLASH]     Combiner threads:      8
# [FLASH]     Input format:          FASTQ, phred_offset=33
# [FLASH]     Output format:         FASTQ, phred_offset=33, gzip
# [FLASH]
# [FLASH] Starting reader and writer threads
# [FLASH] Starting 8 combiner threads
# [FLASH] Processed 25000 read pairs
# [FLASH] Processed 50000 read pairs


## Step 2. Load sequences into Postgres database. This creates a reads table called reads_m493.
# All further steps create new tables based on this reads table.
./load_sequences.py reads_m493 out.extendedFrags.fastq.gz

## Step 3. Demultiplexing into a new table called demuxed_reads_m493.
# Reads are mapped to:
# - M491-S001_cDNA_PCR_TCRB_R1
# - M491-S002_cDNA_PCR_TCRB_R1
# - M491-S003_cDNA_PCR_TCRB_R1
# and so on.
# M491-S001, M491-S002, M491-S003, etc. are the sample names. Notice we keep the same sample names as in M491.
./read_demuxer.py --reverse-skip 4 M493 reads_m493 demuxed_reads_m493

## Step 4. Primer trimming, into a new table called trimmed_reads_m493.
./read_trimmer.py demuxed_reads_m493 trimmed_reads_m493

## Step 5. Run IgBlast.
# First, export fasta and write accompanying shell scripts into a new folder called M493_igblast.
./parse_with_igblast.py --locus TCRB trimmed_reads_m493 demuxed_reads_m493 M493_igblast

# Now run IgBlast (in parallel).
cd M493_igblast
find . -name "job_??????.sh" | xargs -I JOB -n 1 -P 60 sh -c "JOB"

# Finally, load the IgBlast outputs back into the Postgres database. Creates table parsed_igh_igblast_m493.
cd ../
./load_igblast_parse.py --locus TCRB trimmed_reads_m493 parsed_tcrb_igblast_m493 M493_igblast/*.parse.txt
rm -r M493_igblast;

## Step 6. Glue script to copy the sequences into person-specific tables.
./sort_all_participants_tcrb.py M493

## Step 7. Cluster the sequences from each person.
# First, write each person's sequences to disk, creating one folder per person.
mkdir -p clone_clustering
# run_which_part.py returns all participant IDs that belong to this sequencing run.
for p in $(./run_which_part.py M493) ; do ./per_person_split_for_clone_clustering.py --locus TCRB $p clone_clustering/$p/; done

# Then cluster each person's sequences.
cd clone_clustering
find $(~/boydlab/pipeline/run_which_part.py M493) -name "seq.*.fasta" | xargs -I {} -n 1 -P 55 bash -c "../single_linkage_clustering.py --percent-id 0.95 \"{}\" >\"{}.clust\""

# Finally, load the cluster assignments back into the database.
# This creates clones and clone_members tables for each person with names tcrb_clones_BFI-00##### and tcrb_clone_members_BFI-00#####.
cd ../
for p in $(./run_which_part.py M493) ; do find clone_clustering/$p -name "seq.*.*.*.fasta.clust" | ./load_per_person_tcrb_igblast_clusters.py $p ; done
rm -r clone_clustering;
