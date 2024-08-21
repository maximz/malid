### IgH data preprocessing
# These steps are followed for all IgH runs. Here we are using M491 as an example.

## Step 1. Run FLASH v1.2.11
flash --compress --threads=8 --max-overlap=1000 \
    M491_S1_L001_R1_001.fastq.gz \
    M491_S1_L001_R2_001.fastq.gz > flash_output.txt;

# flash_output.txt starts with:
# [FLASH] Starting FLASH v1.2.11
# [FLASH] Fast Length Adjustment of SHort reads
# [FLASH]
# [FLASH] Input files:
# [FLASH]     M491_S1_L001_R1_001.fastq.gz
# [FLASH]     M491_S1_L001_R2_001.fastq.gz
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

## Step 2. Load sequences into Postgres database. This creates a reads table called reads_m491.
# All further steps create new tables based on this reads table.
./load_sequences.py reads_m491 out.extendedFrags.fastq.gz

## Step 3. Demultiplexing into a new table called demuxed_reads_m491.
# Reads are mapped to:
# - M491-S001_cDNA_PCR_IGA
# - M491-S001_cDNA_PCR_IGD
# - M491-S001_cDNA_PCR_IGE
# - M491-S001_cDNA_PCR_IGG
# - M491-S001_cDNA_PCR_IGM
# - M491-S002_cDNA_PCR_IGA
# - M491-S002_cDNA_PCR_IGD
# - M491-S002_cDNA_PCR_IGE
# and so on. (M491-S001, M491-S002, etc. are the sample names.)
./read_demuxer.py M491 reads_m491 demuxed_reads_m491

## Step 4. Primer trimming, into a new table called trimmed_reads_m491.
./read_trimmer.py demuxed_reads_m491 trimmed_reads_m491

## Step 5. Run IgBlast.
# First, export fasta and write accompanying shell scripts into a new folder called M491_igblast.
./parse_with_igblast.py --locus IgH trimmed_reads_m491 demuxed_reads_m491 M491_igblast

# Now run IgBlast (in parallel).
cd M491_igblast
find . -name "job_??????.sh" | xargs -I JOB -n 1 -P 60 sh "JOB"

# Finally, load the IgBlast outputs back into the Postgres database. Creates table parsed_igh_igblast_m491.
cd ../
./load_igblast_parse.py --locus IgH trimmed_reads_m491 parsed_igh_igblast_m491 M491_igblast/*.parse.txt
rm -r M491_igblast;

## Step 6. Identify subisotypes. Creates table isosubtypes_m491.
# Note: Mal-ID does not use subisotype resolution, and isotype information is already available from the demultiplexing step.
# But we're including this step for the sake of completeness.
./iso_subtype.py demuxed_reads_m491 trimmed_reads_m491 parsed_igh_igblast_m491 isosubtypes_m491

## Step 7. Glue script to copy the sequences into person-specific tables.
./sort_all_participants_igh_fast.py M491 \
            --read-chunk-size 100000 \
            --write-chunk-size 5000 \
            --num-jobs 50 &> m491_igh_sort.out;

## Step 8. Cluster the sequences from each person.
# First, write each person's sequences to disk, creating one folder per person.
mkdir -p clone_clustering
# run_which_part.py returns all participant IDs that belong to this sequencing run.
for p in $(./run_which_part.py M491) ; do ./per_person_split_for_clone_clustering.py --locus IgH $p clone_clustering/$p/; done

# Then cluster each person's sequences.
cd clone_clustering
find $(../run_which_part.py M491) -name "seq.*.fasta" | xargs -I {} -n 1 -P 55 bash -c "../single_linkage_clustering.py --percent-id 0.90 \"{}\" >\"{}.clust\""

# Finally, load the cluster assignments back into the database.
# This creates clones and clone_members tables for each person with names igh_clones_BFI-00##### and igh_clone_members_BFI-00#####.
cd ../
for p in $(./run_which_part.py M491) ; do find clone_clustering/$p -name "seq.*.*.*.fasta.clust" | ./load_per_person_igh_igblast_clusters.py $p ; done
rm -r clone_clustering;
