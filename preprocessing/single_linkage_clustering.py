#!/usr/bin/env python

import sys
import argparse
from itertools import combinations, count, izip, izip_longest
import logging

from Bio import SeqIO

import numpy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from numpy import zeros

numpy.set_printoptions(threshold=numpy.nan)

def mismatch_count(seq_i, seq_j, maximum=100000, max_ceiling=1000000):
    mismatches = 0
    for base_i, base_j in izip_longest(seq_i, seq_j):
        assert base_i is not None and base_j is not None
        if base_i != base_j:
            mismatches += 1
        if mismatches > maximum:
            return max_ceiling
    return mismatches

# program options
parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('filename', metavar='file', nargs='+', help='FASTA file to cluster')
parser.add_argument('--percent-id', '-p', metavar='P', default=0.85, type=float,
        help='percent identity cutoff for nucleotide sequence')
parser.add_argument('--log-level', metavar='L', default='INFO', help='logging level',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                'debug', 'info', 'warning', 'error', 'critical'])

# parse the args
args = parser.parse_args()
log_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(level=log_level)

# open file and colapse identical sequences
sequences = {}
for filename in args.filename:
    for record in SeqIO.parse(open(filename, 'r'), 'fasta'):
        seq = str(record.seq)
        if seq in sequences:
            sequences[seq].id += ',' + record.id
        else:
            sequences[seq] = record

sequences = sequences.values()
sequence_count = len(sequences)

# make sure all sequences have the same length
sequence_length = len(sequences[0])
for s in sequences:
    if len(s) != sequence_length:
        logging.error('not all sequences have the same length, %d != %d' % (sequence_length, len(s)))
        sys.exit(10)

if sequence_count == 0:
    pass
elif sequence_count == 1:
    print '%s\t' % sequences[0].id
else:
    max_diff = int(sequence_length - args.percent_id * sequence_length)
    #logging.info('max mismatches %d' % max_diff)

    # make the distance table
    distances = zeros(sequence_count * (sequence_count - 1) / 2)
    #logging.info('calculating distances between %d sequences' % len(sequences))
    for (seq_i, seq_j), idx in izip(combinations(sequences, 2), count()):
        distance = mismatch_count(seq_i.seq, seq_j.seq, max_diff, max_ceiling=1000000)
        distances[idx] = distance
        idx += 1

    clustering = linkage(distances, method='single')

    flat = fcluster(clustering, max_diff, criterion='distance')
    flat_clusters = {}
    for record_idx in range(len(flat)):
        cluster_idx = flat[record_idx]
        if cluster_idx not in flat_clusters:
            flat_clusters[cluster_idx] = set()
        flat_clusters[cluster_idx].add(record_idx)

    for cluster in flat_clusters.values():
        if len(cluster) == 1:
            center = list(cluster)[0]
            print '%s\t' % sequences[center].id
        elif len(cluster) == 2:
            print '\t'.join([sequences[i].id for i in cluster])
        else:
            print '\t'.join([sequences[i].id for i in cluster])
