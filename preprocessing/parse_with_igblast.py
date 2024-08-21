#!/usr/bin/env python

import logging
import sys
import argparse
import itertools
import os
import stat
import pwd

import Bio.Alphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import psycopg2
import sqlalchemy
import sqlalchemy.orm

import boydlib

username = pwd.getpwuid(os.getuid()).pw_name

# program options
parser = argparse.ArgumentParser(description='Parse reads using IgBLAST')
# the table to load and store reads from
parser.add_argument('source_table', metavar='source-table',
        help='the table to read the sequences from, ex. trimmed_reads_t31')
parser.add_argument('dest_table', metavar='dest-table',
        help='the de-multiplexed reads table name, ex. parsed_igh_igblast_t31')
# directory to store the batch info for the parse
parser.add_argument('batch_dir', metavar='batch_dir', help='the directory name to store the batch')

# which locus should be parsed
parser.add_argument('--locus', '-l', metavar='L', required=True,
        choices=['IgH', 'IgK', 'IgL', 'TCRA', 'TCRB', 'TCRG', 'TCRD'],
        help='which locus to parse: IgH, IgK, IgL, TCRA, TCRB, TCRG, TCRD (default: %(default)s)')

# the number of reads to process per run of igBlast
parser.add_argument('--batch-size', '-b', metavar='B', type=int, default=10000,
        help='the number of sequences to process per instance of igBlast (default: %(default)s)')
parser.add_argument('--job-size', '-J', metavar='J', type=int, default=600,
        help='the number of batches to schedule at once (default: %(default)s)')
#
parser.add_argument('--prolog', '-P', metavar='CMD',
        help='prolog commands to put at the top of each job script')
# IgBLAST options
igblast_options = parser.add_argument_group('IgBLAST options')
igblast_options.add_argument('--organism', metavar='O', default='human',
        help='organism for your query sequence (default: %(default)s)')
igblast_options.add_argument('--num-alignments', metavar='N', type=int, default=1,
        help='number of alignments to output (default: %(default)s)')
igblast_options.add_argument('--num-v', metavar='N', type=int, default=1,
        help='number of germline V sequences to show alignments for (default: %(default)s)')
igblast_options.add_argument('--num-d', metavar='N', type=int, default=1,
        help='number of germline D sequences to show alignments for (default: %(default)s)')
igblast_options.add_argument('--num-j', metavar='N', type=int, default=1,
        help='number of germline J sequences to show alignments for (default: %(default)s)')
igblast_options.add_argument('--domain-system', metavar='D', default='imgt',
        help='domain system to be used for segment annotation (default: %(default)s)')
igblast_options.add_argument('--focus-on-v', action='store_true',
        help='should the search only be for V segment')
igblast_options.add_argument('--show-translation', action='store_true',
        help='show translated alignments')
igblast_options.add_argument('--output-format', default='3',
        help='alignment view options (default: %(default)s)')
igblast_options.add_argument('--num-threads', type=int, default=1,
        help='number of threads (CPUs) to use in the BLAST search (default: %(default)s)')
# IgBLAST path and files
path_options = parser.add_argument_group('file locations')
path_options.add_argument('--igblast-dir', '-p', metavar='P',
        default='/home/%s/boydlab/igblast' % username, help='path to IgBLAST (default: /home/%s/boydlab/igblast)' % username)
path_options.add_argument('--files-prefix', default=None,
        help='name of the file containing IgH V repertoire (default: /home/%s/boydlab/igblast/%s_gl)' % (username, '[org]'))
path_options.add_argument('--v-rep', default=None,
        help='name of the file containing IgH V repertoire (default: %s_gl_V)' % '[org]')
path_options.add_argument('--d-rep', default=None,
        help='name of the file containing IgH D repertoire (default: %s_gl_D)' % '[org]')
path_options.add_argument('--j-rep', default=None,
        help='name of the file containing IgH J repertoire (default: %s_gl_J)' % '[org]')
path_options.add_argument('--aux-file', default=None,
        help='name of the file containing auxiliary segment information (default: %s_gl.aux)' % '[org]')
path_options.add_argument('--gene-subset',
        help='file with the subset of the segment names to match against (default: %s_gl_ig[hkl]_seqidlist)' % '[org]')
#
parser.add_argument('--force', '-F', action='store_true', default=False, help='override checks')

boydlib.add_log_level_arg(parser)
boydlib.add_read_database_args(parser)

# parse the args
args = parser.parse_args()
boydlib.set_log_level(args)

boyddb = boydlib.BoydDB(args, tables=[args.dest_table, args.source_table, 'amplifications', 'replicates'])

if args.files_prefix is None:
    args.files_prefix = '/home/%s/boydlab/igblast/%s_gl' % (username, args.organism)
if args.v_rep is None:
    args.v_rep = '%s_gl_V' % args.organism
if args.d_rep is None:
    args.d_rep = '%s_gl_D' % args.organism
if args.j_rep is None:
    args.j_rep = '%s_gl_J' % args.organism
if args.aux_file is None:
    args.aux_file = '%s_gl.aux' % args.organism
if args.gene_subset is None:
    if args.locus == 'IgH':
        args.gene_subset = '%s_gl_igh_seqidlist' % args.organism
    elif args.locus == 'IgK':
        args.gene_subset = '%s_gl_igk_seqidlist' % args.organism
    elif args.locus == 'IgL':
        args.gene_subset = '%s_gl_igl_seqidlist' % args.organism
    elif args.locus == 'TCRA':
        args.gene_subset = '%s_gl_tcra_seqidlist' % args.organism
    elif args.locus == 'TCRB':
        args.gene_subset = '%s_gl_tcrb_seqidlist' % args.organism
    elif args.locus == 'TCRG':
        args.gene_subset = '%s_gl_tcrg_seqidlist' % args.organism
    elif args.locus == 'TCRD':
        args.gene_subset = '%s_gl_tcrd_seqidlist' % args.organism
    else:
        assert 0 == 1

# which amplification loci are allowed and what chain is being processed
if args.locus == 'IgH':
    allowed_loci = ['IgH', 'IgA', 'IgD', 'IgE', 'IgG', 'IgM']
    args.sequence_type = 'Ig'
elif args.locus == 'IgK':
    allowed_loci = ['IgK']
    args.sequence_type = 'Ig'
elif args.locus == 'IgL':
    allowed_loci = ['IgL']
    args.sequence_type = 'Ig'
elif args.locus == 'TCRA':
    allowed_loci = ['TCRA']
    args.sequence_type = 'TCR'
elif args.locus == 'TCRB':
    allowed_loci = ['TCRB']
    args.sequence_type = 'TCR'
elif args.locus == 'TCRG':
    allowed_loci = ['TCRG']
    args.sequence_type = 'TCR'
elif args.locus == 'TCRD':
    allowed_loci = ['TCRD']
    args.sequence_type = 'TCR'
else:
    assert 0 == 1

if os.path.exists(args.batch_dir):
    logging.error('could not create batch directory %s, directory (or file) already exists' % args.batch_dir)
    sys.exit(10)
else:
    logging.info('creating batch directory %s' % args.batch_dir)
    os.mkdir(args.batch_dir)
    logging.info('copying files %s* to batch directory' % args.files_prefix)
    os.system('cp %s* %s' % (args.files_prefix, args.batch_dir))
    logging.info('copying internal_data directory to batch directory')
    os.system('cp -r %s/internal_data/ %s' % (args.igblast_dir, args.batch_dir))

# load the table defs
demuxed_reads  = getattr(boyddb.tables, args.dest_table)
trimmed_reads  = getattr(boyddb.tables, args.source_table)
amplifications = boyddb.tables.amplifications
replicates     = boyddb.tables.replicates

# the basic query
query = boyddb.session.query(trimmed_reads.trimmed_read_id, trimmed_reads.sequence).\
                       join(demuxed_reads).join(replicates).join(amplifications).\
                       filter(amplifications.locus.in_(allowed_loci))

data_iter = iter(query.all())

job_filenames = []

# for each batch
file_index = 0
batch = list(itertools.islice(data_iter, args.batch_size))
while len(batch) > 0:
    records = []
    for id, sequence in batch:
        records.append(SeqRecord(Seq(sequence, Bio.Alphabet.generic_dna), id=str(id), description=''))

    batch_filename = 'seq_%06d.fasta' % file_index
    job_filename   = 'job_%06d.sh' % file_index
    batch_pathname = '%s/%s' % (args.batch_dir, batch_filename)
    job_pathname   = '%s/%s' % (args.batch_dir, job_filename)
    batch_size     = len(batch)
    logging.info('Writing batch to %s' % batch_filename)

    output_handle = open(batch_pathname, "w")
    SeqIO.write(records, output_handle, "fasta")
    output_handle.close()

    output_handle = open(job_pathname, 'w')
    print >>output_handle, '#!/bin/bash'
    if args.prolog:
        print >>output_handle, args.prolog
    print >>output_handle, 'filename=$(mktemp --tmpdir)'
    print >>output_handle, 'cp %s $filename' % batch_filename
    command = ('%(igblast_dir)s/igblastn -ig_seqtype %(sequence_type)s -germline_db_V %(v_rep)s -germline_db_D %(d_rep)s -germline_db_J %(j_rep)s '\
            + '-organism %(organism)s -domain_system %(domain_system)s -num_threads %(num_threads)d '\
            + '-num_alignments_V %(num_v)d -num_alignments_D %(num_d)d -num_alignments_J %(num_j)d '\
            + '-auxiliary_data %(aux_file)s '\
            + '-outfmt "%(output_format)s"') % vars(args)
    if args.gene_subset:
        command += ' -germline_db_V_seqidlist ' + args.gene_subset
        command += ' -germline_db_D_seqidlist ' + args.gene_subset
        command += ' -germline_db_J_seqidlist ' + args.gene_subset
    if args.focus_on_v:
        command += ' -focus_on_V_segment'
    if args.show_translation:
        command += ' -show_translation'
    if '7' in args.output_format:
        command += ' -max_target_seqs ' + str(args.num_alignments)
    else:
        command += ' -num_alignments ' + str(args.num_alignments)
    command += ' -query %s >%s.parse.txt' % ('${filename}', '${filename}', )
    print >>output_handle, command
    print >>output_handle, 'cp ${filename}.parse.txt %s.parse.txt' % batch_filename
    print >>output_handle, 'rm ${filename} ${filename}.parse.txt'
    output_handle.close()
    os.chmod(job_pathname, os.stat(job_pathname).st_mode | stat.S_IXUSR)

    batch = list(itertools.islice(data_iter, args.batch_size))
    file_index += 1

boyddb.session.rollback()
