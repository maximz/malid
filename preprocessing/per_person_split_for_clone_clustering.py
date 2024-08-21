#!/usr/bin/env python

import logging
import argparse
import sys
import os
import psycopg2

import Bio.Alphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import boydlib

import sqlalchemy
from sqlalchemy import Table
from sqlalchemy.sql import select, and_

def main(arguments):
    # program options
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('participant_label', metavar='participant',
            help='the label of the participant to cluster')
    # directory to store the batch info for the parse
    parser.add_argument('batch_dir', metavar='batch_dir', help='the directory name to store the batch')
    # which locus should be parsed
    parser.add_argument('--locus', '-l', metavar='L', required=True,
            choices=['IgH', 'IgK', 'IgL', 'TCRA', 'TCRB', 'TCRG', 'TCRD'],
            help='which locus to parse: IgH, IgK, IgL')
    parser.add_argument('--specimens', metavar='S', nargs='+', type=str, help='only extract the reads from these specimens')
    parser.add_argument('--schema', '-s', metavar='S', type=str, default='person_wise',
            help='the schema to create the table in')
    parser.add_argument('--nospamfilter', action='store_true', help='do not filter by spam score')

    boydlib.add_log_level_arg(parser)
    boydlib.add_read_database_args(parser)

    # parse the args
    args = parser.parse_args(arguments)
    boydlib.set_log_level(args)

    # connect to the database
    def socket_connect():  # need to do this to connect to socket
            return psycopg2.connect(user=args.db_user, database=args.db_database)
    engine = sqlalchemy.create_engine('postgresql://', creator=socket_connect,
                                    implicit_returning=False)
    meta   = sqlalchemy.schema.MetaData(bind=engine)

    # get the table
    if args.locus == 'IgH':
        part_igh_table_name = 'participant_igh_%s' % args.participant_label
    elif args.locus == 'TCRB':
        part_igh_table_name = 'participant_tcrb_%s' % args.participant_label
    else:
        assert False

    part_igh_table = Table(part_igh_table_name, meta, schema=args.schema, autoload=True)
    replicates_table = Table('replicates', meta, autoload=True)
    amplifications_table = Table('amplifications', meta, autoload=True)
    specimens_table = Table('specimens', meta, autoload=True)

    primary_key = [i for i in part_igh_table.columns if i.primary_key][0]
    logging.info('using primary key %s' % primary_key)

    if os.path.exists(args.batch_dir):
        logging.error('could not create batch directory %s, directory (or file) already exists' % args.batch_dir)
        sys.exit(10)
    else:
        logging.info('creating batch directory %s' % args.batch_dir)
        os.mkdir(args.batch_dir)

    sequence_count = 0
    file_count = 0
    data = {}

    query = select([primary_key,
                    part_igh_table.c.v_segment,
                    part_igh_table.c.j_segment,
                    part_igh_table.c.cdr3_seq_nt_q]).\
            select_from(part_igh_table.join(replicates_table).\
                                       join(amplifications_table).\
                                       join(specimens_table))

    # if subsetting by specimen
    if args.specimens:
        query = query.where(and_(specimens_table.c.label.in_(args.specimens), part_igh_table.c.cdr3_seq_nt_q != None))
    else:
        query = query.where(part_igh_table.c.cdr3_seq_nt_q != None)

    if args.locus == 'IgH':
        if args.nospamfilter:
            pass
        else:
            query = query.where(part_igh_table.c.spam_score <= 0.0)

    for part_igh_id, v_seg, j_seg, cdr3_nt in engine.execute(query):
        sequence_count += 1

        cdr3_nt = cdr3_nt.replace('.', '').replace('-', '').upper()
        v_seg, _ = boydlib.split_allele(v_seg)
        j_seg, _ = boydlib.split_allele(j_seg)
        cdr3_len = len(cdr3_nt)

        if (v_seg, cdr3_len, j_seg) not in data:
            data[(v_seg, cdr3_len, j_seg)] = {}
        if cdr3_nt not in data[(v_seg, cdr3_len, j_seg)]:
            data[(v_seg, cdr3_len, j_seg)][cdr3_nt] = []
        data[(v_seg, cdr3_len, j_seg)][cdr3_nt].append('%d' % (part_igh_id))

    for (v_seg, cdr3_len, j_seg) in data:
        batch_filename = 'seq.%s.%d.%s.fasta' % (v_seg.replace('/', '_'), cdr3_len, j_seg)
        batch_pathname = '%s/%s' % (args.batch_dir, batch_filename)

        output_handle = open(batch_pathname, 'w')
        file_count += 1
        for cdr3_nt in data[(v_seg, cdr3_len, j_seg)]:
            record = SeqRecord(
                        Seq(cdr3_nt, Bio.Alphabet.generic_dna),
                        id=','.join(data[(v_seg, cdr3_len, j_seg)][cdr3_nt]),
                        description='')
            SeqIO.write(record, output_handle, 'fasta')

    print >>sys.stderr, 'processed %d sequences' % sequence_count
    print >>sys.stderr, 'created %d files' % file_count

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
