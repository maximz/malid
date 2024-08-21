#!/usr/bin/env python

import logging
import sys
import argparse
import gzip
import bz2
from contextlib import contextmanager
import psycopg2
import getpass

from Bio import SeqIO

import boydlib

import sqlalchemy
from sqlalchemy import Column, Integer, String, CheckConstraint
from sqlalchemy.sql import text

@contextmanager
def out_constraints(engine, *tables):
    table_names = tuple(map(str, tables))
    # get the commands to drop the constraints
    cmds_query = engine.execute(text("""
    SELECT 'ALTER TABLE "' || relname || '" DROP CONSTRAINT "' || conname || '"'
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname = 'public' AND relname IN :tables
    ORDER BY CASE WHEN contype='f' THEN 0 ELSE 1 END, contype, relname, conname;
    """), tables=table_names)
    drop_cmds = [i[0] for i in cmds_query]

    # get the commands to re-add the constraints
    cmds_query = engine.execute(text("""
    SELECT 'ALTER TABLE "' || relname || '" ADD CONSTRAINT "' || conname || '" ' || pg_get_constraintdef(pg_constraint.oid)
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname = 'public' AND relname IN :tables
    ORDER BY CASE WHEN contype='f' THEN 0 ELSE 1 END DESC, contype DESC, relname DESC, conname DESC;
    """), tables=table_names)
    add_cmds = [i[0] for i in cmds_query]

    # drop constraints
    engine.execute(' ; '.join(drop_cmds))

    try:
        yield
    finally:
        # and then add them back
        engine.execute(' ; '.join(add_cmds))

class batched(object):
    """Context manager that gathers items for batch processing.

    This context manager batches items until a maximum batch size has been
    reached, then processes the batch with the given processor function (that
    takes a list of items to process). At the end of the context, any
    remaining items are processed.
    """
    def __init__(self, max_size, processor):
        self.max_size = max_size
        self.processor = processor
    def __enter__(self):
        self.current_batch = []
        return self
    def add(self, item):
        if len(self.current_batch) >= self.max_size:
            self.processor(self.current_batch)
            self.current_batch = []
        self.current_batch.append(item)
    def __exit__(self, type, value, traceback):
        if len(self.current_batch) > 0:
            self.processor(self.current_batch)
            self.current_batch = []

# program options
parser = argparse.ArgumentParser(description='Load sequences into the database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--format', '-f', metavar='F', default='fastq',
        help='input file format')
parser.add_argument('reads_table', metavar='reads_table',
        help='the name of the reads table for these sequences, e.g. reads_m54')
parser.add_argument('sequence_files', metavar='seq_file', nargs='+',
        help='the sequences to load')
parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=10000,
        help='the number of sequences to insert at once')

boydlib.add_log_level_arg(parser)
boydlib.add_read_write_database_args(parser)

args = parser.parse_args()
boydlib.set_log_level(args)

password = getpass.getpass('Enter password for user %s to access database %s:' % (args.db_user, args.db_database))
def socket_connect():  # need to do this to connect to socket
        return psycopg2.connect(user=args.db_user, database=args.db_database, password=password)

# connect to the database
engine = sqlalchemy.create_engine('postgresql://', creator=socket_connect,
                                  implicit_returning=False)
meta   = sqlalchemy.schema.MetaData(bind=engine)

# make sure the target table doesn't already exist
table_exists = engine.has_table(args.reads_table)
if table_exists:
    logging.error('table %s already exists' % args.reads_table)
    sys.exit(10)

# make target table
reads_table = sqlalchemy.Table(args.reads_table, meta,
    Column('read_id',  Integer, primary_key=True),
    Column('label',    String(126), unique=True),
    Column('sequence', String, nullable=False),
    Column('phred33', String),
    CheckConstraint('phred33 IS NULL OR length(phred33) = length(sequence)'))
logging.info('creating read table %s' % args.reads_table)
reads_table.create()

# get an insert statement for this table
reads_insert = reads_table.insert()

load_count = 0

# do the following without constraints
with out_constraints(engine, reads_table):
    # batch items until we hit user given maximum batch size
    with batched(args.batch_size, lambda batch: engine.execute(reads_insert, batch)) as batch:
        # iterate over the files
        for filename in args.sequence_files:
            if filename == '-':
                file_handle = sys.stdin
                logging.info('processing stdin')
            elif filename.endswith('.gz'):
                file_handle = gzip.GzipFile(filename, 'r')
                logging.info('processing gzip\'d file %s' % filename)
            elif filename.endswith('.bz') or filename.endswith('.bz2'):
                file_handle = bz2.BZ2File(filename, 'r')
                logging.info('processing bzip2\'d file %s' % filename)
            else:
                file_handle = open(filename, 'rU')
                logging.info('processing file %s' % filename)

            # iterate over the sequences in the file
            for record in SeqIO.parse(file_handle, args.format):
                # is there a qual score
                if 'phred_quality' in record.letter_annotations:
                    phred33 = str(''.join(map(lambda n: chr(n + 33), record.letter_annotations['phred_quality'])))
                else:
                    phred33 = None
                # create a new seqeucne
                new_sequence = {'label': str(record.id),
                                'sequence': str(record.seq),
                                'phred33': phred33}

                load_count += 1

                # output status at 100000 intervals
                if load_count % 100000 == 0:
                    logging.info('loaded %s reads' % load_count)

                # add it to the batch
                batch.add(new_sequence)

logging.info('loaded %d sequences' % load_count)
