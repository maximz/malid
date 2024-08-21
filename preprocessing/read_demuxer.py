#!/usr/bin/env python

import logging
import sys
import argparse
from contextlib import contextmanager
import psycopg2
import getpass

from Bio.Seq import Seq

import boydlib

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from sqlalchemy.sql import select, text

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
parser = argparse.ArgumentParser(description='Assign reads to replicate and trim off the barcodes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# run id
parser.add_argument('run', metavar='run', help='the run to identify the barcode map to use')
# the tables to load and store reads from
parser.add_argument('source_table', metavar='source-table',
        help='the table to read the sequences from, e.g. reads_m54')
parser.add_argument('dest_table', metavar='dest-table',
        help='the table the de-multiplexed reads should be stored in, e.g. demuxed_reads_m54')
parser.add_argument('--rc', action='store_true', default=False, help='reverse comp the sequence')
parser.add_argument('--forward-skip', '-f', metavar='N', type=int, default=4,
        help='number of bases to skip from the 5\' end before looking for the barcode')
parser.add_argument('--reverse-skip', '-r', metavar='N', type=int, default=0,
        help='number of bases to skip from the 3\' end before looking for the barcode')
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
table_exists = engine.has_table(args.dest_table)
if table_exists:
    logging.error('table %s already exists' % args.dest_table)
    sys.exit(10)

# get the aux tables
reads_table = Table(args.source_table, meta, autoload=True)
replicates_table = Table('replicates', meta, autoload=True)
runs_table = Table('runs', meta, autoload=True)
barcode_maps_table = Table('barcode_maps', meta, autoload=True)
forward_barcodes_table = Table('forward_barcodes', meta, autoload=True)
reverse_barcodes_table = Table('reverse_barcodes', meta, autoload=True)


# the barcode map
barcode_map = {}
# set of sizes of the forward/reverse barcode sequences
forward_barcode_sizes = set()
reverse_barcode_sizes = set()
# what ranks were found
ranks = set()
# replicate id to replicate label mapping
replicate_id_to_label = {}
# used to store replicate counts by label
mapped_replicate_counts = {}
# trim amounts as a (forward, reverse) tuple
trim_amounts = {}

barcode_query = select([forward_barcodes_table.c.sequence,
                        forward_barcodes_table.c.trim_amount,
                        reverse_barcodes_table.c.sequence,
                        reverse_barcodes_table.c.trim_amount,
                        barcode_maps_table.c.rank,
                        barcode_maps_table.c.replicate_id,
                        replicates_table.c.label]).\
        select_from(barcode_maps_table.\
            join(forward_barcodes_table).\
            join(reverse_barcodes_table).\
            join(replicates_table).\
            join(runs_table)).\
        where(runs_table.c.label == args.run)
for forward_seq, forward_trim, reverse_seq, reverse_trim, rank, replicate_id, replicate_label in engine.execute(barcode_query):
    # update sizes
    forward_barcode_sizes.add(len(forward_seq))
    reverse_barcode_sizes.add(len(reverse_seq))

    ranks.add(rank)

    # add the replicate id and label count
    replicate_id_to_label[replicate_id] = str(replicate_label)
    mapped_replicate_counts[str(replicate_label)] = 0

    # store barcode map
    assert (str(forward_seq), str(reverse_seq)) not in barcode_map
    barcode_map[(str(forward_seq), str(reverse_seq))] = int(replicate_id)

    # store trim amounts
    assert (str(forward_seq), str(reverse_seq)) not in trim_amounts
    trim_amounts[(str(forward_seq), str(reverse_seq))] = (forward_trim, reverse_trim)

assert len(forward_barcode_sizes) == 1 and len(reverse_barcode_sizes) == 1, 'multiple barcode sizes not currenty supported'
assert ranks == set([1]), 'only rank 1 barcode maps are currently supported'

forward_barcode_size = forward_barcode_sizes.pop()
reverse_barcode_size = reverse_barcode_sizes.pop()

logging.info('reading reads from table %s' % args.source_table)
# make target table
demuxed_table = Table(args.dest_table, meta,
    Column('demuxed_read_id',  Integer, primary_key=True),
    Column('read_id', Integer, ForeignKey(reads_table.name), unique=True, nullable=False),
    Column('replicate_id', Integer, ForeignKey(replicates_table.name), index=True, nullable=False),
    Column('sequence', String, nullable=False),
    Column('trim_start', Integer),
    Column('trim_for', Integer))
logging.info('creating demuxed read table %s' % args.dest_table)
demuxed_table.create()
# get an insert statement for this table
demuxed_insert = demuxed_table.insert()

# do the following without database constraints
with out_constraints(engine, demuxed_table):
    # batch items until we hit user given maximum batch size
    with batched(args.batch_size, lambda batch: engine.execute(demuxed_insert, batch)) as batch:
        # stats
        reads_processed = 0
        reads_mapped    = 0

        reads_query = select([reads_table.c.read_id, reads_table.c.sequence])
        for read_id, sequence in engine.execute(reads_query):
            reads_processed += 1

            # rc the sequences if asked
            if args.rc:
                sequence = str(Seq(sequence).reverse_complement())

            # output status at 100000 intervals
            if reads_processed % 100000 == 0:
                logging.info('processed %s reads (%.2f%% mapped)' % (reads_processed, 100.0 * reads_mapped / reads_processed))

            # get the possible barcode sequence
            possible_forward_barcode = sequence[args.forward_skip:args.forward_skip + forward_barcode_size]
            possible_reverse_barcode = sequence[-reverse_barcode_size - args.reverse_skip:-args.reverse_skip if args.reverse_skip != 0 else None]
            # check if we have a match
            if (possible_forward_barcode, possible_reverse_barcode) in barcode_map:
                reads_mapped += 1

                replicate_id = barcode_map[(possible_forward_barcode, possible_reverse_barcode)]

                forward_trim, reverse_trim = trim_amounts[(possible_forward_barcode, possible_reverse_barcode)]

                forward_trim += args.forward_skip
                reverse_trim += args.reverse_skip

                # new demuxed_read row
                new_record = {'read_id': read_id,
                            'replicate_id': replicate_id,
                            'sequence': str(sequence[forward_trim:-reverse_trim if reverse_trim> 0 else None]),
                            'trim_start': forward_trim + 1,
                            'trim_for': len(sequence) - reverse_trim - forward_trim}
                # add it to the batch
                batch.add(new_record)

                # increment the replicate count
                mapped_replicate_counts[replicate_id_to_label[barcode_map[(possible_forward_barcode, possible_reverse_barcode)]]] += 1

logging.info('process %d reads' % reads_processed)
logging.info('mapped %d reads (%.2f%%)' % (reads_mapped, 100.0 * reads_mapped / reads_processed))
logging.info('replicate counts:')
total = 0
for replicate_label in sorted(mapped_replicate_counts):
    logging.info('\t%s\t%d' % (replicate_label, mapped_replicate_counts[replicate_label]))
    total += mapped_replicate_counts[replicate_label]
assert total == reads_mapped
