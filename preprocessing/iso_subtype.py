#!/usr/bin/env python

import logging
import argparse
from contextlib import contextmanager
import getpass
import sys

import psycopg2
import sqlalchemy
import sqlalchemy.orm

from Bio import SeqIO
from Bio import Seq
from Bio import pairwise2

import boydlib
import timblast

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.sql import select, text, or_, func

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
parser = argparse.ArgumentParser(description='')

parser.add_argument('demuxed_reads_tablename', metavar='demuxed-read-table',
        help='the demuxed read table, e.g. demuxed_reads_m54')
parser.add_argument('trimmed_reads_tablename', metavar='trimmed-read-table',
        help='the trimmed read table, e.g. trimmed_reads_m54')
parser.add_argument('parse_tablename', metavar='parse-table',
        help='the parse table, e.g. parsed_igh_igblast_m54')
parser.add_argument('isosubtype_tablename', metavar='isosubtype-table',
        help='the table to load iso-subtypes into, e.g. isosubtypes_m54')
parser.add_argument('--tolerance', type=int, default=2, help='how many extra bases 3\'to consider when looking for a match to the upstream fragment')
parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=10000,
        help='the number of sequences to insert at once')

boydlib.add_log_level_arg(parser)
boydlib.add_read_write_database_args(parser)

# parse the args
args = parser.parse_args()
boydlib.set_log_level(args)

password = getpass.getpass('Enter password for user %s to access database %s:' % (args.db_user, args.db_database))
def socket_connect():  # need to do this to connect to socket
        return psycopg2.connect(user=args.db_user, database=args.db_database, password=password)

# connect to the database
engine = sqlalchemy.create_engine('postgresql://', creator=socket_connect,
                                  implicit_returning=False)
meta = sqlalchemy.schema.MetaData(bind=engine)

# make sure the target table doesn't already exist
table_exists = engine.has_table(args.isosubtype_tablename)
if table_exists:
    logging.error('table %s already exists' % args.isosubtype_tablename)
    sys.exit(10)

# get the aux tables
demuxed_read_table = Table(args.demuxed_reads_tablename, meta, autoload=True)
trimmed_read_table = Table(args.trimmed_reads_tablename, meta, autoload=True)
parse_table = Table(args.parse_tablename, meta, autoload=True)
j_seg_table = Table('igh_j_segments', meta, autoload=True)
known_isosubtypes_table = Table('igh_isosubtypes', meta, autoload=True)
primer_upstream_table = Table('igh_const_upstream', meta, autoload=True)
replicates_table = Table('replicates', meta, autoload=True)

# load constant region primers by primer id
primer_upstreams_by_id = {}

query = select([primer_upstream_table.c.primer_set_id,
                primer_upstream_table.c.label,
                primer_upstream_table.c.sequence]).\
        select_from(primer_upstream_table)
for primer_set_id, upstream_label, upstream_sequence in engine.execute(query):
    if primer_set_id not in primer_upstreams_by_id:
        primer_upstreams_by_id[primer_set_id] = []
    primer_upstreams_by_id[primer_set_id].append((upstream_label, upstream_sequence))

# make target table
isosubtype_table = Table(args.isosubtype_tablename, meta,
    Column('read_isosubtype_id',  Integer, primary_key=True),
    Column('parsed_igh_igblast_id', Integer, ForeignKey(parse_table.name), unique=True, nullable=False),
    Column('isosubtype', String(16), ForeignKey(known_isosubtypes_table.c.name), nullable=False))

logging.info('%s' % known_isosubtypes_table.name)

logging.info('creating isosubtype table %s' % args.isosubtype_tablename)
isosubtype_table.create()
# get an insert statement for this table
isosubtype_insert = isosubtype_table.insert()


# do the following without database constraints
with out_constraints(engine, isosubtype_table):
    # batch items until we hit user given maximum batch size
    with batched(args.batch_size, lambda batch: engine.execute(isosubtype_insert, batch)) as batch:

        # stats
        reads_processed   = 0
        reads_subtyped    = 0
        reads_0_tolerance = 0
        reads_ignored     = 0
        subtype_counts  = {}

        query = select([replicates_table.c.primer_set_id,
                        trimmed_read_table.c.sequence,
                        parse_table.c.q_end,
                        parse_table.c.j_end,
                        func.length(j_seg_table.c.sequence),
                        parse_table.c.parsed_igh_igblast_id]).\
                select_from(replicates_table.\
                    join(demuxed_read_table).\
                    join(trimmed_read_table).\
                    join(parse_table).\
                    join(j_seg_table, j_seg_table.c.name == parse_table.c.j_segment))
        for primer_set_id, trimmed_sequence, parse_end, j_end, j_length, parse_id in engine.execute(query):
            reads_processed += 1

            # output status at 100000 intervals
            if reads_processed % 100000 == 0:
                if reads_processed == reads_ignored == 0:
                    logging.info('processed %s reads (%.2f%% typed, %.2f%% at zero tolerance)' % (reads_processed, 100.0 * reads_subtyped / (reads_processed - reads_ignored),
                                                                                                                100.0 * reads_0_tolerance / reads_subtyped))
                else:
                    logging.info('processed %s reads (%.2f%% typed, %.2f%% at zero tolerance)' % (reads_processed, 0.0, float('nan')))

            if primer_set_id not in primer_upstreams_by_id:
                reads_ignored += 1
            else:
                # only examine the sequence after the J with the tolerance
                post_j_seq = trimmed_sequence[parse_end + (j_length - j_end) - args.tolerance:]

                subtype = None
                # iterate over subtypes for the primer set
                for upstream_label, upstream_sequence in primer_upstreams_by_id[primer_set_id]:
                    # look for the sequence at the end +/- the tolerance
                    if upstream_sequence in post_j_seq[-len(upstream_sequence) - args.tolerance:]:
                        subtype = upstream_label

                        # check if it still would match at zero tolerance
                        if upstream_sequence in post_j_seq[-len(upstream_sequence):]:
                            reads_0_tolerance += 1

                        break   # stop after first match

                if subtype is not None:
                    reads_subtyped += 1

                    new_record = {'parsed_igh_igblast_id': parse_id,
                                  'isosubtype': subtype}

                    # add it to the batch
                    batch.add(new_record)

                    # gather stats
                    if subtype not in subtype_counts:
                        subtype_counts[subtype] = 0
                    subtype_counts[subtype] += 1

logging.info('ignored %d reads (no upstream region found for their primer sets)' % reads_ignored)
if reads_processed == reads_ignored:
    logging.info('processed %s reads (%.2f%% typed, %.2f%% at zero tolerance)' % (reads_processed, 0.0, float('nan')))
else:
    logging.info('processed %s reads (%.2f%% typed, %.2f%% at zero tolerance)' % (reads_processed, 100.0 * reads_subtyped / (reads_processed - reads_ignored),
                                                                                                100.0 * reads_0_tolerance / reads_subtyped))
logging.info('isosubtype counts:')
for subtypes in sorted(subtype_counts):
    logging.info('\t%s\t%d' % (subtypes, subtype_counts[subtypes]))
