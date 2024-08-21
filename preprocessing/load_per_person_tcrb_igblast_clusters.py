#!/usr/bin/env python

import logging
import sys
import argparse
from contextlib import contextmanager
import psycopg2
import getpass

import boydlib

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from sqlalchemy.sql import select, text

@contextmanager
def out_constraints(engine, *tables):
    table_names = tuple(map(str, tables))
    # get the commands to drop the constraints
    cmds_query = engine.execute(text("""
    SELECT 'ALTER TABLE "' || nspname || '"."' || relname || '" DROP CONSTRAINT "' || conname || '"'
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname || '.' || relname IN :tables
    ORDER BY CASE WHEN contype='f' THEN 0 ELSE 1 END, contype, relname, conname;
    """), tables=table_names)
    drop_cmds = [i[0] for i in cmds_query]

    # get the commands to re-add the constraints
    cmds_query = engine.execute(text("""
    SELECT 'ALTER TABLE "' || nspname || '"."' || relname || '" ADD CONSTRAINT "' || conname || '" ' || pg_get_constraintdef(pg_constraint.oid)
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname || '.' || relname IN :tables
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

def main(arguments):
    # program options
    parser = argparse.ArgumentParser(description='Load clusters derived from using IgBLAST on IgH sequences',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('participant_label', metavar='participant',
            help='the label of the participant to cluster')
    # directory to store the batch info for the parse
    parser.add_argument('--schema', '-s', metavar='s', type=str, default='person_wise',
            help='the schema to create the table in')
    parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=10000,
            help='the number of sequences to insert at once')

    boydlib.add_log_level_arg(parser)
    boydlib.add_read_write_database_args(parser)

    args = parser.parse_args(arguments)
    boydlib.set_log_level(args)

    password = getpass.getpass('Enter password for user %s to access database %s:' % (args.db_user, args.db_database))
    def socket_connect():  # need to do this to connect to socket
            return psycopg2.connect(user=args.db_user, database=args.db_database, password=password)

    # connect to the database
    engine = sqlalchemy.create_engine('postgresql://', creator=socket_connect,
                                    implicit_returning=False)
    meta   = sqlalchemy.schema.MetaData(bind=engine)

    # make sure the target table doesn't already exist
    clones_exists  = engine.has_table('tcrb_clones_%s' % args.participant_label,schema=args.schema)
    members_exists = engine.has_table('tcrb_clone_members_%s' % args.participant_label, schema=args.schema)
    if clones_exists or members_exists:
        if clones_exists:
            logging.error('clones table tcrb_clones_%s already exists' % args.participant_label)
        if members_exists:
            logging.error('clone member table tcrb_clone_members_%s already exists' % args.participant_label)
        sys.exit(10)

    # get the table
    part_tcrb_table_name = 'participant_tcrb_%s' % args.participant_label
    part_tcrb_table = Table(part_tcrb_table_name, meta, schema=args.schema, autoload=True)

    logging.info('reading clusters from stdin')

    load_count = 0

    clone_ids = set()
    clone_members = set()

    for filename in sys.stdin:
        file_handle = open(filename.strip(), 'r')
        # pull out the clusters
        for cluster in file_handle:
            members = []
            for group in cluster.strip().split('\t'):
                members += group.split(',')     # any group with a , is split into its members
            members = map(int, members) # members are ints

            # the representative member is the first one give
            representative = members[0]

            # ass the new clone
            clone_ids.add(representative)

            # add the members
            for m in members:
                clone_members.add((representative, m))

    clones_table =  Table('tcrb_clones_%s' % args.participant_label, meta,
            Column('tcrb_clone_id', Integer, ForeignKey('%s.%s' % (args.schema, part_tcrb_table.c.part_tcrb_id)), primary_key=True),
            schema=args.schema)
    logging.info('creating clones table tcrb_clones_%s' % args.participant_label)
    clones_table.create()
    clones_insert = clones_table.insert()

    # do the following without database constraints
    with out_constraints(engine, clones_table):
        # batch items until we hit user given maximum batch size
        with batched(args.batch_size, lambda batch: engine.execute(clones_insert, batch)) as batch:
            for ident in clone_ids:
                batch.add({'tcrb_clone_id': ident})

    members_table = Table('tcrb_clone_members_%s' % args.participant_label, meta,
            Column('tcrb_clone_id', Integer, ForeignKey('%s.%s' % (args.schema, part_tcrb_table.c.part_tcrb_id)), nullable=False, primary_key=True),
            Column('part_tcrb_id', Integer, ForeignKey('%s.%s' % (args.schema, part_tcrb_table.c.part_tcrb_id)), nullable=False, primary_key=True),
            schema=args.schema)
    logging.info('creating clone member table tcrb_clone_members_%s' % args.participant_label,)
    members_table.create()
    members_insert = members_table.insert()

    # do the following without database constraints
    with out_constraints(engine, members_table):
        # batch items until we hit user given maximum batch size
        with batched(args.batch_size, lambda batch: engine.execute(members_insert, batch)) as batch:
            for clone, member in clone_members:
                batch.add({'tcrb_clone_id': clone,
                           'part_tcrb_id': member})

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
