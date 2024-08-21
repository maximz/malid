#!/usr/bin/env python

import logging
import sys
import argparse
import psycopg2

import boydlib

import sqlalchemy
from sqlalchemy import Table
from sqlalchemy.sql import select

def main(arguments):
    # program options
    parser = argparse.ArgumentParser(description='which runs contrain data for the given participant',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_label', metavar='run', nargs='+',
            help='the run to query')

    boydlib.add_log_level_arg(parser)
    boydlib.add_read_database_args(parser)

    args = parser.parse_args(arguments)
    boydlib.set_log_level(args)

    def socket_connect():  # need to do this to connect to socket
            return psycopg2.connect(user=args.db_user, database=args.db_database)

    # connect to the database
    engine = sqlalchemy.create_engine('postgresql://', creator=socket_connect,
                                    implicit_returning=False)
    meta   = sqlalchemy.schema.MetaData(bind=engine)

    # get the tables
    runs_table = Table('runs', meta, autoload=True)
    participants_table = Table('participants', meta, autoload=True)
    specimens_table = Table('specimens', meta, autoload=True)
    amplifications_table = Table('amplifications', meta, autoload=True)
    replicates_table = Table('replicates', meta, autoload=True)
    barcode_maps_table = Table('barcode_maps', meta, autoload=True)

    # get all the replicates for the person
    query = select([participants_table.c.label]).\
            select_from(participants_table.\
                   join(specimens_table, specimens_table.c.participant_id == participants_table.c.participant_id).\
                   join(amplifications_table).\
                   join(replicates_table).\
                   join(barcode_maps_table).\
                   join(runs_table)).\
            where(runs_table.c.label.in_(args.run_label)).\
            distinct().order_by(participants_table.c.label)
    for r in engine.execute(query):
        print r[0]

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
