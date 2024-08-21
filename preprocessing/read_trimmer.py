#!/usr/bin/env python

import logging
import sys
import pwd
import argparse
from collections import defaultdict
from contextlib import contextmanager
import re
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
parser = argparse.ArgumentParser(description='trim primers off both ends of the reads',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# the table to load and store reads from
parser.add_argument('source_table', metavar='source-table',
        help='the table to read the sequences from, e.g. demuxed_reads_m54')
parser.add_argument('dest_table', metavar='dest-table',
        help='the table of trimmed reads should be stored in, e.g. trimmed_reads_m54')
parser.add_argument('--no-rc', action='store_true', default=False, help='do not reverse comp the sequence')
parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=10000,
        help='the number of sequences to insert at once')
parser.add_argument('--forward-tolerance', type=int, default=10, help='how far away the primer can be from the start of the read')
parser.add_argument('--reverse-tolerance', type=int, default=10, help='how far away the primer can be from the end of the read')
parser.add_argument('--shorten-forward', '-s', type=int, default=0, metavar='N', help='remove this number of bases from the 3\' end of the forward primer before looking for a match, full primer length is still removed')
parser.add_argument('--shorten-reverse', '-S', type=int, default=0, metavar='N', help='remove this number of bases from the 5\' end of the reverse primer before looking for a match, full primer length is still removed')
parser.add_argument('--min-length', '-l', type=int, default=1, metavar='N', help='minimum length of the resulting sequence')
parser.add_argument('--test', nargs='?', metavar='N', const=10000, type=int, help='only process some of the sequences in a test run')

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

if args.test:
    logging.info('processing %d reads in a test run' % args.test)
if args.shorten_forward > 0:
    logging.info('ignoring %d base(s) of the 5\' end of the forward primer' % args.shorten_forward)
if args.shorten_reverse > 0:
    logging.info('ignoring %d base(s) of the 3\' end of the reverse primer' % args.shorten_reverse)

# get the aux tables
demuxed_table = Table(args.source_table, meta, autoload=True)
replicates_table = Table('replicates', meta, autoload=True)
primer_sets_table = Table('primer_sets', meta, autoload=True)
primers_table = Table('primers', meta, autoload=True)

# stats
no_hit_count = 0
single_hit_count = 0

# the filehandles to write out the no-hits to, opened as needed
no_hit_file = None

primer_set_seqs = defaultdict(lambda: ([], []))

logging.info('looking up all primers')
query = select([primers_table.c.primer_id, primers_table.c.direction, primers_table.c.sequence, primers_table.c.primer_set_id]).\
        where(primers_table.c.sequence is not None)
for primer_id, direction, sequence, primer_set_id in engine.execute(query):
    sequence = str(sequence)
    if direction == 'forward':
        if args.shorten_forward > 0:
            sequence = sequence[:-args.shorten_forward]
        primer_set_seqs[primer_set_id][0].append((sequence, primer_id))
    else:
        if args.shorten_reverse > 0:
            sequence = sequence[args.shorten_reverse:]
        primer_set_seqs[primer_set_id][1].append((sequence, primer_id))

# patterns for the full match
primer_pairs_re = {}
# patterns for matches that are off-by-one
primer_pairs_1mismatch_re = {}
# for each primer set
for primer_set_id in primer_set_seqs:
    # make an re for the exact forward match to the primer
    # we used the group name to remember the primer_id
    forward_patterns = []
    for sequence, primer_id in primer_set_seqs[primer_set_id][0]:
        forward_patterns.append('(?P<f%d>%s)' % (primer_id, sequence))

    # make an re for the exact reverse match to the primer
    reverse_patterns = []
    for sequence, primer_id in primer_set_seqs[primer_set_id][1]:
        reverse_patterns.append('(?P<r%d>%s)' % (primer_id, sequence))

    # construct the final pattern
    primer_pairs_re[primer_set_id] = re.compile('(?:' + '|'.join(forward_patterns) + ')' +
                                                '.*' +
                                                '(?:' + '|'.join(reverse_patterns) + ')')

    # build the pattens for the off-by-one forward matches
    forward_patterns = []
    for sequence, primer_id in primer_set_seqs[primer_set_id][0]:
        forward_subpatterns = []
        # build all the off-by-one matches
        for i in range(len(sequence)):
            forward_subpatterns.append(sequence[:i] + '.' + sequence[i + 1:])
        # assemble them into one group
        forward_patterns.append('(?P<f%d>' % primer_id + '|'.join(forward_subpatterns) + ')')

    # build the pattens for the off-by-one reverse matches
    reverse_patterns = []
    for sequence, primer_id in primer_set_seqs[primer_set_id][1]:
        reverse_subpatterns = []
        # build all the off-by-one matches
        for i in range(len(sequence)):
            reverse_subpatterns.append(sequence[:i] + '.' + sequence[i + 1:])
        reverse_patterns.append('(?P<r%d>' % primer_id + '|'.join(reverse_subpatterns) + ')')

    # construct the final pattern
    primer_pairs_1mismatch_re[primer_set_id] = re.compile('(?:' + '|'.join(forward_patterns) + ')' +
                                                          '.*' +
                                                          '(?:' + '|'.join(reverse_patterns) + ')')

trimmed_table = Table(args.dest_table, meta,
    Column('trimmed_read_id', Integer, primary_key=True),
    Column('demuxed_read_id', Integer, ForeignKey(demuxed_table.name), unique=True, nullable=False),
    Column('sequence', String, nullable=False),
    Column('forward_primer_id', Integer, ForeignKey(primers_table.c.primer_id)),
    Column('reverse_primer_id', Integer, ForeignKey(primers_table.c.primer_id)))
logging.info('creating trimmed read table %s' % args.dest_table)
trimmed_table.create()
# get an insert statement for this table
trimmed_insert = trimmed_table.insert()

# do the following without database constraints
with out_constraints(engine, trimmed_table):
    # batch items until we hit user given maximum batch size
    with batched(args.batch_size, lambda batch: engine.execute(trimmed_insert, batch)) as batch:
        # stats
        reads_processed  = 0
        trimmed_count    = 0
        off_by_one_count = 0
        too_short_count  = 0

        query = select([demuxed_table.c.demuxed_read_id, demuxed_table.c.sequence, replicates_table.c.primer_set_id]).\
                    select_from(demuxed_table.join(replicates_table))
        if args.test:
            query = query.limit(args.test)

        for demuxed_read_id, sequence, primer_set_id in engine.execute(query):
            reads_processed += 1

            # output status at 100000 intervals
            if reads_processed % 100000 == 0:
                logging.info('processed %s reads (%.2f%% trimmed)' % (reads_processed, 100.0 * trimmed_count / reads_processed))

            # rc the sequence unless asked not to
            if not args.no_rc:
                sequence = str(Seq(sequence).reverse_complement())
            # try matching the perfect matches
            match_count = 0
            match = primer_pairs_re[primer_set_id].search(sequence)
            if match:
                match_ids = [k for k, v in match.groupdict().items() if v is not None]
                assert len(match_ids) == 2, str(demuxed_read_id) + ' ' + str(match_ids) + str(primer_pairs_re[primer_set_id].pattern)
                # extract the id of the matching primer
                if match_ids[0].startswith('f'):
                    forward_primer = int(match_ids[0][1:])
                    assert match_ids[1].startswith('r')
                    reverse_primer = int(match_ids[1][1:])
                elif match_ids[0].startswith('r'):
                    reverse_primer = int(match_ids[0][1:])
                    assert match_ids[1].startswith('f')
                    forward_primer = int(match_ids[1][1:])
                else:
                    assert False

                # if the match is within the tolerances
                if match.start('f%d' % forward_primer) <= args.forward_tolerance and \
                len(sequence) - match.end('r%d' % reverse_primer) + args.shorten_reverse <= args.reverse_tolerance:

                    forward_end   = match.end('f%d' % forward_primer) + args.shorten_forward
                    reverse_start = match.start('r%d' % reverse_primer) - args.shorten_reverse

                    if reverse_start - forward_end >= args.min_length:
                        trimmed_count += 1
                        # add the record
                        new_record = {'demuxed_read_id': demuxed_read_id,
                                    'sequence': sequence[forward_end:reverse_start],
                                    'forward_primer_id': forward_primer,
                                    'reverse_primer_id': reverse_primer}
                        batch.add(new_record)
                    else:
                        too_short_count += 1
            else:
                # if no perfect match, look for off-by-one matches
                match = primer_pairs_1mismatch_re[primer_set_id].search(sequence)
                if match:
                    match_ids = [k for k, v in match.groupdict().items() if v is not None]
                    assert len(match_ids) == 2
                    # extract the id of the matching primer
                    if match_ids[0].startswith('f'):
                        forward_primer = int(match_ids[0][1:])
                        assert match_ids[1].startswith('r')
                        reverse_primer = int(match_ids[1][1:])
                    elif match_ids[0].startswith('r'):
                        reverse_primer = int(match_ids[0][1:])
                        assert match_ids[1].startswith('f')
                        forward_primer = int(match_ids[1][1:])
                    else:
                        assert False

                    # if the match is within the tolerances
                    if match.start('f%d' % forward_primer) <= args.forward_tolerance and \
                    len(sequence) - match.end('r%d' % reverse_primer) <= args.reverse_tolerance:

                        forward_end   = match.end('f%d' % forward_primer) + args.shorten_forward
                        reverse_start = match.start('r%d' % reverse_primer) - args.shorten_reverse

                        if reverse_start - forward_end >= args.min_length:
                            trimmed_count += 1
                            off_by_one_count += 1
                            # add the record
                            new_record = {'demuxed_read_id': demuxed_read_id,
                                          'sequence': sequence[forward_end:reverse_start],
                                          'forward_primer_id': forward_primer,
                                          'reverse_primer_id': reverse_primer}
                            batch.add(new_record)
                        else:
                            too_short_count += 1

logging.info('processed %d reads' % reads_processed)
logging.info('trimmed %d reads (%.2f%%)' % (trimmed_count, 100.0 * trimmed_count / reads_processed))
logging.info('%d trimmed reads had off-by-one matches (%f%%)' % (off_by_one_count, 100*float(off_by_one_count) / trimmed_count))
logging.info('%d trimmed reads were too short (<%d)' % (too_short_count, args.min_length))
