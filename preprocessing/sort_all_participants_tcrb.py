#!/usr/bin/env python

import logging
import sys
import argparse
import bz2
from contextlib import contextmanager
from contextlib import nested
import psycopg2
import getpass
import tempfile
import pickle
import os

from Bio import SeqIO

import boydlib

import sqlalchemy
from sqlalchemy import Table, Column, Integer, Float, Boolean, String, ForeignKey, UniqueConstraint
from sqlalchemy.sql import select, text


segment_prefixes = ['pre_seq_nt_', 'fr1_seq_nt_', 'cdr1_seq_nt_', 'fr2_seq_nt_', 'cdr2_seq_nt_', 'fr3_seq_nt_', 'cdr3_seq_nt_', 'post_seq_nt_']

def get_refering_tables(tables, meta, schema=None):
    if type(tables) is str or type(tables) is unicode:
        tables = set([tables])

    meta.reflect(schema=schema)

    referencing_tables = set()

    for table in meta.tables.values():
        if table.name not in tables:
            for fk in table.foreign_keys:
                if fk.column.table.name in tables:
                    referencing_tables.add(table.name)

    return referencing_tables

def get_full_sequence(row, sequence_char):
    result = ''
    for prefix in segment_prefixes:
        if row[prefix + sequence_char] is not None:
            result += row[prefix + sequence_char]
    return result

def left_right_mask(query, mask, blanks=' '):
    if len(mask) == 0:
        return None
    else:
        assert len(query) == len(mask)
        left_trim  = len(mask) - len(mask.lstrip(blanks))
        right_trim = len(mask) - len(mask.rstrip(blanks))

        if left_trim == len(mask):
            return ''
        else:
            if right_trim == 0:
                return query[left_trim:]
            else:
                return query[left_trim:-right_trim]

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
    def __len__(self):
        return len(self.current_batch)
    def __exit__(self, type, value, traceback):
        if len(self.current_batch) > 0:
            self.processor(self.current_batch)
            self.current_batch = []

class Commiter:
    def __init__(self, engine, insert):
        self.engine = engine
        self.insert = insert
    def __call__(self, batch):
        self.engine.execute(self.insert, batch)

def main(arguments):
    # program options
    parser = argparse.ArgumentParser(description='sort IgH reads from a run into person specific table',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_label', metavar='run',
            help='the run to pull reads from')
    parser.add_argument('--participants', '-p', metavar='P', nargs='+',
            help='only sort the given participants from the run')
    parser.add_argument('--drop-replicate', '-d',  metavar='R',  nargs='+',
            help='don\'t sort from the given replicates')
    parser.add_argument('--schema', '-s', metavar='S', type=str, default='person_wise',
            help='the schema containing the part-tables')
    parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=10000,
            help='the number of sequences to insert at once')
    parser.add_argument('--presorted-dir', metavar='D',
            help='load pre-sorted data from the given temp directory')

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

    # get the tables
    runs_table = Table('runs', meta, autoload=True)
    barcode_map_table = Table('barcode_maps', meta, autoload=True)
    participants_table = Table('participants', meta, autoload=True)
    specimens_table = Table('specimens', meta, autoload=True)
    amplifications_table = Table('amplifications', meta, autoload=True)
    replicates_table = Table('replicates', meta, autoload=True)
    demuxed_reads_table = Table('demuxed_reads_%s' % args.run_label.lower(), meta, autoload=True)
    trimmed_reads_table = Table('trimmed_reads_%s' % args.run_label.lower(), meta, autoload=True)
    parsed_table = Table('parsed_tcrb_igblast_%s' % args.run_label.lower(), meta, autoload=True)
    barcode_maps_table = Table('barcode_maps', meta, autoload=True)

    # get the run id
    query = select([runs_table.c.run_id]).\
            select_from(runs_table).\
            where(runs_table.c.label == args.run_label)
    run_id = engine.execute(query).scalar()

    # get all the participant labels for the people
    query = select([participants_table.c.label]).\
            select_from(participants_table.\
                   join(specimens_table, specimens_table.c.participant_id == participants_table.c.participant_id).\
                   join(amplifications_table).\
                   join(replicates_table).\
                   join(barcode_maps_table).\
                   join(runs_table)).\
            where(runs_table.c.label == args.run_label).\
            distinct().order_by(participants_table.c.label)
    participant_labels = [r[0] for r in engine.execute(query)]

    # if list of participants to sort were given
    if args.participants:
        # check to make sure they are a subset of the ones in the run
        if set(args.participants).issubset(set(participant_labels)):
            participant_labels = args.participants
        else:
            # printing participants not in the run and exit
            logging.error('the following participants where given but are not in run %s' % args.run_label)
            for p in set(args.participants) - set(participant_labels):
                logging.error('\t' + p)
            return 10

    # if we're dropping some replicates
    if args.drop_replicate:
        # get all the replicate ids to drop
        replicate_labels = {}
        query = select([replicates_table.c.label, replicates_table.c.replicate_id]).\
                select_from(replicates_table).\
                where(replicates_table.c.label.in_(args.drop_replicate))
        for l, i in engine.execute(query):
            replicate_labels[i] = l

        # make sure all replicates are found
        if set(args.drop_replicate) != set(replicate_labels.values()):
            # print replicate labels that weren't found
            logging.error('the following replicates were given, to drop, but not found')
            for r in set(args.drop_replicate) - set(replicate_labels.values()):
                logging.error('\t' + r)
            return 10

        # set of ids to drop
        drop_replicate_ids = replicate_labels.keys()
        # number of reads that were dropped
        drop_rep_counts = {i: 0 for i in drop_replicate_ids}

    # check to make sure all the participants have part-tables
    missing_part_tables = []
    for p in participant_labels:
        part_tcrb_table_name = 'participant_tcrb_%s' % p
        if not engine.has_table(part_tcrb_table_name, schema=args.schema):
            missing_part_tables.append(part_tcrb_table_name)
    if len(missing_part_tables) > 0:
        logging.error('the following participants were given but do not have part-tables')
        for t in missing_part_tables:
            logging.error('\t' + t)
        return 10

    logging.info('sorting data from run %s (id %d) for people:', args.run_label, run_id)
    for p in participant_labels:
        logging.info('\t%s', p)

    # check to make sure there are no clones tables
    logging.info('checking for clone tables')
    refering_tables = get_refering_tables(set('participant_tcrb_%s' % p for p in participant_labels), meta, schema=args.schema)
    if len(refering_tables) > 0:
        logging.error('the following tables have foreign keys into the part-tables:')
        for i in refering_tables:
            logging.error('\t' + i)
        return 10

    # per-person read counts
    counts = {i: 0  for i in participant_labels}

    if args.presorted_dir:
        temp_dir = args.presorted_dir
    else:
        # if an already created temp directory is not given, make one
        # create temp directory to store working files
        temp_dir = tempfile.mkdtemp()
        logging.info('creating temp files in %s' % temp_dir)

        # batches for each person to be inserted after processing all the data
        batches = {i: bz2.BZ2File('%s/%s.bz2' % (temp_dir, i), 'wb') for i in participant_labels}

        # build the main query
        query = select([participants_table.c.label.label('participant_label'),
                        trimmed_reads_table.c.trimmed_read_id,
                        trimmed_reads_table.c.sequence.label('trimmed_sequence'),
                        trimmed_reads_table.c.forward_primer_id,
                        trimmed_reads_table.c.reverse_primer_id,
                        demuxed_reads_table.c.replicate_id,
                        parsed_table.c.parsed_tcrb_igblast_id,
                        parsed_table.c.v_segment,
                        parsed_table.c.d_segment,
                        parsed_table.c.j_segment,
                        parsed_table.c.v_score,
                        parsed_table.c.d_score,
                        parsed_table.c.j_score,
                        parsed_table.c.stop_codon,
                        parsed_table.c.v_j_in_frame,
                        parsed_table.c.productive,
                        parsed_table.c.strand,
                        parsed_table.c.n1_sequence,
                        parsed_table.c.n2_sequence,
                        parsed_table.c.n1_overlap,
                        parsed_table.c.n2_overlap,
                        parsed_table.c.q_start,
                        parsed_table.c.q_end,
                        parsed_table.c.v_start,
                        parsed_table.c.v_end,
                        parsed_table.c.d_start,
                        parsed_table.c.d_end,
                        parsed_table.c.j_start,
                        parsed_table.c.j_end,
                        parsed_table.c.pre_seq_nt_q,
                        parsed_table.c.pre_seq_nt_v,
                        parsed_table.c.pre_seq_nt_d,
                        parsed_table.c.pre_seq_nt_j,
                        parsed_table.c.fr1_seq_nt_q,
                        parsed_table.c.fr1_seq_nt_v,
                        parsed_table.c.fr1_seq_nt_d,
                        parsed_table.c.fr1_seq_nt_j,
                        parsed_table.c.cdr1_seq_nt_q,
                        parsed_table.c.cdr1_seq_nt_v,
                        parsed_table.c.cdr1_seq_nt_d,
                        parsed_table.c.cdr1_seq_nt_j,
                        parsed_table.c.fr2_seq_nt_q,
                        parsed_table.c.fr2_seq_nt_v,
                        parsed_table.c.fr2_seq_nt_d,
                        parsed_table.c.fr2_seq_nt_j,
                        parsed_table.c.cdr2_seq_nt_q,
                        parsed_table.c.cdr2_seq_nt_v,
                        parsed_table.c.cdr2_seq_nt_d,
                        parsed_table.c.cdr2_seq_nt_j,
                        parsed_table.c.fr3_seq_nt_q,
                        parsed_table.c.fr3_seq_nt_v,
                        parsed_table.c.fr3_seq_nt_d,
                        parsed_table.c.fr3_seq_nt_j,
                        parsed_table.c.cdr3_seq_nt_q,
                        parsed_table.c.cdr3_seq_nt_v,
                        parsed_table.c.cdr3_seq_nt_d,
                        parsed_table.c.cdr3_seq_nt_j,
                        parsed_table.c.post_seq_nt_q,
                        parsed_table.c.post_seq_nt_v,
                        parsed_table.c.post_seq_nt_d,
                        parsed_table.c.post_seq_nt_j,
                        parsed_table.c.pre_seq_aa_q,
                        parsed_table.c.fr1_seq_aa_q,
                        parsed_table.c.cdr1_seq_aa_q,
                        parsed_table.c.fr2_seq_aa_q,
                        parsed_table.c.cdr2_seq_aa_q,
                        parsed_table.c.fr3_seq_aa_q,
                        parsed_table.c.cdr3_seq_aa_q,
                        parsed_table.c.post_seq_aa_q,
                        parsed_table.c.insertions_pre,
                        parsed_table.c.insertions_fr1,
                        parsed_table.c.insertions_cdr1,
                        parsed_table.c.insertions_fr2,
                        parsed_table.c.insertions_cdr2,
                        parsed_table.c.insertions_fr3,
                        parsed_table.c.insertions_post,
                        parsed_table.c.deletions_pre,
                        parsed_table.c.deletions_fr1,
                        parsed_table.c.deletions_cdr1,
                        parsed_table.c.deletions_fr2,
                        parsed_table.c.deletions_cdr2,
                        parsed_table.c.deletions_fr3,
                        parsed_table.c.deletions_post]).\
                select_from(trimmed_reads_table.\
                        join(demuxed_reads_table).\
                        join(replicates_table).\
                        join(amplifications_table).\
                        join(specimens_table).\
                        join(participants_table, specimens_table.c.participant_id == participants_table.c.participant_id).\
                        join(parsed_table)).\
                where(participants_table.c.label.in_(participant_labels))

        # batch items until we hit user given maximum batch size
        sort_count = 0
        for row in engine.execute(query):
            # add drop if it's not in a dropped replicate or we're not dropping replicates at all
            if not args.drop_replicate or row['replicate_id'] not in drop_replicate_ids:
                new_row = dict(row)
                new_row['run_id'] = run_id
                participant_label = new_row['participant_label']
                del new_row['participant_label']

                full_q = get_full_sequence(row, 'q')
                full_v = get_full_sequence(row, 'v')
                full_d = get_full_sequence(row, 'd')
                full_j = get_full_sequence(row, 'j')

                new_row['v_sequence'] = left_right_mask(full_q, full_v)
                new_row['d_sequence'] = left_right_mask(full_q, full_d)
                new_row['j_sequence'] = left_right_mask(full_q, full_j)

                # output status at 100000 intervals
                sort_count += 1
                if sort_count % 100000 == 0:
                    logging.info('sorted %d reads' % sort_count)

                # add it to the batch
                pickle.dump(new_row, batches[participant_label], pickle.HIGHEST_PROTOCOL)

                counts[participant_label] += 1
            if args.drop_replicate and row['replicate_id'] in drop_replicate_ids:
                drop_rep_counts[row['replicate_id']] += 1

        logging.info('adding data to per-person tables')

        # close files
        for participant_label in participant_labels:
            batches[participant_label].close()

    # add the rows
    for participant_label in participant_labels:
        # open file
        temp_file = bz2.BZ2File('%s/%s.bz2' % (temp_dir, participant_label), 'rb')

        # if we are counting reads here
        if args.presorted_dir:
            counts[participant_label] += 1
        else:
            # we counted when sorting
            logging.info('adding %d reads to table for %s' % (counts[participant_label], participant_label))

        # get the table name
        part_tcrb_table_name = 'participant_tcrb_%s' % participant_label
        part_tcrb_table = Table(part_tcrb_table_name, meta, schema=args.schema, autoload=True)
        part_tcrb_insert = part_tcrb_table.insert()

        # drop constraints before adding rows
        with out_constraints(engine, part_tcrb_table):
            with batched(args.batch_size, lambda batch: engine.execute(part_tcrb_insert, batch)) as batch:
                while True:
                    try:
                        new_row = pickle.load(temp_file)
                        batch.add(new_row)
                    except EOFError:
                        break

        temp_file.close()

    # clean up temp files
    logging.info('removing temp files')
    for participant_label in participant_labels:
        os.remove('%s/%s.bz2' % (temp_dir, participant_label))
    os.rmdir(temp_dir)

    # summary counts
    logging.info('sorted %d reads', sum(counts.values()))
    for p in counts:
        logging.info('\t%s: %d', p, counts[p])

    # summary counts of dropped reads
    if args.drop_replicate:
        logging.info('dropped %d reads from following replicates', sum(drop_rep_counts.values()))
        for r in drop_rep_counts:
            logging.info('\t%s: %d', replicate_labels[r], drop_rep_counts[r])

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
