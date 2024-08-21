#!/usr/bin/env python

import sys
import pwd
import os
import logging
import getpass
import itertools
import re

import psycopg2
import sqlalchemy
import sqlalchemy.orm

import numpy

def add_log_level_arg(parser):
    """ add log level flags to the given args parser """
    parser.add_argument('--log-level', metavar='L', default='INFO', help='logging level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                 'debug', 'info', 'warning', 'error', 'critical'])

def add_read_database_args(parser):
    """ add read-only database connections flags to the give args parser """
    db_group = parser.add_argument_group('database arguments')
    db_group.add_argument('--db-database', metavar='DB', default='boydlab',
            help='the database to connect to')
    db_group.add_argument('--db-user', metavar='U', default=pwd.getpwuid(os.getuid()).pw_name,
            help='the username to use when connection to the database')
    db_group.add_argument('--db-pass', action='store_false', dest='db_no_pass', default=True,
            help='do not provide a password for the database')

def add_read_write_database_args(parser):
    """ add read and write database connections flags to the give args parser """
    db_group = parser.add_argument_group('database arguments')
    db_group.add_argument('--db-database', metavar='DB', default='boydlab',
            help='the database to connect to')
    db_group.add_argument('--db-user', metavar='U', default='boydlab_administrator',
            help='the username to use when connection to the database')
    db_group.add_argument('--db-no-pass', action='store_true', default=False,
            help='do not provide a password for the database')

def add_amp_selector_args(parser, suffix = '', action='store'):
    """ add amplification selector flags to the give args parser """
    if suffix == '':
        sel = parser.add_argument_group('amplification selector arguments').add_mutually_exclusive_group()
    else:
        sel = parser.add_argument_group('amplification selector arguments, group %s' % suffix).add_mutually_exclusive_group()

    sel.add_argument('--cohorts' + suffix, metavar='C', nargs='+', action=action,
                    help='select all amplifications that are from participants in the given cohorts')
    sel.add_argument('--participants' + suffix, metavar='P', nargs='+', action=action,
                    help='select all amplifications that are from the given participants')
    sel.add_argument('--collections' + suffix, metavar='C', nargs='+', action=action,
                    help='select all amplifications that are from the specimens in the given collection')
    sel.add_argument('--specimens' + suffix, metavar='S', nargs='+', action=action,
                    help='select all amplifications that are from the given specimens')
    sel.add_argument('--amplifications' + suffix, metavar='A', nargs='+', action=action,
                    help='select the given amplifications')

def add_read_set_args(parser, suffix='', action='store'):
    parser.add_argument('--read-sets' + suffix, metavar='R', nargs='+', action=action,
            help='restricts the reads to ones that are in one of the given read sets')

def set_log_level(args):
    """ set the log level using the args from argparse """
    log_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(level=log_level)

def create_engine_meta_session(args=None, echo=False, database=None, password=None, reflect=True):
    db_user     = args.db_user
    db_database = args.db_database
    db_no_pass  = args.db_no_pass
    log_level   = args.log_level

    if database:
        db_database = database

    print >>sys.stderr, 'connection to database %s' % db_database

    """ create a bunch of SQLAlchemy options given program flags """
    if db_no_pass:
        def connect():  # need to do this to connect to socket
                return psycopg2.connect(user=db_user, database=db_database)
    else:
        if password == None:
            password = getpass.getpass('Enter password for user %s to access database %s:' % (db_user, db_database))
        def connect():  # need to do this to connect to socket
                return psycopg2.connect(user=db_user, database=db_database, password=password)

    engine = sqlalchemy.create_engine('postgresql://',
                creator=connect, echo=True if log_level.upper() == 'DEBUG' or echo else False)
    meta = sqlalchemy.schema.MetaData(bind=engine)
    if reflect:
        meta.reflect()
    session = sqlalchemy.orm.sessionmaker(bind=engine, autoflush=False)()
    return engine, meta, session

def map_table(meta, table):
    """ map a table names """
    class C(sqlalchemy.Table): pass
    C.__name__ = table
    sqlalchemy.orm.Mapper(C, meta.tables[table])
    return C

def map_tables(meta, *tables):
    """ map a list of table names to a list of objects """
    classes = []
    for t in tables:
        C = map_table(meta, t)
        classes.append(C);
    return classes

def map_view(meta, view_name, key_name, key_type, foreign_constraints=[]):
    view_table = sqlalchemy.Table(view_name, meta,
            sqlalchemy.Column(key_name, key_type, primary_key=True), autoload=True)

    # add foreign key constraints
    for col_name, col_foreign in foreign_constraints:
        view_table.append_constraint(sqlalchemy.ForeignKeyConstraint([col_name], [col_foreign]))

    class View(object): pass
    sqlalchemy.orm.Mapper(View, view_table)
    return View

class BoydDB:
    """Object used to encapsulate a bunch of database information
    """
    def __init__(self, args, echo=False, tables=None, views=[]):
        # args is the result of argparse that can be used to pass db options
        if tables is not None:
            self.engine, self.meta, self.session = create_engine_meta_session(args, echo, reflect=False)
            default_tables = tables
        else:
            self.engine, self.meta, self.session = create_engine_meta_session(args, echo)
            default_tables = map(str, self.meta.tables.keys())

        #self.meta.reflect()

        # I don't think this list can be gotten from the database. The extra details need to be given
        default_views = views

        class Namespace : pass
        self.tables = Namespace()
        for name in default_tables:
            table_def = sqlalchemy.Table(name, self.meta, autoload=True, autoload_with=self.engine)
            class table(sqlalchemy.Table): pass
            sqlalchemy.orm.Mapper(table, self.meta.tables[name])
            setattr(self.tables, name, table)

        for v in default_views:
            name, key_name, key_type, others = v
            view = map_view(self.meta, name, key_name, key_type, others)
            setattr(self.tables, name, view)

class LocalDB:
    """Object used to encapsulate a bunch of database information
    """
    def __init__(self, args, echo=False):
        # args is the result of argparse that can be used to pass db options
        self.engine, self.meta, self.session = create_engine_meta_session(args, database=args.db_user, echo=echo)

        # list of tables to map
        default_tables = [str(t.name) for t in self.meta.sorted_tables]

        class Namespace : pass
        self.tables = Namespace()
        for name in default_tables:
            table = map_table(self.meta, name)
            setattr(self.tables, name, table)

def align_v_to_gapped(v_seqeunce, gapped_sequence):
    """
    Aligned an gapped V-sequence (as outputted by iHMMune-align) to an IMGT gapped V sequence
    """
    position_in_v = 0
    gapped_v = ''

    for nt in gapped_sequence:
        if position_in_v >= len(v_seqeunce):
            break
        if nt == '.':
            gapped_v += '.'
        else:
            gapped_v += v_seqeunce[position_in_v]
            position_in_v += 1
    return gapped_v + v_seqeunce[position_in_v:]

def remove_double_gaps(seq_a, seq_b, gap='-'):
    assert len(seq_a) == len(seq_b)
    new_a = ''
    new_b = ''
    for a, b in itertools.izip(seq_a, seq_b):
        if a == gap and b == gap:
            pass
        else:
            new_a += a
            new_b += b

    return new_a, new_b

def select_amps(boyddb, args=None, cohorts=None, participants=None, specimens=None, collections=None, amplifications=None):
    """
    Return a list of amplifications (either their id or label as determined by the ids flag).

    Amplifications are selected based on their name, the specimen their are from, the
    participants they are from, or a cohort of participants they are from. The parsed
    arguments from argparse can be used to pass the arguments.
    """

    if args:
        assert sum(map(lambda i: 0 if i == None or i == [] else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 0, 'no individual selectors can be given when args object is passed'

        cohorts = getattr(args, 'cohorts', None)
        participants = getattr(args, 'participants', None)
        specimens = getattr(args, 'specimens', None)
        collections = getattr(args, 'collections', None)
        amplifications = getattr(args, 'amplifications', None)
    else:
        assert sum(map(lambda i: 0 if i == None or i == [] else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 1, 'only one selector can be given'

    if cohorts:
        if type(cohorts) != list:
            cohorts = [cohorts]

        amps = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.amplifications.label).join(boyddb.tables.specimens).\
                join((boyddb.tables.participants, boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
                join(boyddb.tables.cohort_members).join(boyddb.tables.cohorts).filter(boyddb.tables.cohorts.label.in_(cohorts)).all()
    elif participants:
        if type(participants) != list:
            participants = [participants]

        amps = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.amplifications.label).join(boyddb.tables.specimens).\
                join((boyddb.tables.participants, boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
                filter(boyddb.tables.participants.label.in_(participants)).all()
    elif specimens:
        if type(specimens) != list:
            specimens = [specimens]

        amps = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.amplifications.label).join(boyddb.tables.specimens).\
                filter(boyddb.tables.specimens.label.in_(specimens)).all()
    elif collections:
        if type(collections) != list:
            collections = [collections]

        amps = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.amplifications.label).join(boyddb.tables.specimens).\
                join(boyddb.tables.collection_members).join(boyddb.tables.collections).filter(boyddb.tables.collections.label.in_(collections)).all()
    elif amplifications:
        if type(amplifications) != list:
            amplifications = [amplifications]

        amps = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.amplifications.label).filter(boyddb.tables.amplifications.label.in_(amplifications)).all()
    else:
        raise ValueError, 'no selectors gives'

    result = {}
    for id, label in amps:
        result[id] = label
    return result

def select_participants(boyddb, args=None, cohorts=None, participants=None, specimens=None, collections=None, amplifications=None):
    """
    Return a list of participants.

    Participants are selected based on their name, the specimens they contain, the
    amplifications they contain, or a cohort of participants. The parsed
    arguments from argparse can be used to pass the arguments.
    """

    if args:
        assert sum(map(lambda i: 0 if i == None else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 0, 'no individual selectors can be given when args object is passed'
        cohorts = getattr(args, 'cohorts', None)
        participants = getattr(args, 'participants', None)
        specimens = getattr(args, 'specimens', None)
        collections = getattr(args, 'collections', None)
        amplifications = getattr(args, 'amplifications', None)
    else:
        assert sum(map(lambda i: 0 if i == None else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 1, 'only one selector can be given'

    if cohorts:
        if type(cohorts) != list:
            cohorts = [cohorts]

        parts = boyddb.session.query(boyddb.tables.participants.participant_id, boyddb.tables.participants.label).\
                join(boyddb.tables.cohort_members).join(boyddb.tables.cohorts).filter(boyddb.tables.cohorts.label.in_(cohorts)).all()
    elif participants:
        if type(participants) != list:
            participants = [participants]

        parts = boyddb.session.query(boyddb.tables.participants.participant_id, boyddb.tables.participants.label).\
                filter(boyddb.tables.participants.label.in_(participants)).all()
    elif specimens:
        if type(specimens) != list:
            specimens = [specimens]

        parts = boyddb.session.query(boyddb.tables.participants.participant_id, boyddb.tables.participants.label).\
                join((boyddb.tables.specimens, boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
                filter(boyddb.tables.specimens.label.in_(specimens)).all()
    elif collections:
        if type(collections) != list:
            collections = [collections]

        parts = boyddb.session.query(boyddb.tables.participants.participant_id, boyddb.tables.participants.label).\
                join((boyddb.tables.specimens, boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
                join(boyddb.tables.collection_members).join(boyddb.tables.collections).\
                filter(boyddb.tables.collections.label.in_(collections)).all()
    elif amplifications:
        if type(amplifications) != list:
            amplifications = [amplifications]

        parts = boyddb.session.query(boyddb.tables.participants.participant_id, boyddb.tables.participants.label).\
                join((boyddb.tables.specimens, boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
                join(boyddb.tables.amplifications).\
                filter(boyddb.tables.amplifications.label.in_(amplifications)).all()
    else:
        assert 1 == 0

    result = {}
    for id, label in parts:
        result[id] = label
    return result

def select_specimens(boyddb, args=None, cohorts=None, participants=None, specimens=None, collections=None, amplifications=None):
    """
    Return a list of specimens.

    Specimens are slected based on a participant, their name, the
    amplifications they contain, or a cohort of participants. The parsed
    arguments from argparse can be used to pass the arguments.
    """

    if args:
        assert sum(map(lambda i: 0 if i == None else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 0, 'no individual selectors can be given when args object is passed'
        cohorts = getattr(args, 'cohorts', None)
        participants = getattr(args, 'participants', None)
        specimens = getattr(args, 'specimens', None)
        collections = getattr(args, 'collections', None)
        amplifications = getattr(args, 'amplifications', None)
    else:
        assert sum(map(lambda i: 0 if i == None else 1,
            [cohorts, participants, specimens, collections, amplifications])) == 1, 'only one selector can be given'

    if cohorts:
        if type(cohorts) != list:
            cohorts = [cohorts]

        spec = boyddb.session.query(boyddb.tables.specimens.specimen_id, boyddb.tables.specimens.label).\
               join((boyddb.tables.participants, boyddb.tables.specimens.participant_id == boyddb.tables.participants.participant_id)).\
               join(boyddb.tables.cohort_members).join(boyddb.tables.cohorts).filter(boyddb.tables.cohorts.label.in_(cohorts)).all()
    elif participants:
        if type(participants) != list:
            participants = [participants]

        spec = boyddb.session.query(boyddb.tables.specimens.specimen_id, boyddb.tables.specimens.label).\
               join((boyddb.tables.participants, boyddb.tables.specimens.participant_id == boyddb.tables.participants.participant_id)).\
               filter(boyddb.tables.participants.label.in_(participants)).all()
    elif specimens:
        if type(specimens) != list:
            specimens = [specimens]

        spec = boyddb.session.query(boyddb.tables.specimens.specimen_id, boyddb.tables.specimens.label).\
               filter(boyddb.tables.specimens.label.in_(specimens)).all()
    elif collections:
        if type(collections) != list:
            collections = [collections]

        spec = boyddb.session.query(boyddb.tables.specimens.specimen_id, boyddb.tables.specimens.label).\
               join(boyddb.tables.collection_members).join(boyddb.tables.collections).\
                               filter(boyddb.tables.collections.label.in_(collections)).all()
    elif amplifications:
        if type(amplifications) != list:
            amplifications = [amplifications]

        spec = boyddb.session.query(boyddb.tables.specimens.specimen_id, boyddb.tables.specimens.label).\
               join(boyddb.tables.amplifications).\
               filter(boyddb.tables.amplifications.label.in_(amplifications)).all()
    else:
        assert 1 == 0

    result = {}
    for id, label in spec:
        result[id] = label
    return result

def get_all_participant_attrs_keys(boyddb, participants):
    # build the query
    query = boyddb.session.query(sqlalchemy.distinct(boyddb.tables.participant_attributes.key)).\
            join(boyddb.tables.participants).\
            filter(boyddb.tables.participants.participant_id.in_(participants))
    return ['label', 'alternative_label', 'age', 'sex', 'diagnosis'] + [str(key) for key, in query]

def participant_attr_to_amp(boyddb, amps, attributes):
    if type(attributes) != list:
        attributes = [attributes]
    results = {}

    # build the base query
    query = boyddb.session.query(boyddb.tables.amplifications.amplification_id).\
            join(boyddb.tables.specimens).join((boyddb.tables.participants,
                boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
            filter(boyddb.tables.amplifications.amplification_id.in_(amps))

    for attribute in attributes:
        if attribute == 'label':
            query = query = query.add_columns(boyddb.tables.participants.label)
        elif attribute == 'alternative_label':
            query = query = query.add_columns(boyddb.tables.participants.alternative_label)
        elif attribute == 'age':
            query = query = query.add_columns(boyddb.tables.participants.age)
        elif attribute == 'sex':
            query = query = query.add_columns(boyddb.tables.participants.sex)
        elif attribute == 'diagnosis':
            query = query = query.add_columns(boyddb.tables.participants.diagnosis)
        else:
            # each attribute look up adds an aliased copy of participant_attributes
            table_alias = sqlalchemy.orm.aliased(boyddb.tables.participant_attributes)

            # add the value to the SELECT clause
            query = query.add_columns(table_alias.value)

            # add the outer join with the key as a join condition
            query = query.outerjoin(table_alias,
                    sqlalchemy.and_(boyddb.tables.participants.participant_id == table_alias.participant_id,
                                    table_alias.key == attribute))

    for row in query:
        amp_id, values = row[0], row[1:]
        results[amp_id] = values

    return results

def amplification_mutation_counts(boyddb, amps, spam_filter=True, filter_by=None):
    """
    """
    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_mutations,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_sequence).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    mut_counts = dict([(i, numpy.array([])) for i in amps])
    v_lengths  = dict([(i, numpy.array([])) for i in amps])

    count = 0
    for amp_id, mut_count, v_sequence in query:
        mut_counts[amp_id] = numpy.append(mut_counts[amp_id], int(mut_count))
        v_lengths[amp_id] = numpy.append(v_lengths[amp_id], int(len(v_sequence)))
        count += 1
    logging.info('got mutation counts for %d reads' % count)

    return mut_counts, v_lengths

def amplification_read_pid(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Get the percent id of the reads from the given amplifications.

    Returns a dictionary for each of the given amps that contains a list of the observed
    percent identities of each of the reads in that amplification. The reads can be filtered
    by only selectng reads that are present in all of the given read_sets.
    """
    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id, boyddb.tables.ihmmune_pid.v_pid).\
            join(boyddb.tables.ihmmune_pid).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    pids = dict([(i, numpy.array([])) for i in amps])

    count = 0
    for amp_id, pid in query:
        pids[amp_id] = numpy.append(pids[amp_id], float(pid))
        count += 1
    logging.info('got percent ids for %d reads' % count)

    return pids

def apply_read_set_filters(boyddb, query, filter_by, table_name="run_ihmmune_scored_frcdr_reads"):
    """
    A set of read_set filters to the given query.

    Returns a new query with that restricts the reads to only those that belong
    to all of the given read_sets.
    """

    table = getattr(boyddb.tables, table_name)

    if type(filter_by) != list:
        filter_by = [filter_by]
    for f in filter_by:
        read_set_alias = sqlalchemy.orm.aliased(boyddb.tables.read_sets)
        read_set_member_alias = sqlalchemy.orm.aliased(boyddb.tables.read_set_members)
        query = query.join(read_set_member_alias, read_set_member_alias.read_id == table.read_id).\
                        join(read_set_alias).\
                        filter(read_set_alias.label == f)
    return query

def amplification_n1_n2_nt(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Returns a dictionary for each of the given amps that contains a list of the
    N1 and N2 nucleotides of each of the reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.n1_sequence,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.n2_sequence).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    amp_n1n2_nt = dict([(i, [numpy.array([]), numpy.array([])]) for i in amps])

    for amp_id, n1_nt, n2_nt in query:
        amp_n1n2_nt[amp_id][0] = numpy.append(amp_n1n2_nt[amp_id][0], n1_nt)
        amp_n1n2_nt[amp_id][1] = numpy.append(amp_n1n2_nt[amp_id][1], n2_nt)

    return amp_n1n2_nt

def amplification_n1n2_nt_length(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Get the N1 and N2 length of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains a list of the length in
    nucleotides of the N1 and N2 region of each of the reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    amp_n1n2_nt = amplification_n1_n2_nt(boyddb, amps, spam_filter, filter_by)

    amp_n1n2_len = dict([(i, [numpy.array([]), numpy.array([])]) for i in amp_n1n2_nt])

    for amp_id in amp_n1n2_len:
        n1_lens, n2_lens = amp_n1n2_nt[amp_id]
        for n1_nt in n1_lens:
            if n1_nt != None:
                amp_n1n2_len[amp_id][0] = numpy.append(amp_n1n2_len[amp_id][0], len(n1_nt))
        for n2_nt in n2_lens:
            if n2_nt != None:
                amp_n1n2_len[amp_id][1] = numpy.append(amp_n1n2_len[amp_id][1], len(n2_nt))

    return amp_n1n2_len


def amplification_n1n2_nt_mean_length(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Get the mean N1 and N2 length of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains the mean length, in
    nucleotides, of the N1 and N2 regions of all reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    amp_n1n2_len = amplification_n1n2_nt_length(boyddb, amps, spam_filter, filter_by)

    result = dict([(i, [numpy.mean(amp_n1n2_len[i][0]), numpy.mean(amp_n1n2_len[i][1])]) for i in amp_n1n2_len])
    return result

def amplification_mean_v_d_j_chewback(boyddb, amps, gapped_vs, spam_filter=True, filter_by=None):
    """
    """

    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_segment,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.d_segment,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.j_segment,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_sequence,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.d_sequence,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.j_sequence).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    amp_v_chewback = dict([(i, numpy.array([])) for i in amps])
    amp_d_left_chewback = dict([(i, numpy.array([])) for i in amps])
    amp_d_right_chewback = dict([(i, numpy.array([])) for i in amps])
    amp_j_chewback = dict([(i, numpy.array([])) for i in amps])

    for amp_id, v_name, d_name, j_name, v_seq, d_seq, j_seq in query:
        if v_name:
            aligned_v = align_v_to_gapped(v_seq, gapped_vs[v_name])
            assert len(aligned_v) == len(gapped_vs[v_name])
            v_chewback = len(aligned_v) - len(aligned_v.rstrip('.'))
            amp_v_chewback[amp_id] = numpy.append(amp_v_chewback[amp_id], v_chewback)
        if d_name:
            d_left_chewback = len(d_seq) - len(d_seq.lstrip('.'))
            d_right_chewback = len(d_seq) - len(d_seq.rstrip('.'))

            amp_d_left_chewback[amp_id]  = numpy.append(amp_d_left_chewback[amp_id], d_left_chewback)
            amp_d_right_chewback[amp_id] = numpy.append(amp_d_right_chewback[amp_id], d_right_chewback)
        if j_name:
            j_chewback = len(j_seq) - len(j_seq.lstrip('.'))

            amp_j_chewback[amp_id] = numpy.append(amp_j_chewback[amp_id], j_chewback)

    results = {}
    for amp_id in amps:
        results[amp_id] = (numpy.mean(amp_v_chewback[amp_id]),
                           numpy.mean(amp_d_left_chewback[amp_id]),
                           numpy.mean(amp_d_right_chewback[amp_id]),
                           numpy.mean(amp_j_chewback[amp_id]))
    return results
def mutation_hotspots(boyddb, amps, gapped_vs, spam_filter=True, filter_by=None):
    """
    """
    wa_hotspot_pattern   = re.compile('(A|T)A')
    tw_hotspot_pattern   = re.compile('T(A|T)')
    rgyw_hotspot_pattern = re.compile('(A|G)G(C|T)(A|T)')
    wrcy_hotspot_pattern = re.compile('(A|T)(A|G)C(C|T)')

    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_segment,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_sequence).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    wa_a_counts = dict([(i, 0) for i in amps])
    wa_c_counts = dict([(i, 0) for i in amps])
    wa_g_counts = dict([(i, 0) for i in amps])
    wa_t_counts = dict([(i, 0) for i in amps])

    tw_a_counts = dict([(i, 0) for i in amps])
    tw_c_counts = dict([(i, 0) for i in amps])
    tw_g_counts = dict([(i, 0) for i in amps])
    tw_t_counts = dict([(i, 0) for i in amps])

    rgyw_a_counts = dict([(i, 0) for i in amps])
    rgyw_c_counts = dict([(i, 0) for i in amps])
    rgyw_g_counts = dict([(i, 0) for i in amps])
    rgyw_t_counts = dict([(i, 0) for i in amps])

    wrcy_a_counts = dict([(i, 0) for i in amps])
    wrcy_c_counts = dict([(i, 0) for i in amps])
    wrcy_g_counts = dict([(i, 0) for i in amps])
    wrcy_t_counts = dict([(i, 0) for i in amps])

    for amp_id, v_name, v_seq in query:
        if v_name:
            aligned_v = align_v_to_gapped(v_seq, gapped_vs[v_name])
            gapped_v, aligned_v = remove_double_gaps(gapped_vs[v_name], aligned_v, gap='.')

            gapped_v = gapped_v.upper()
            aligned_v = aligned_v.upper()

            for match in wa_hotspot_pattern.finditer(gapped_v):
                if aligned_v[match.start(0) + 1] == 'A':
                    wa_a_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'C':
                    wa_c_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'G':
                    wa_g_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'T':
                    wa_t_counts[amp_id] += 1

            for match in tw_hotspot_pattern.finditer(gapped_v):
                if aligned_v[match.start(0)] == 'A':
                    tw_a_counts[amp_id] += 1
                elif aligned_v[match.start(0)] == 'C':
                    tw_c_counts[amp_id] += 1
                elif aligned_v[match.start(0)] == 'G':
                    tw_g_counts[amp_id] += 1
                elif aligned_v[match.start(0)] == 'T':
                    tw_t_counts[amp_id] += 1

            for match in rgyw_hotspot_pattern.finditer(gapped_v):
                if aligned_v[match.start(0) + 1] == 'A':
                    rgyw_a_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'C':
                    rgyw_c_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'G':
                    rgyw_g_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 1] == 'T':
                    rgyw_t_counts[amp_id] += 1

            for match in wrcy_hotspot_pattern.finditer(gapped_v):
                if aligned_v[match.start(0) + 2] == 'A':
                    wrcy_a_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 2] == 'C':
                    wrcy_c_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 2] == 'G':
                    wrcy_g_counts[amp_id] += 1
                elif aligned_v[match.start(0) + 2] == 'T':
                    wrcy_t_counts[amp_id] += 1

    results = {}
    for amp_id in amps:
        results[amp_id] = (wa_a_counts[amp_id],
                           wa_c_counts[amp_id],
                           wa_g_counts[amp_id],
                           wa_t_counts[amp_id],
                           tw_a_counts[amp_id],
                           tw_c_counts[amp_id],
                           tw_g_counts[amp_id],
                           tw_t_counts[amp_id],
                           rgyw_a_counts[amp_id],
                           rgyw_c_counts[amp_id],
                           rgyw_g_counts[amp_id],
                           rgyw_t_counts[amp_id],
                           wrcy_a_counts[amp_id],
                           wrcy_c_counts[amp_id],
                           wrcy_g_counts[amp_id],
                           wrcy_t_counts[amp_id])

    return results

def amplification_mutation_track(boyddb, amps, gapped_vs, spam_filter=True, filter_by=None):
    """
    """

    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_segment,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.v_sequence).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    amp_mutation_counts = dict([(i, numpy.array([0 for k in range(325)])) for i in amps])
    amp_mutation_totals = dict([(i, numpy.array([0 for k in range(325)])) for i in amps])

    for amp_id, v_name, v_seq in query:
        if v_name:
            aligned_v = align_v_to_gapped(v_seq, gapped_vs[v_name])
            for base, position in itertools.izip(aligned_v, itertools.count()):
                if base != '.' and base != 'n' and base != 'N':
                    amp_mutation_totals[amp_id][position] += 1
                    if base == 'A' or base == 'C' or base == 'G' or base == 'T':
                        amp_mutation_counts[amp_id][position] += 1

    results = {}
    for amp_id in amps:
        results[amp_id] = (amp_mutation_counts[amp_id],
                           amp_mutation_totals[amp_id])
    return results

def amplification_cdr3_aa(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    """
    Get the amino acids of the CDR3 of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains a list of the
    amino acids of the CDR3 of each of the reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """
    query = boyddb.session.query(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id,
                                 boyddb.tables.run_ihmmune_scored_frcdr_reads.cdr3_aa).\
            filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.amplification_id.in_(amps))
    if spam_filter:
        query = query.filter(boyddb.tables.run_ihmmune_scored_frcdr_reads.spam_score < 0.0)
    if filter_by:
        query = apply_read_set_filters(boyddb, query, filter_by)

    amp_cdr3_aa = dict([(i, numpy.array([])) for i in amps])

    for amp_id, cdr3_aa in query:
        if remove_hash:
            cdr3_aa = cdr3_aa.replace('#', '')
        amp_cdr3_aa[amp_id] = numpy.append(amp_cdr3_aa[amp_id], cdr3_aa)

    return amp_cdr3_aa

def amplification_cdr3_charge(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    amp_cdr3_aa = amplification_cdr3_aa(boyddb, amps, spam_filter, filter_by, remove_hash)

    cdr3_charge = dict([(i, numpy.array([])) for i in amp_cdr3_aa])

    for amp_id in amp_cdr3_aa:
        for aa in amp_cdr3_aa[amp_id]:
            charge = 0
            for r in aa:
                if r != '*' and r != 'X':
                    charge += aa_charge(r)
            cdr3_charge[amp_id] = numpy.append(cdr3_charge[amp_id], charge)

    return cdr3_charge

def amplification_cdr3_hydropathy(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    amp_cdr3_aa = amplification_cdr3_aa(boyddb, amps, spam_filter, filter_by, remove_hash)

    cdr3_hydropathy = dict([(i, numpy.array([])) for i in amp_cdr3_aa])

    for amp_id in amp_cdr3_aa:
        for aa in amp_cdr3_aa[amp_id]:
            hydropathy = 0
            length = 0
            for r in aa:
                if r != '*' and r != 'X':
                    hydropathy += aa_hydropathy(r)
                    length += 1
            if length > 0:
                cdr3_hydropathy[amp_id] = numpy.append(cdr3_hydropathy[amp_id], hydropathy / length)
            else:
                cdr3_hydropathy[amp_id] = numpy.append(cdr3_hydropathy[amp_id], 0.0)

    return cdr3_hydropathy

def amplification_cdr3_mean_charge(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    cdr3_charge = amplification_cdr3_charge(boyddb, amps, spam_filter, filter_by, remove_hash)

    result = dict([(i, numpy.mean(cdr3_charge[i])) for i in cdr3_charge])
    return result

def amplification_cdr3_mean_hydropathy(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    cdr3_hydropathy = amplification_cdr3_hydropathy(boyddb, amps, spam_filter, filter_by, remove_hash)

    result = dict([(i, numpy.mean(cdr3_hydropathy[i])) for i in cdr3_hydropathy])
    return result

def amplification_cdr3_aa_length(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    """
    Get the CDR3 length of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains a list of the length in
    amino acids of the CDR3 of each of the reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    amp_cdr3_aa = amplification_cdr3_aa(boyddb, amps, spam_filter, filter_by, remove_hash)

    cdr3_aa_len = dict([(i, numpy.array([])) for i in amp_cdr3_aa])

    for amp_id in amp_cdr3_aa:
        for aa in amp_cdr3_aa[amp_id]:
            cdr3_aa_len[amp_id] = numpy.append(cdr3_aa_len[amp_id], len(aa))

    return cdr3_aa_len

def amplification_cdr3_aa_mean_length(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    """
    Get the mean CDR3 length of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains the mean length, in
    amino acids, of the CDR3 of all reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    cdr3_aa_len = amplification_cdr3_aa_length(boyddb, amps, spam_filter, filter_by, remove_hash)

    result = dict([(i, numpy.mean(cdr3_aa_len[i])) for i in cdr3_aa_len])
    return result

def amplification_cdr3_aa_median_length(boyddb, amps, spam_filter=True, filter_by=None, remove_hash=True):
    """
    Get the median CDR3 length of the reads from the give amplifications.

    Returns a dictionary for each of the given amps that contains the median length, in
    amino acids, of the CDR3 of all reads in those amplifications. The reads can be
    filtered by only selectng reads that are present in all of the given read_sets.
    """

    cdr3_aa_len = amplification_cdr3_aa_length(boyddb, amps, spam_filter, filter_by, remove_hash)

    result = dict([(i, numpy.median(cdr3_aa_len[i])) for i in cdr3_aa_len])
    return result

def amplification_mean_pid(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Returns the mean percent identity of the reads in the given amplification

    Returns a dictionary for each of the given amps that contains the mean percent
    identity of the reads in the given amplification. The reads can be filtered
    by only selectng reads that are present in all of the given read_sets.
    """

    pids = amplification_read_pid(boyddb, amps, spam_filter, filter_by)

    result = dict([(i, numpy.mean(pids[i])) for i in pids])
    return result

def amplification_median_pid(boyddb, amps, spam_filter=True, filter_by=None):
    """
    Returns the median percent identity of the reads in the given amplification

    Returns a dictionary for each of the given amps that contains the median percent
    identity of the reads in the given amplification. The reads can be filtered
    by only selectng reads that are present in all of the given read_sets.
    """

    pids = amplification_read_pid(boyddb, amps, spam_filter, filter_by)

    result = dict([(i, numpy.median(pids[i])) for i in pids])
    return result

def amplification_alt_participant_label(boyddb, amps):
    query = boyddb.session.query(boyddb.tables.amplifications.amplification_id, boyddb.tables.participants.alternative_label).\
            join(boyddb.tables.specimens).join((boyddb.tables.participants,
                boyddb.tables.participants.participant_id == boyddb.tables.specimens.participant_id)).\
            filter(boyddb.tables.amplifications.amplification_id.in_(amps))

    amp_alt_label = dict()
    for amp, alt_label in query:
        amp_alt_label[amp] = alt_label
    return amp_alt_label

def split_allele(segment):
    if '*' in segment:
        gene, allele = segment.split('*')
        return gene, allele
    return segment, None

def remove_allele(segment):
    return split_allele(segment)[0]

def vdjnn_fron_parse(v, d, j, n1, n2, d_sequence, allele=True):
    if allele:
        return v, d, j, n1, n2
    else:
        return remove_allele(v), remove_allele(d), remove_allele(j), n1, n2

def vbj_fron_parse(v, d, j, n1, n2, d_sequence, allele=True):
    """ return a V, B, J tuple for the give iHMMune parse """
    v = str(v) if v != None else None
    d = str(d) if d != None else None
    j = str(j) if j != None else None

    if n1 == None:
        n1 = ''
    else:
        n1 = str(n1)

    if d_sequence == None:
        d_sequence = ''
    else:
        d_sequence = str(d_sequence)

    if n2 == None:
        n2 = ''
    else:
        n2 = str(n2)

    b = n1 + d_sequence + n2
    b = b.upper().replace('.', '').replace('-', '').replace('(', '').replace(')', '')

    if allele:
        return v, b, j
    else:
        return remove_allele(v), b, remove_allele(j)

def split_ihmmune_v_call(all_v_segments):
    if '[or' in all_v_segments:
        first, rest = all_v_segments.split('[or')
        if rest.endswith(']'):
            rest = rest[:-1]
        if 'or' in rest:
            rest = rest.split('or')
        else:
            rest = [rest]
        vs = [first] + rest
        for i in range(len(vs)):
            if vs[i].strip().endswith('[indels]'):
                vs[i] = vs[i].strip()[:-len('[indels]')].strip()
            else:
                vs[i] = vs[i].strip()
        return vs
    else:
        return [str(clean_ihmmune_v_call(all_v_segments.strip()))]

def clean_ihmmune_v_call(v_segment):
    if v_segment.startswith('IGHV1_VH1'):
        v_segment = v_segment[6:]
    elif v_segment.startswith('IGHV3_VH3'):
        v_segment = v_segment[6:]
    elif v_segment.startswith('IGHV7_VH7'):
        v_segment = v_segment[6:]
    elif v_segment.startswith('IGHV4_VH4'):
        v_segment = v_segment[6:]
    elif v_segment.startswith('humIGHV181'):
        v_segment = v_segment[3:]

    if v_segment.endswith('(L1)') or v_segment.endswith('(L2)') or v_segment.endswith('(L3)') or\
            v_segment.endswith('(L4)') or v_segment.endswith('(L5)'):
        v_segment = v_segment[:-4]
    elif v_segment.endswith('(L1)]') or v_segment.endswith('(L2)]') or v_segment.endswith('(L3)]') or\
            v_segment.endswith('(L4)]') or v_segment.endswith('(L5)]'):
        v_segment = v_segment[:-5]

    if v_segment.endswith('(L4)/04'):
        v_segment = v_segment[:-7]

    # turn not founds into Nones
    if v_segment.startswith('NA') or v_segment == 'State':
        v_segment = None
    if v_segment != None and '(hum' in v_segment:
        v_segment = v_segment.split('(hum')[0]
    if v_segment != None and '_' in v_segment:
        v_segment = v_segment.split('_')[0]
    return v_segment

def coin_score(list_of_reps, order):
    """
    Calculate the coin score of the given order for a list of replcates.

    Each replicate is a dictionary of ids and their counts
    """

    if type(list_of_reps) == dict:
        list_of_reps = list_of_reps.values()

    coincidence_pairs = 0
    possible_pairs    = 0

    sizes = map(lambda r: sum(r.values()), list_of_reps)

    # calculate the number of possible tuples
    for combination in itertools.combinations(sizes, order):
        product = 1
        for size in combination:
            product *= size
        possible_pairs += product

    if possible_pairs == 0:
        return float('nan')

    for combination in itertools.combinations(list_of_reps, order):
        # split the first rep
        first_rep = combination[0]
        rest_reps = combination[1:]

        # look for match to elements from the first rep in the others
        for elem in first_rep:
            #print 'looking for %s in other reps' % elem
            coin_product = first_rep[elem]
            for rep in rest_reps:
                if elem in rep:
                    coin_product *= rep[elem]
                else:
                    coin_product = 0
                    break   # can stop looking, results will be zero
            #print 'coincidences with %s: %d' % (elem, coin_product)

            #if coin_product != 0:
            #    print 'found %s in order %d' % (elem, order)
            coincidence_pairs += coin_product

    return float(coincidence_pairs) / possible_pairs

def read_amp_result_table(stream):
    headers = stream.next()
    headers = headers.strip().split('\t')
    assert headers[0] == 'amplifications'
    assert headers[1] == 'read_set_spec'
    table = AmpResultsTable(headers[2:])
    for row in stream:
        column_values = row.strip().split('\t')
        table.add_row(column_values)
    return table

class AmpResultsTable:
    def __init__(self, other_headers):
        self.headers = ['amplifications', 'read_set_spec'] + other_headers
        self.column_name_to_index = dict([(h, i) for h, i in zip(self.headers, range(len(self.headers)))])
        self.columns = [[] for i in self.headers]
        self.amp_readset_to_row = {}

    def get_column(self, name):
        return self.columns[self.column_name_to_index[name]]

    def add_row(self, row):
        assert len(row) == len(self.columns)
        key = (row[0], row[1])
        assert key not in self.amp_readset_to_row
        self.amp_readset_to_row[key] = len(self.columns[0])
        for element, column in zip(row, self.columns):
            column.append(element)

    def add_column(self, new_column_names, default='NA'):
        if type(new_column_names) != list:
            new_column_names = [new_column_names]
        for c in new_column_names:
            assert c not in self.column_name_to_index
        # add new columns
        self.headers =  self.headers + new_column_names
        # update column name to index, just remake the whole thing
        self.column_name_to_index = dict([(h, i) for h, i in zip(self.headers, range(len(self.headers)))])
        # add default values
        row_count = len(self.columns[0])
        self.columns = self.columns +  [[default] * row_count for i in new_column_names]

    def set_amp_columm_values(self, column_names, amp_values):
        row_count = len(self.columns[0])
        for row in range(row_count):
            if self.columns[0][row] in amp_values:
                assert len(column_names) == len(amp_values[self.columns[0][row]])
                for n, v in zip(column_names, amp_values[self.columns[0][row]]):
                    self.columns[self.column_name_to_index[n]][row] = str(v)
    def set_amp_readset_column_values(self, column_names, amp_readset_values):
        if type(column_names) != list and type(column_names) != tuple:
            column_names = [column_names]
        row_count = len(self.columns[0])
        for row in range(row_count):
            amp = self.columns[0][row]
            read_set = self.columns[1][row]
            if (amp, read_set) in amp_readset_values:
                values = amp_readset_values[(amp, read_set)]
                if type(values) != list:
                    values = [values]
                assert len(column_names) == len(values), '%s does not have the same len as %s' % (str(column_names), str(values))
                for n, v in zip(column_names, values):
                    self.columns[self.column_name_to_index[n]][row] = str(v)

    def __str__(self):
        string = '\t'.join(self.headers)
        for row_index in range(len(self.columns[0])):
            row = [columns[row_index] for columns in self.columns]
            string += '\n' + '\t'.join(row)
        return string

amino_acid_charge_table = \
    {'A':  0,
     'R': +1,
     'N':  0,
     'D': -1,
     'C':  0,
     'E': -1,
     'Q':  0,
     'G':  0,
     'H':  0,
     'I':  0,
     'L':  0,
     'K': +1,
     'M':  0,
     'F':  0,
     'P':  0,
     'S':  0,
     'T':  0,
     'W':  0,
     'Y':  0,
     'V':  0}

def aa_charge(aa):
    aa = aa.upper()
    return amino_acid_charge_table[aa]

amino_acid_hydropathy_table = \
    {'A':  1.8,
     'R': -4.5,
     'N': -3.5,
     'D': -3.5,
     'C':  2.5,
     'E': -3.5,
     'Q': -3.5,
     'G': -0.4,
     'H': -3.2,
     'I':  4.5,
     'L':  3.8,
     'K': -3.9,
     'M':  1.9,
     'F':  2.8,
     'P': -1.6,
     'S': -0.8,
     'T': -0.7,
     'W': -0.9,
     'Y': -1.3,
     'V':  4.2}

def aa_hydropathy(aa):
    aa = aa.upper()
    return amino_acid_hydropathy_table[aa]

def mean_aa_hydropathy(aa, no_data=float('nan')):
    hydropathy = 0.0
    length = 0
    for r in aa:
        if r != '*' and r != 'X' and r != '#':
            hydropathy += aa_hydropathy(r)
            length += 1
    if length == 0:
        return no_data
    else:
        return hydropathy / length

def protein_charge(aa):
    charge = 0
    for r in aa:
        if r != '*' and r != 'X' and r != '#':
            charge += aa_charge(r)
    return charge

def protein_positive_charge(aa):
    charge = 0
    for r in aa:
        if r != '*' and r != 'X' and r != '#':
            c = aa_charge(r)
            if c > 0:
                charge += c
    return charge

def protein_negative_charge(aa):
    charge = 0
    for r in aa:
        if r != '*' and r != 'X' and r != '#':
            c = aa_charge(r)
            if c < 0:
                charge += c
    return charge

###############################################################################
# Parser for CD-HIT cluster files

# holds the current file handle and the line
class HandleLine:
    def __init__(self, handle, next_line):
        self.handle = handle
        self.next_line = next_line

# generator for the list of clusters
def cdhit_cluster_file(handle):
    # if given a string, use it as a filename
    if type(handle) == str:
        handle = open(filename, 'r')

    # holds the shared data
    next_line = handle.readline()
    handle_line = HandleLine(handle, next_line)

    while handle_line.next_line != '':
        assert handle_line.next_line.startswith('>')

        cluster_name = handle_line.next_line[1:-1]

        handle_line.next_line = handle_line.handle.readline()

        cluster = CDHitCluster(cluster_name, handle_line)
        yield cluster

        # skip lines until you get to the next cluster
        while handle_line.next_line != '' and not handle_line.next_line.startswith('>'):
            handle_line.next_line = handle_line.handle.readline()

# holds the cluster info
class CDHitCluster:
    def __init__(self, name, handle_line):
        self.name = name
        # data shared with the generator
        self.handle_line = handle_line

        self.representative = CDHitClusterMember(handle_line.next_line)
        self.rep_shown = False

        self.handle_line.next_line = self.handle_line.handle.readline()

    def __str__(self):
        return '%s (with rep %s)' % (self.name, self.representative)

    def __iter__(self):
        return self

    # iterate over the cluster members
    def next(self):
        if not self.rep_shown:
            self.rep_shown = True
            return self.representative

        if self.handle_line.next_line == '' or self.handle_line.next_line.startswith('>'):
            raise StopIteration

        result = CDHitClusterMember(self.handle_line.next_line)
        self.handle_line.next_line = self.handle_line.handle.readline()

        return result

# a cluster member
class CDHitClusterMember:
    def __init__(self, line):
        # pull out the member number
        member_number, rest = line.split('\t')
        self.member_number = int(member_number)

        # extract the length
        length, name, rest = rest.split(' ', 2)
        assert length.endswith('nt,')
        self.length = int(length[:-3])

        # extract the name
        assert name.startswith('>') and name.endswith('...')
        self.name = name[1:-3]

        # extract the match percent
        if rest == '*\n':
            self.match_percent = 1.0
        else:
            assert rest.startswith('at +/') and rest.endswith('%\n')
            self.match_percent = float(rest[5:-2])/100.0
    def __str__(self):
        return '%02d, %dnt, %s, %f' % (self.member_number, self.length, self.name, self.match_percent)
