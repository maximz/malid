#!/usr/bin/env python

"""Divide IgH reads from a run into participant-specific tables.

Depends on python2 (as with rest of boydlib)

Install:
    $ pip install --user --upgrade pandas joblib

Usage, for example for sorting M378:

    First run `add_participant_table.py` as usual to create IgH participant tables in the `person_wise` schema.

    Then, in a tmux or screen, run:

        $ ./sort_all_participants_igh_fast.py M378 \
            --read-chunk-size 100000 \
            --write-chunk-size 5000 \
            --num-jobs 50 &> m378_igh_sort.out;

    Monitor:
        $ tail -f m378_igh_sort.out;
        $ htop -u $USER;

    If something goes wrong:
        - see undo_sort.py
        - manually run the printed sql commands to re-add constraints
"""


import sys
import argparse
import psycopg2
import getpass
import time
import pandas as pd
import boydlib
from joblib import Parallel, delayed, parallel_backend


import sqlalchemy
from sqlalchemy import Table
from sqlalchemy.sql import select, text
from sqlalchemy.pool import NullPool


segment_prefixes = [
    "pre_seq_nt_",
    "fr1_seq_nt_",
    "cdr1_seq_nt_",
    "fr2_seq_nt_",
    "cdr2_seq_nt_",
    "fr3_seq_nt_",
    "cdr3_seq_nt_",
    "post_seq_nt_",
]


def left_right_mask(query, mask, blanks=" "):
    if len(mask) == 0:
        return None
    else:
        assert len(query) == len(mask)
        left_trim = len(mask) - len(mask.lstrip(blanks))
        right_trim = len(mask) - len(mask.rstrip(blanks))

        if left_trim == len(mask):
            return ""
        else:
            if right_trim == 0:
                return query[left_trim:]
            else:
                return query[left_trim:-right_trim]


def plan_constraint_drop_add(engine, table_names, schema):
    # must be a tuple for sqlalchemy text() to wrap this right
    table_names = tuple("%s.%s" % (schema, tblname) for tblname in table_names)

    # get the commands to drop the constraints
    cmds_query = engine.execute(
        text(
            """
    SELECT 'ALTER TABLE "' || nspname || '"."' || relname || '" DROP CONSTRAINT "' || conname || '"'
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname || '.' || relname IN :tables
    ORDER BY CASE WHEN contype='f' THEN 0 ELSE 1 END, contype, relname, conname;
    """
        ),
        tables=table_names,
    )
    drop_cmds = [i[0] for i in cmds_query]

    # get the commands to re-add the constraints
    cmds_query = engine.execute(
        text(
            """
    SELECT 'ALTER TABLE "' || nspname || '"."' || relname || '" ADD CONSTRAINT "' || conname || '" ' || pg_get_constraintdef(pg_constraint.oid)
    FROM pg_constraint
    INNER JOIN pg_class ON conrelid = pg_class.oid
    INNER JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
    WHERE nspname || '.' || relname IN :tables
    ORDER BY CASE WHEN contype='f' THEN 0 ELSE 1 END DESC, contype DESC, relname DESC, conname DESC;
    """
        ),
        tables=table_names,
    )
    add_cmds = [i[0] for i in cmds_query]

    return drop_cmds, add_cmds


def drop_constraints(engine, drop_cmds):
    # drop constraints
    if len(drop_cmds) > 0:
        engine.execute(" ; ".join(drop_cmds))


def readd_constraints(engine, add_cmds):
    # and then add them back
    if len(add_cmds) > 0:
        engine.execute(" ; ".join(add_cmds))


def socket_connect(user, db, password):
    return psycopg2.connect(user=user, database=db, password=password)


def make_part_table_name(participant_label):
    return "participant_igh_%s" % (participant_label)


def progress_report(participant_label, chunk_id, msg, start_time, previous_time):
    new_time = time.time()
    print "\t".join(
        [
            participant_label,
            "Chunk %d" % chunk_id,
            msg,
            "%0.0f seconds elapsed (%0.0f seconds total)"
            % (new_time - previous_time, new_time - start_time,),
        ]
    )

    # necessary in multiprocessing:
    sys.stdout.flush()
    sys.stderr.flush()

    return new_time


# from io import StringIO
from io import BytesIO  # python2
import csv


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Faster than using df.to_sql(method='multi')

    Execute SQL statement inserting data
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
    # https://stackoverflow.com/a/55495065/130164

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """

    # gets a DBAPI connection that can provide a cursor
    with conn.connection.cursor() as cur:
        s_buf = BytesIO()  # StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '"{}"."{}"'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

def run_participant(
    participant_label,
    run_id,
    run_label,
    user,
    db,
    password,
    schema,
    read_chunksize,
    write_chunksize,
    cols_to_map_to_int
):
    # open a database connection in this process
    engine = sqlalchemy.create_engine(
        "postgresql://",
        creator=lambda: socket_connect(user, db, password),
        poolclass=NullPool,
        # echo_pool='debug',
        implicit_returning=False,
        connect_args={'connect_timeout': 24 * 60 * 60} # 24 hour timeout
    )


    # Manage connection manually, rather than relying on sqlalchemy's "engine execute" abstraction.

    # Why? Because we suspect connections are leaking in the pd.read_sql_query(..., chunksize) iterator when passing an Engine,
    # which caused us to hit the Postgres maximum connection limit, set to 100 for us.

    # In that particular circumstance, at the time we hit the 100 cap:
    # - We were reading in chunk #44 for one participant
    # - We were writing chunk #1 for two participants
    # - We were reading in chunk #1 for 18 participants
    # - Also in the master process we had an Engine that had been used several times for gathering participant labels and table constraints

    # The suspicion is that the read iterator may have had 44 open connections for that first participant,
    # or that something is funky in the engine connection management at the stage of to_sql() writing
    # where pandas runs table create() --> exists() --> has_table() to check for table existence.

    # Whatever the cause, each process's engine defaults to pool size of up to 5.
    # So in the circumstance described above, we could have hit the cap of 100 that way: 21 * 5 > 100

    # We can also use create_engine(..., poolclass=NullPool) to enforce this further.
    # And enable echo_pool='debug' to watch connection pool status.

    # See also https://stackoverflow.com/a/51242577/130164
    with engine.connect() as conn:

        meta = sqlalchemy.schema.MetaData(bind=conn)
        query = make_select_query(meta, run_label, participant_label)
        # print(str(query))

        itime = time.time()
        cur_time = itime

        # get the table name
        part_igh_table_name = make_part_table_name(participant_label)

        for idx, df in enumerate(
            pd.read_sql_query(query, conn, chunksize=read_chunksize), start=1
        ):
            cur_time = progress_report(
                participant_label,
                idx,
                "Loaded %s, memory usage %d bytes"
                % (str(df.shape), df.memory_usage(index=True, deep=True).sum()),
                itime,
                cur_time,
            )

            df["run_id"] = run_id
            full_sequences = {}
            for sequence_type in ["q", "v", "d", "j"]:
                col_names = [prefix + sequence_type for prefix in segment_prefixes]
                # sum these columns for each row (result is a series with shape df.shape[0])
                full_sequences[sequence_type] = df[col_names].fillna("").sum(axis=1)
            full_sequences = pd.DataFrame(full_sequences)
            if full_sequences.shape[0] != df.shape[0]:
                raise ValueError("shape error")

            # do this in one scan
            # https://stackoverflow.com/a/49192682/130164
            def get_full_sequences(row):
                return (
                    left_right_mask(row["q"], row["v"]),
                    left_right_mask(row["q"], row["d"]),
                    left_right_mask(row["q"], row["j"]),
                )

            # df[['v_sequence', 'd_sequence', 'j_sequence']] = full_sequences.apply(get_full_sequences, axis=1, result_type="expand")
            # even faster: https://stackoverflow.com/a/48134659/130164 :
            df["v_sequence"], df["d_sequence"], df["j_sequence"] = zip(
                *full_sequences.apply(get_full_sequences, axis=1)
            )

            cur_time = progress_report(participant_label, idx, "Processed", itime, cur_time)

            if not all(df["participant_label"] == participant_label):
                raise ValueError(
                    "Wrong participant label arrived from DB. Something is wrong with the query"
                )

            df = df.drop("participant_label", axis=1)

            # convert dtypes back to int as necessary
            # some columns are nullable ints, so pandas default reads them in with float dtype
            # this causes errors in the sql COPY statement, since passing a float but expecting an int
            # here we convert to a nullable int dtype.
            # TODO: is there a better way to do this in read_sql_query? or read all as strings?
            for c in cols_to_map_to_int:
                if c in df.columns:
                    # use pandas's support for nullable int dtype: capitalalized Int64
                    # https://pandas.pydata.org/pandas-docs/version/0.24/whatsnew/v0.24.0.html#optional-integer-na-support
                    df[c] = df[c].astype('Int64')

            # https://stackoverflow.com/a/55495065/130164
            df.to_sql(
                name=part_igh_table_name,
                con=conn,
                schema=schema,
                if_exists="append",
                method=psql_insert_copy,
                index=False,
                chunksize=write_chunksize,
            )

            cur_time = progress_report(
                participant_label,
                idx,
                "Wrote (append) to %s" % part_igh_table_name,
                itime,
                cur_time,
            )

        elapsed_time = time.time() - itime
        print "\t".join(
            [
                participant_label,
                "Finished participant",
                "%0.0f seconds total" % elapsed_time,
            ]
        )

    # Dispose all connections that have been checked back in.
    engine.dispose()

    # return total elapsed seconds
    return elapsed_time


def get_participants(engine, meta, run_label):
    # get the tables
    runs_table = Table("runs", meta, autoload=True)
    participants_table = Table("participants", meta, autoload=True)
    specimens_table = Table("specimens", meta, autoload=True)
    amplifications_table = Table("amplifications", meta, autoload=True)
    replicates_table = Table("replicates", meta, autoload=True)
    barcode_maps_table = Table("barcode_maps", meta, autoload=True)

    # get the run id
    query = (
        select([runs_table.c.run_id])
        .select_from(runs_table)
        .where(runs_table.c.label == run_label)
    )
    run_id = engine.execute(query).scalar()

    # get all the participant labels for the people
    query = (
        select([participants_table.c.label])
        .select_from(
            participants_table.join(
                specimens_table,
                specimens_table.c.participant_id == participants_table.c.participant_id,
            )
            .join(amplifications_table)
            .join(replicates_table)
            .join(barcode_maps_table)
            .join(runs_table)
        )
        .where(runs_table.c.label == run_label)
        .distinct()
        .order_by(participants_table.c.label)
    )
    participant_labels = [r[0] for r in engine.execute(query)]

    return run_id, participant_labels


def make_select_query(meta, run_label, participant_label):
    # get the tables
    participants_table = Table("participants", meta, autoload=True)
    specimens_table = Table("specimens", meta, autoload=True)
    amplifications_table = Table("amplifications", meta, autoload=True)
    replicates_table = Table("replicates", meta, autoload=True)
    demuxed_reads_table = Table(
        "demuxed_reads_%s" % run_label.lower(), meta, autoload=True
    )
    trimmed_reads_table = Table(
        "trimmed_reads_%s" % run_label.lower(), meta, autoload=True
    )
    parsed_table = Table(
        "parsed_igh_igblast_%s" % run_label.lower(), meta, autoload=True
    )
    scores_table = Table(
        "spam_scored_igh_igblast_%s" % run_label.lower(), meta, autoload=True
    )
    isosubtype_table = Table("isosubtypes_%s" % run_label.lower(), meta, autoload=True)

    # build the main query
    query = (
        select(
            [
                participants_table.c.label.label("participant_label"),
                trimmed_reads_table.c.trimmed_read_id,
                trimmed_reads_table.c.sequence.label("trimmed_sequence"),
                trimmed_reads_table.c.forward_primer_id,
                trimmed_reads_table.c.reverse_primer_id,
                demuxed_reads_table.c.replicate_id,
                parsed_table.c.parsed_igh_igblast_id,
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
                parsed_table.c.deletions_post,
                scores_table.c.spam_score,
                isosubtype_table.c.isosubtype,
            ]
        )
        .select_from(
            trimmed_reads_table.join(demuxed_reads_table)
            .join(replicates_table)
            .join(amplifications_table)
            .join(specimens_table)
            .join(
                participants_table,
                specimens_table.c.participant_id == participants_table.c.participant_id,
            )
            .join(parsed_table)
            .join(scores_table)
            .outerjoin(isosubtype_table)
        )
        .where(participants_table.c.label == participant_label)
    )
    return query


def main(arguments):
    # program options
    parser = argparse.ArgumentParser(
        description="sort IgH reads from a run into person specific table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_label", metavar="run", help="the run to pull reads from")
    parser.add_argument(
        "--schema",
        "-s",
        metavar="S",
        type=str,
        default="person_wise",
        help="the schema containing the part-tables",
    )
    parser.add_argument("--read-chunk-size", metavar="N", type=int, required=True)
    parser.add_argument("--write-chunk-size", metavar="N", type=int, required=True)
    parser.add_argument("--num-jobs", metavar="N", type=int, required=True)

    boydlib.add_log_level_arg(parser)
    boydlib.add_read_write_database_args(parser)

    args = parser.parse_args(arguments)
    boydlib.set_log_level(args)

    password = getpass.getpass(
        "Enter password for user %s to access database %s:"
        % (args.db_user, args.db_database)
    )
    # connect to the database
    engine = sqlalchemy.create_engine(
        "postgresql://",
        creator=lambda: socket_connect(args.db_user, args.db_database, password),
        poolclass=NullPool, # See pooling discussion above
        # echo_pool='debug',
        implicit_returning=False,
        connect_args={'connect_timeout': 24 * 60 * 60} # 24 hour timeout
    )
    meta = sqlalchemy.schema.MetaData(bind=engine)

    run_id, participant_labels = get_participants(engine, meta, args.run_label)
    print "Run id", run_id
    print "Participant labels", participant_labels

    itime = time.time()

    # remove constraints first for all tables
    table_names = [
        make_part_table_name(participant_label)
        for participant_label in participant_labels
    ]
    drop_cmds, add_cmds = plan_constraint_drop_add(engine, table_names, args.schema)
    drop_constraints(engine, drop_cmds)
    print "Dropped constraints in %0.0f total seconds" % (time.time() - itime)
    print "Later will run these commands to re-add constraints:"
    print '*' * 60
    print

    print ";\n".join(add_cmds)


    print
    print '*' * 60
    print "Now starting per-participant division."

    example_destination_table = Table(table_names[0], meta, autoload=True, schema=args.schema)
    example_destination_cols_types = [(c.name, c.type.python_type) for c in example_destination_table.columns]
    cols_to_map_to_int = [c[0] for c in example_destination_cols_types if c[1] == int]

    # parallelize across participants
    # run in separate processes, to avoid global interpreter lock
    # db connection will be reestablished in each process
    with parallel_backend("loky", n_jobs=args.num_jobs):
        Parallel()(
            delayed(run_participant)(
                participant_label,
                run_id,
                args.run_label,
                args.db_user,
                args.db_database,
                password,
                args.schema,
                args.read_chunk_size,
                args.write_chunk_size,
                cols_to_map_to_int
            )
            for participant_label in participant_labels
        )
    print "Completed all participant runs in %0.0f total seconds" % (
        time.time() - itime
    )
    print "Readding constraints..."

    # readd constraints after all done
    readd_constraints(engine, add_cmds)
    print "Done in %0.0f total seconds" % (time.time() - itime)

    # TODO:
    # readd support for
    # - subset to list of participants to sort
    # - drop some replicates
    # - checking tables do not exist
    # - output read counts per individual

    # TODO: use logging instead of print
    # TODO: sys.stdout.flush ?
    # TODO: what is optimal chunksize here? Best so far are 100k/200
    # TODO: https://github.com/tqdm/tqdm

if __name__ == "__main__":
    main(sys.argv[1:])
