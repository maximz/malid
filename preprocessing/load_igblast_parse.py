#!/usr/bin/env python

import logging
import sys
import argparse
from contextlib import contextmanager
import re

import psycopg2
import getpass
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.sql import select, text

from Bio import SeqIO
from Bio import Seq

import boydlib

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

def remove_dash_translate(s):
    """Outputs translated sequence for a nucleotide sequence while skipping dashes.
    """
    aa_seq = ''
    codon = ''
    for c in s:
        if c == '-':
            aa_seq += ' '
        else:
            codon += c
            if len(codon) == 3:
                aa_seq += ' ' + str(Seq.Seq(codon).translate()) + ' '
                codon = ''
    return aa_seq + ' ' * len(codon)

def strip_and_replace(s, char, rep=' '):
    """Replace starting and ending runs of the given character with another character.
    """
    if s.count(char) == len(s):
        return rep * len(s)
    else:
        return (rep * (len(s) - len(s.lstrip(char)))) + s.strip(char) + (rep * (len(s) - len(s.rstrip(char))))

def int_or_none(s):
    s = s.strip()
    if s == '':
        return None
    else:
        return int(s)

def map_mutations(q_seq, v_seq, d_seq, j_seq):
    """Turns the mutations in the query into upper case.

       Takes the full alignment and turns the query sequence into lower case except in those instances where
       there is a mutation away from the germline. Mutations in the V take precedence over any non-mutations
       in the other segments. J mutations take precedence over mutations in the D.
    """
    # turns Nones into strings of spaces
    if v_seq is None:
        v_seq = ' ' * len(q_seq)
    if d_seq is None:
        d_seq = ' ' * len(q_seq)
    if j_seq is None:
        j_seq = ' ' * len(q_seq)
    assert len(q_seq) == len(v_seq) == len(d_seq) == len(j_seq)
    new_q_seq = ''
    for q, v, d, j in zip(q_seq, v_seq, d_seq, j_seq):
        new_base = q.lower()
        if v != ' ':
            if v != '.' and v != '-':
                new_base = new_base.upper()
        elif j != ' ':
            if j != '.' and j != '-':
                new_base = new_base.upper()
        elif d != ' ':
            if d != '.' and d != '-':
                new_base = new_base.upper()
        new_q_seq += new_base
    return new_q_seq

def count_query_deletes(q_seq, v_seq, d_seq, j_seq):
    """Count the number of bases deleted from the query given the alignment.
    """
    return q_seq.count('-')

def count_query_inserts(q_seq, v_seq, d_seq, j_seq):
    """Count the number of bases insert into the query given the alignment.
    """
    # turns Nones into strings of spaces
    if v_seq is None:
        v_seq = ' ' * len(q_seq)
    if d_seq is None:
        d_seq = ' ' * len(q_seq)
    if j_seq is None:
        j_seq = ' ' * len(q_seq)

    assert len(q_seq) == len(v_seq) == len(d_seq) == len(j_seq)

    insert_count = 0
    for q, v, d, j in zip(q_seq, v_seq, d_seq, j_seq):
        if v != ' ':
            if v == '-':
                insert_count += 1
        elif j != ' ':
            if j == '-':
                insert_count += 1
        elif d != ' ':
            if d == '-':
                insert_count += 1
    return insert_count

wgxg_motif_re = re.compile('(?P<codon1>T-*G-*G)-*(?P<codon2>G-*G-*[ACGT])-*(?P<codon3>[ACGT]-*[ACGT]-*[ACGT])-*(?P<codon4>G-*G-*[ACGT])', flags=re.IGNORECASE)
def find_wgqg_wgxg_motif(seq):
    """Searches the given sequence for codons that code for WGXG and returns the start position.

       Searches the given sequence for the nucleotides that code for the amino acid sequence
       WGXG and returns the right most start position. Gap characters are allowed in the codon.
       WGQG sequences are prefered to WGXG.
    """
    wgqg_position = None
    wgxg_position = None
    # sadly, the motif can self-overlap so we can't use finditer or findall :(
    start_pos = 0
    match = wgxg_motif_re.search(seq, start_pos)
    while match:
        match_start = match.start('codon1')
        codon3 = match.group('codon3').upper().replace('-', '')

        if codon3 == 'CAA' or codon3 == 'CAG':
            wgqg_position = match_start
        else:
            wgxg_position = match_start

        # start looking for the next match from just after this position
        start_pos = match_start + 1
        match = wgxg_motif_re.search(seq, start_pos)

    if wgqg_position is not None:
        return wgqg_position
    else:
        return wgxg_position

fgxg_motif_re = re.compile('(?P<codon1>T-*T-*[CT])-*(?P<codon2>G-*G-*[ACGT])-*(?P<codon3>[ACGT]-*[ACGT]-*[ACGT])-*(?P<codon4>G-*G-*[ACGT])', flags=re.IGNORECASE)
def find_fggg_fgxg_motif(seq):
    """Searches the given sequence for codons that code for FGXG and returns the start position.

       Searches the given sequence for the nucleotides that code for the amino acid sequence
       FGXG and returns the right most start position. Gap characters are allowed in the codon.
       FGGG sequences are prefered to FGXG.
    """
    fggg_position = None
    fgxg_position = None
    # sadly, the motif can self-overlap so we can't use finditer or findall :(
    start_pos = 0
    match = fgxg_motif_re.search(seq, start_pos)
    while match:
        match_start = match.start('codon1')
        codon3 = match.group('codon3').upper().replace('-', '')

        if codon3 == 'GGA' or codon3 == 'GGC' or codon3 == 'GGG' or codon3 == 'GGT':
            fggg_position = match_start
        else:
            fgxg_position = match_start

        # start looking for the next match from just after this position
        start_pos = match_start + 1
        match = fgxg_motif_re.search(seq, start_pos)

    if fggg_position is not None:
        return fggg_position
    else:
        return fgxg_position

def IgBLASTAlignment(handle):
    """Breaks down an IgBLAST output file into blocks for each alignment

       Parses the outout of IgBLAST and yields a query name and a string of the block
       content for each query.
    """
    # skip any header stuff
    while True:
        line = handle.readline()
        if line == '':
            return  # Premature end of file, or just empty?
        elif line.startswith('Query= '):
            break

    while True:
        # parse the query name
        query_name = line[7:].rstrip()
        lines = []
        line = handle.readline()

        # extract the lines in the block
        while True:
            if not line:
                break
            elif line.startswith('Query= '):
                break
            elif line.startswith('  Database: '):
                break
            lines.append(line)
            line = handle.readline()

        yield query_name, ''.join(lines).rstrip('\n')

        if not line or line.startswith('  Database: '):
            return

def ProcessHeader(header):
    """Extracts the list of hits and their score from header of the IgBLAST alignment

       Parses the length and lits of hits (with their scores) from the top block
       of an IgBLAST alignmment. The length and a dictionary of the hits and their
       scores are returned.
    """
    # extract the length from the top
    blank, length_line, rest = header.split('\n', 2)
    assert blank == ''
    assert length_line.startswith('Length=')
    length = int(length_line[7:])

    # remove the table header lines
    header, label, blank, alignments = rest.split('\n', 3)
    assert header == '                                                                                                      Score     E'
    assert label  == 'Sequences producing significant alignments:                                                          (Bits)  Value'
    assert blank == ''

    hits_table = {}

    # process each alignment hit
    for hit_line in alignments.split('\n'):
        name, bit_score, e_value = hit_line.rsplit(None, 2)
        # remove the starting lcl|
        assert name.startswith('lcl|')
        name = name[4:]
        bit_score = float(bit_score)
        e_value   = float(e_value)
        # name is acutally suffixed with '  germline gene'
        if name.endswith('  germline gene'):
            name, germline_gene = name.split('  ')
            assert germline_gene == 'germline gene'

        hits_table[name] = bit_score, e_value

    return length, hits_table

def ProcessDomin(domain):
    """Process the domain classification line

       Returns the domain classification (IMGT, Kabat) requested for this alignment.
    """
    assert domain.startswith('Domain classification requested: ')
    return domain[33:]

def ProcessSummary(summary):
    """Process the rearrangement, junction, and FR/CDR region summary.
    """
    # break summary into its parts
    if summary.count('\n\n') == 1:  # no framework regions found
        rearrangement, junction = summary.split('\n\n')
        fr_cdr_regions = None
        reverse_complemented = False
    elif summary.count('\n\n') == 2:
        # sequence has been reverse complemented but contains no framework
        if summary.startswith('Note that your query represents the minus strand of a V gene and has been converted to the plus strand. The sequence positions refer to the converted sequence. \n'):
            header, rearrangement, junction = summary.split('\n\n')
            fr_cdr_regions = None
            assert header == 'Note that your query represents the minus strand of a V gene and has been converted to the plus strand. The sequence positions refer to the converted sequence. ', header
            reverse_complemented = True
        else:   # sequence has NOT been reverse complemented and contains a framework
            rearrangement, junction, fr_cdr_regions = summary.split('\n\n')
            reverse_complemented = False
    elif summary.count('\n\n') == 3:    # sequence has been reverse complemented
        header, rearrangement, junction, fr_cdr_regions = summary.split('\n\n')
        assert header == 'Note that your query represents the minus strand of a V gene and has been converted to the plus strand. The sequence positions refer to the converted sequence. ', header
        reverse_complemented = True
    else:
        assert False

    # process the rearrangment summary
    header, rearrangement = rearrangement.split('\n')
    # the split for IgH
    if rearrangement.count('\t') ==  7:
        found_igh = True
        top_v_genes, top_d_genes, top_j_genes, chain_type, stop_codon, v_j_in_frame, productive, strand = rearrangement.split('\t')
    # split for IgL and IgK
    elif rearrangement.count('\t') ==  6:
        found_igh = False
        top_v_genes, top_j_genes, chain_type, stop_codon, v_j_in_frame, productive, strand = rearrangement.split('\t')
        top_d_genes = 'N/A'
    else:
        assert False

    if found_igh:
        assert header == 'V-(D)-J rearrangement summary for query sequence (Top V gene match, Top D gene match, Top J gene match, Chain type, stop codon, V-J frame, Productive, Strand).  Multiple equivalent top matches having the same score and percent identity, if present, are separated by a comma.'
    else:
        assert header == 'V-(D)-J rearrangement summary for query sequence (Top V gene match, Top J gene match, Chain type, stop codon, V-J frame, Productive, Strand).  Multiple equivalent top matches having the same score and percent identity, if present, are separated by a comma.'

    # make sure the strand is - if the sequence has been reverse complemented
    if reverse_complemented:
        assert strand == '-'

    # turn top hits into lists
    if top_v_genes == 'N/A':
        top_v_genes = [None]
    else:
        top_v_genes = top_v_genes.split(',')
    if top_d_genes == 'N/A':
        top_d_genes = [None]
    else:
        top_d_genes = top_d_genes.split(',')
    if top_j_genes == 'N/A':
        top_j_genes = [None]
    else:
        top_j_genes = top_j_genes.split(',')
    # parse if there is a top codon
    if stop_codon == 'Yes':
        stop_codon = True
    elif stop_codon == 'No':
        stop_codon = False
    elif stop_codon == 'N/A':
        stop_codon = None
    else:
        assert False, stop_codon
    # parse if the V and J are in frame
    if v_j_in_frame == 'In-frame':
        v_j_in_frame = True
    elif v_j_in_frame == 'Out-of-frame':
        v_j_in_frame = False
    elif v_j_in_frame == 'N/A':
        v_j_in_frame = None
    else:
        assert False, v_j_in_frame
    # parse if the rearrangement is productive
    if productive == 'Yes':
        productive = True
    elif productive == 'No':
        productive = False
    elif productive == 'N/A':
        productive = None
    else:
        assert False, productive

    assert strand == '+' or strand == '-'

    # process the junction summary
    header, junction = junction.split('\n')
    if found_igh:
        assert header == 'V-(D)-J junction details based on top germline gene matches (V end, V-D junction, D region, D-J junction, J start).  Note that possible overlapping nucleotides at VDJ junction (i.e, nucleotides that could be assigned to either rearranging gene) are indicated in parentheses (i.e., (TACT)) but are not included under the V, D, or J gene itself'
        v_end_seq, n1_seq, d_seq, n2_seq, j_start_seq, blank = junction.split('\t')
    else:
        assert header == 'V-(D)-J junction details based on top germline gene matches (V end, V-J junction, J start).  Note that possible overlapping nucleotides at VDJ junction (i.e, nucleotides that could be assigned to either rearranging gene) are indicated in parentheses (i.e., (TACT)) but are not included under the V, D, or J gene itself'
        v_end_seq, n1_seq, j_start_seq, blank = junction.split('\t')
        n2_seq = None
        d_seq  = None

    assert blank == ''
    if v_end_seq == 'N/A':
        v_end_seq = None
    if j_start_seq == 'N/A':
        j_start_seq = None
    # turns N/A into None and assume that shared sequences belong to the V
    if n1_seq == 'N/A':
        n1_seq = None
        n1_overlap = None
    elif n1_seq.startswith('(') and n1_seq.endswith(')'):
        v_end_seq = (v_end_seq + n1_seq[1:-1])[-5:]
        n1_seq = ''
        n1_overlap = True
    else:
        n1_overlap = False
    # if the D sequence is N/A, turn that into a None
    if d_seq == 'N/A':
        d_seq = None
    # turns N/A into None and assume that shared sequences belong to the J
    if n2_seq is None or n2_seq == 'N/A':
        n2_seq = None
        n2_overlap = None
    elif n2_seq.startswith('(') and n2_seq.endswith(')'):
        j_start_seq = (n2_seq[1:-1] + j_start_seq)[:5]
        n2_seq = ''
        n2_overlap = True
    else:
        n2_overlap = False

    fr_cdr_ranges = []

    if fr_cdr_regions is not None:
        header, regions = fr_cdr_regions.split('\n', 1)
        assert header == 'Alignment summary between query and top germline V gene hit (from, to, length, matches, mismatches, gaps, percent identity)'
        for region_line in regions.split('\n'):
            if region_line.count('\t') == 7:
                label, from_pos, to_pos, length, matches, mismatches, gaps, percent_id = region_line.split('\t')
            elif region_line.count('\t') == 9:
                label, from_pos, to_pos, length, matches, mismatches, gaps, percent_id, thing1, thing2 = region_line.split('\t')
                logging.info('found two extra columns (%s and %s) in region %s' % (thing1, thing2, label))
            else:
                raise ValueError('found weird alignment summary line:\n%s' % region_line)
            from_pos = None if from_pos == 'N/A' else int(from_pos)
            to_pos = None if to_pos == 'N/A' else int(to_pos)
            length = None if length == 'N/A' else int(length)
            matches = None if matches == 'N/A' else int(matches)
            mismatches = None if mismatches == 'N/A' else int(mismatches)
            gaps = None if gaps == 'N/A' else int(gaps)
            percent_id = None if percent_id == 'N/A' else float(percent_id)

            fr_cdr_ranges.append((label, from_pos, to_pos))

    return top_v_genes[0], top_d_genes[0], top_j_genes[0], fr_cdr_ranges, chain_type, stop_codon, v_j_in_frame, productive, strand, reverse_complemented,\
            v_end_seq, n1_seq, d_seq, n2_seq, j_start_seq, n1_overlap, n2_overlap

query_line_re = re.compile('(?P<startpad> +)(?P<query>\S+ +)(?P<startpos>\d+ +)(?P<seq>\S+)  (?P<endpos>\d+)')

def ProcessAlignment(alignment, best_v, best_d, best_j):
    header, rest = alignment.split('\n\n', 1)
    assert header == 'Alignments'

    # alignment for the framework string, query, and the other matches
    framework_align = ''
    query_seq_align = ''
    other_seq_align = {}
    other_seq_type  = {}

    # the first start position in the query
    first_query_from = None

    # start position top in V, D, and J
    top_v_from = None
    top_d_from = None
    top_j_from = None

    # end position in top V, D, and J
    top_v_to = None
    top_d_to = None
    top_j_to = None

    # prev. to position used to ensure that the blocks are contiguous
    prev_query_to = None

    for block in rest.split('\n\n'):
        # pull off the first line and see if it looks like a query
        maybe_query, rest = block.split('\n', 1)
        match = query_line_re.match(maybe_query)
        if match:   # if match, no framework
            query = maybe_query
            framework = None
        else:       # if no match, assume it's framework and next line is query
            framework, query, rest = block.split('\n', 2)
            match = query_line_re.match(query)
            assert match, query

        # the start positon of the query name
        label_from = match.start('query')
        # the start of start position
        start_pos_from = match.start('startpos')
        # the start of end position
        stop_pos_from = match.start('endpos')
        # the from and to positions of the alignment rect
        alignment_rect_from = match.end('startpos')
        alignment_rect_to   = match.end('seq')

        # use the rectangle to extract out the framework and query alignment sequence
        if framework is None:
            framework_align += ' ' * (alignment_rect_to - alignment_rect_from)
        else:
            framework_align += framework[alignment_rect_from:alignment_rect_to]
        query_seq_align += query[alignment_rect_from:alignment_rect_to]

        query_from = int(match.group('startpos'))
        query_to   = int(match.group('endpos'))

        # if this is the first query, store it's from position
        # and init. prev_query_to to just before that
        if first_query_from is None:
            first_query_from = query_from
            prev_query_to = query_from - 1

        # if this isn't the first positon, make sure the current from position is right next to the prev. to position
        assert prev_query_to + 1 == query_from
        # update the previous to position
        prev_query_to = query_to

        for match_line in rest.split('\n'):
            # get the name and positions
            match_name = match_line[label_from:start_pos_from].rstrip()
            match_seq  = match_line[alignment_rect_from:alignment_rect_to]

            seq_type, percent_id, counts = match_line[:label_from].split()
            assert seq_type == 'V' or seq_type == 'D' or seq_type == 'J'

            if seq_type == 'V':
                if match_name == best_v:
                    if top_v_from is None:
                        top_v_from = int_or_none(match_line[start_pos_from:alignment_rect_from])
                        top_v_to   = int_or_none(match_line[stop_pos_from:])
                    else:
                        top_v_to   = int_or_none(match_line[stop_pos_from:])
            elif seq_type == 'D':
                if match_name == best_d:
                    if top_d_from is None:
                        top_d_from = int_or_none(match_line[start_pos_from:alignment_rect_from])
                        top_d_to   = int_or_none(match_line[stop_pos_from:])
                    else:
                        top_d_to   = int_or_none(match_line[stop_pos_from:])
            elif seq_type == 'J':
                if match_name == best_j:
                    if top_j_from is None:
                        top_j_from = int_or_none(match_line[start_pos_from:alignment_rect_from])
                        top_j_to   = int_or_none(match_line[stop_pos_from:])
                    else:
                        top_j_to   = int_or_none(match_line[stop_pos_from:])

            # store the type of this match
            if match_name not in other_seq_type:
                other_seq_type[match_name] = seq_type
            assert other_seq_type[match_name] == seq_type

            # convert percent into float
            assert percent_id.endswith('%')
            percent_id = float(percent_id[:-1])

            # extract numerator and denominator
            assert counts.startswith('(') and counts.endswith(')')
            numerator, denominator = counts[1:-1].split('/')

            # if this is the first
            if match_name not in other_seq_align:
                other_seq_align[match_name] = '-' * (len(query_seq_align) - len(match_seq))
            # append this part of the sequence to the alignment
            other_seq_align[match_name] += match_seq

    # make sure the framework and query are the same length
    assert len(framework_align) == len(query_seq_align)
    for other in other_seq_align:
        # add any missing - to the end of the other sequences
        other_seq_align[other] += '-' * (len(query_seq_align) - len(other_seq_align[other]))
        # remove the start and ending - since dash has different meanings there
        other_seq_align[other] = strip_and_replace(other_seq_align[other], '-', ' ')

    if best_v is not None:
        best_v_align = other_seq_align[best_v]
    else:
        best_v_align = None

    if best_d is not None:
        best_d_align = other_seq_align[best_d]
    else:
        best_d_align = None

    if best_j is not None:
        best_j_align = other_seq_align[best_j]
    else:
        best_j_align = None

    return framework_align, query_seq_align, first_query_from, prev_query_to, top_v_from, top_v_to, top_d_from, top_d_to, top_j_from, top_j_to, best_v_align, best_d_align, best_j_align

fr_cdr_region_re = re.compile('(?P<start><)-*(?P<label>[^->]*)(-(?P<class>[^->]+))?-*(?P<end>>)')

def ProcessFRCDR(framework_align, query_seq_align, expect_chain_type, best_v_align, best_d_align, best_j_align, start_in_j, domain_class, query_aa_align, new_record):
    looked_for_cdr3 = False
    found_motif     = False

    domain_class = domain_class.upper()
    # process the prefix part of the alignment
    from_pos = 0
    if '<' in framework_align:
        to_pos = framework_align.find('<')
    else:
        to_pos = len(framework_align)
    if from_pos < to_pos:
        label ='pre'
        q_seq_nt = query_seq_align[from_pos:to_pos]
        q_seq_aa = query_aa_align[from_pos:to_pos]
        if best_v_align is not None:
            v_seq_nt = best_v_align[from_pos:to_pos]
        else:
            v_seq_nt = None
        if best_d_align is not None:
            d_seq_nt = best_d_align[from_pos:to_pos]
        else:
            d_seq_nt = None
        if best_j_align is not None:
            j_seq_nt = best_j_align[from_pos:to_pos]
        else:
            j_seq_nt = None
        new_record['%s_seq_nt_q' % label] = q_seq_nt
        new_record['%s_seq_aa_q' % label] = q_seq_aa
        new_record['%s_seq_nt_v' % label] = v_seq_nt
        new_record['%s_seq_nt_d' % label] = d_seq_nt
        new_record['%s_seq_nt_j' % label] = j_seq_nt
        new_record['insertions_%s' % label] = count_query_inserts(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)
        new_record['deletions_%s' % label]  = count_query_deletes(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)
    # process the regions
    for region in fr_cdr_region_re.finditer(framework_align):
        # if we there eenough info about he domain system, make sure it match what we've parsed
        if region.group('class') is not None:
            assert domain_class.startswith(region.group('class')), \
                    'domain system %s not matching region label %s' % (domain_class, region.group('class'))
        label = region.group('label')

        assert label != 'CDR3'
        from_pos = region.start('start')
        to_pos   = region.end('end')
        if (label.startswith('CDR') and len(label) == 4) or \
           (label.startswith('FR')  and len(label) == 3):
            label = label.lower()

            q_seq_nt = query_seq_align[from_pos:to_pos]
            q_seq_aa = query_aa_align[from_pos:to_pos]
            if best_v_align is not None:
                v_seq_nt = best_v_align[from_pos:to_pos]
            else:
                v_seq_nt = None
            if best_d_align is not None:
                d_seq_nt = best_d_align[from_pos:to_pos]
            else:
                d_seq_nt = None
            if best_j_align is not None:
                j_seq_nt = best_j_align[from_pos:to_pos]
            else:
                j_seq_nt = None
            new_record['%s_seq_nt_q' % label] = q_seq_nt
            new_record['%s_seq_aa_q' % label] = q_seq_aa
            new_record['%s_seq_nt_v' % label] = v_seq_nt
            new_record['%s_seq_nt_d' % label] = d_seq_nt
            new_record['%s_seq_nt_j' % label] = j_seq_nt
            new_record['insertions_%s' % label] = count_query_inserts(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)
            new_record['deletions_%s' % label]  = count_query_deletes(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)

            # if there is a FR3 and a J, look for a CDR3
            if label == 'fr3' and best_j_align is not None:
                looked_for_cdr3 = True
                from_pos = to_pos   # CDR3 starts after FR3 ends
                # find the start of the J match
                current_pos = from_pos
                while current_pos < len(best_j_align) and best_j_align[current_pos] == ' ':
                    current_pos += 1
                if current_pos != len(best_j_align):    # if we found the J alignment after the FR3
                    # TODO: query_from_j = query_seq_align[current_pos - (start_in_j - 1):]
                    query_from_j = query_seq_align[current_pos:]

                    # look for the correct motif for the chain type
                    if expect_chain_type == 'VH':
                        motif_offset = find_wgqg_wgxg_motif(query_from_j)
                    else:
                        motif_offset = find_fggg_fgxg_motif(query_from_j)

                    if motif_offset is not None:
                        found_motif = True
                        to_pos = current_pos + motif_offset
                        label = 'cdr3'
                        q_seq_nt = query_seq_align[from_pos:to_pos]
                        q_seq_aa = query_aa_align[from_pos:to_pos]
                        if best_v_align is not None:
                            v_seq_nt = best_v_align[from_pos:to_pos]
                        else:
                            v_seq_nt = None
                        if best_d_align is not None:
                            d_seq_nt = best_d_align[from_pos:to_pos]
                        else:
                            d_seq_nt = None
                        if best_j_align is not None:
                            j_seq_nt = best_j_align[from_pos:to_pos]
                        else:
                            j_seq_nt = None
                        new_record['%s_seq_nt_q' % label] = q_seq_nt
                        new_record['%s_seq_aa_q' % label] = q_seq_aa
                        new_record['%s_seq_nt_v' % label] = v_seq_nt
                        new_record['%s_seq_nt_d' % label] = d_seq_nt
                        new_record['%s_seq_nt_j' % label] = j_seq_nt
                        new_record['insertions_%s' % label] = count_query_inserts(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)
                        new_record['deletions_%s' % label]  = count_query_deletes(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)

    # output the rest of the alignment
    from_pos = to_pos
    to_pos =  len(query_seq_align)
    if from_pos < to_pos:
        label = 'post'
        q_seq_nt = query_seq_align[from_pos:to_pos]
        q_seq_aa = query_aa_align[from_pos:to_pos]
        if best_v_align is not None:
            v_seq_nt = best_v_align[from_pos:to_pos]
        else:
            v_seq_nt = None
        if best_d_align is not None:
            d_seq_nt = best_d_align[from_pos:to_pos]
        else:
            d_seq_nt = None
        if best_j_align is not None:
            j_seq_nt = best_j_align[from_pos:to_pos]
        else:
            j_seq_nt = None
        new_record['%s_seq_nt_q' % label] = q_seq_nt
        new_record['%s_seq_aa_q' % label] = q_seq_aa
        new_record['%s_seq_nt_v' % label] = v_seq_nt
        new_record['%s_seq_nt_d' % label] = d_seq_nt
        new_record['%s_seq_nt_j' % label] = j_seq_nt
        new_record['insertions_%s' % label] = count_query_inserts(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)
        new_record['deletions_%s' % label]  = count_query_deletes(q_seq_nt, v_seq_nt, d_seq_nt, j_seq_nt)

    return looked_for_cdr3, found_motif

def main():
    # program options
    parser = argparse.ArgumentParser(description='Load IgBlast parsed sequences into the database',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # table to load parses into
    parser.add_argument('trimmed_table', metavar='trimmed-table', help='the table of trimmed reads e.g. trimmed_reads_m54')
    parser.add_argument('parse_table', metavar='parse-table', help='the table to load the parses into, e.g. parsed_igh_igblast_t18')
    # files
    parser.add_argument('parse_files', metavar='parse-files', nargs='+',
            help='files with the IgBlasted parses to load')
    # list of trimmed read ids to exclude
    parser.add_argument('--exclude', metavar='F', help='file with trimmed_read_ids to exclude')

    # which locus should be parsed
    parser.add_argument('--locus', '-l', metavar='L', required=True,
            choices=['IgH', 'IgK', 'IgL', 'TCRA', 'TCRB', 'TCRG', 'TCRD'],
            help='which locus to parse: IgH, IgK, IgL (default: %(default)s)')

    #
    parser.add_argument('--v-name-map', metavar='v_name_map', help='a map V segment names to used to rename the V segments')
    parser.add_argument('--d-name-map', metavar='d_name_map', help='a map D segment names to used to rename the D segments')
    parser.add_argument('--j-name-map', metavar='j_name_map', help='a map J segment names to used to rename the J segments')
    parser.add_argument('--name-prefix', metavar='PRE', default='', help='string to append of the V, D, and J segment names')

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
    table_exists = engine.has_table(args.parse_table)
    if table_exists:
        logging.error('table %s already exists' % args.parse_table)
        sys.exit(10)

    if args.v_name_map:
        logging.info('Loading V name map from %s' % args.v_name_map)
        v_name_map = {None: None}
        for line in open(args.v_name_map, 'r'):
            row = line[:-1].split('\t')
            if row[1] == '':
                v_name_map[row[0]] = None
            else:
                v_name_map[row[0]] = row[1]
    else:
        v_name_map = None

    if args.d_name_map:
        logging.info('Loading D name map from %s' % args.d_name_map)
        d_name_map = {None: None}
        for line in open(args.d_name_map, 'r'):
            row = line[:-1].split('\t')
            if row[1] == '':
                d_name_map[row[0]] = None
            else:
                d_name_map[row[0]] = row[1]
    else:
        d_name_map = None

    if args.j_name_map:
        logging.info('Loading J name map from %s' % args.j_name_map)
        j_name_map = {None: None}
        for line in open(args.j_name_map, 'r'):
            row = line[:-1].split('\t')
            if row[1] == '':
                j_name_map[row[0]] = None
            else:
                j_name_map[row[0]] = row[1]
    else:
        j_name_map = None

    exclude_ids = set()
    if args.exclude:
        logging.info('excluding read ids from %s' % args.exclude)
        for ident in open(args.exclude, 'r'):
            exclude_ids.add(int(ident.strip()))

    trimmed_table = Table(args.trimmed_table, meta, autoload=True)
    if args.locus == 'IgH':
        key_name = 'parsed_igh_igblast_id'
        v_segments = Table('igh_v_segments', meta, autoload=True)
        d_segments = Table('igh_d_segments', meta, autoload=True)
        j_segments = Table('igh_j_segments', meta, autoload=True)
        expect_chain_type = 'VH'
    elif args.locus == 'IgK':
        key_name = 'parsed_igk_igblast_id'
        v_segments = Table('igk_v_segments', meta, autoload=True)
        d_segments = Table('igk_d_segments', meta, autoload=True)
        j_segments = Table('igk_j_segments', meta, autoload=True)
        expect_chain_type = 'VK'
    elif args.locus == 'IgL':
        key_name = 'parsed_igl_igblast_id'
        v_segments = Table('igl_v_segments', meta, autoload=True)
        d_segments = Table('igl_d_segments', meta, autoload=True)
        j_segments = Table('igl_j_segments', meta, autoload=True)
        expect_chain_type = 'VL'
    elif args.locus == 'TCRA':
        key_name = 'parsed_tcra_igblast_id'
        v_segments = Table('tcra_v_segments', meta, autoload=True)
        d_segments = Table('tcra_d_segments', meta, autoload=True)
        j_segments = Table('tcra_j_segments', meta, autoload=True)
        expect_chain_type = 'VA'
    elif args.locus == 'TCRB':
        key_name = 'parsed_tcrb_igblast_id'
        v_segments = Table('tcrb_v_segments', meta, autoload=True)
        d_segments = Table('tcrb_d_segments', meta, autoload=True)
        j_segments = Table('tcrb_j_segments', meta, autoload=True)
        expect_chain_type = 'VB'
    elif args.locus == 'TCRG':
        key_name = 'parsed_tcrg_igblast_id'
        v_segments = Table('tcrg_v_segments', meta, autoload=True)
        d_segments = Table('tcrg_d_segments', meta, autoload=True)
        j_segments = Table('tcrg_j_segments', meta, autoload=True)
        expect_chain_type = 'VG'
    elif args.locus == 'TCRD':
        key_name = 'parsed_tcrd_igblast_id'
        v_segments = Table('tcrd_v_segments', meta, autoload=True)
        d_segments = Table('tcrd_d_segments', meta, autoload=True)
        j_segments = Table('tcrd_j_segments', meta, autoload=True)
        expect_chain_type = 'VD'
    else:
        assert 0 == 1

    parse_table = Table(args.parse_table, meta,
        Column(key_name, Integer, primary_key=True),
        Column('trimmed_read_id', Integer, ForeignKey(trimmed_table.name), unique=True, nullable=False),
        Column('v_segment', String(50), ForeignKey(v_segments.c.name)),
        Column('d_segment', String(50), ForeignKey(d_segments.c.name)),
        Column('j_segment', String(50), ForeignKey(j_segments.c.name)),
        Column('v_score', Float),
        Column('d_score', Float),
        Column('j_score', Float),
        Column('stop_codon', Boolean),
        Column('v_j_in_frame', Boolean),
        Column('productive', Boolean),
        Column('strand', String(1)),
        Column('v_end_sequence', String),
        Column('n1_sequence', String),
        Column('d_sequence', String),
        Column('n2_sequence', String),
        Column('j_start_sequence', String),
        Column('n1_overlap', Boolean),
        Column('n2_overlap', Boolean),
        Column('q_start', Integer),
        Column('q_end', Integer),
        Column('v_start', Integer),
        Column('v_end', Integer),
        Column('d_start', Integer),
        Column('d_end', Integer),
        Column('j_start', Integer),
        Column('j_end', Integer),
        Column('pre_seq_nt_q', String),
        Column('pre_seq_nt_v', String),
        Column('pre_seq_nt_d', String),
        Column('pre_seq_nt_j', String),
        Column('fr1_seq_nt_q', String),
        Column('fr1_seq_nt_v', String),
        Column('fr1_seq_nt_d', String),
        Column('fr1_seq_nt_j', String),
        Column('cdr1_seq_nt_q', String),
        Column('cdr1_seq_nt_v', String),
        Column('cdr1_seq_nt_d', String),
        Column('cdr1_seq_nt_j', String),
        Column('fr2_seq_nt_q', String),
        Column('fr2_seq_nt_v', String),
        Column('fr2_seq_nt_d', String),
        Column('fr2_seq_nt_j', String),
        Column('cdr2_seq_nt_q', String),
        Column('cdr2_seq_nt_v', String),
        Column('cdr2_seq_nt_d', String),
        Column('cdr2_seq_nt_j', String),
        Column('fr3_seq_nt_q', String),
        Column('fr3_seq_nt_v', String),
        Column('fr3_seq_nt_d', String),
        Column('fr3_seq_nt_j', String),
        Column('cdr3_seq_nt_q', String),
        Column('cdr3_seq_nt_v', String),
        Column('cdr3_seq_nt_d', String),
        Column('cdr3_seq_nt_j', String),
        Column('post_seq_nt_q', String),
        Column('post_seq_nt_v', String),
        Column('post_seq_nt_d', String),
        Column('post_seq_nt_j', String),
        Column('pre_seq_aa_q', String),
        Column('fr1_seq_aa_q', String),
        Column('cdr1_seq_aa_q', String),
        Column('fr2_seq_aa_q', String),
        Column('cdr2_seq_aa_q', String),
        Column('fr3_seq_aa_q', String),
        Column('cdr3_seq_aa_q', String),
        Column('post_seq_aa_q', String),
        Column('insertions_pre', Integer),
        Column('insertions_fr1', Integer),
        Column('insertions_cdr1', Integer),
        Column('insertions_fr2', Integer),
        Column('insertions_cdr2', Integer),
        Column('insertions_fr3', Integer),
        Column('insertions_post', Integer),
        Column('deletions_pre', Integer),
        Column('deletions_fr1', Integer),
        Column('deletions_cdr1', Integer),
        Column('deletions_fr2', Integer),
        Column('deletions_cdr2', Integer),
        Column('deletions_fr3', Integer),
        Column('deletions_post', Integer))
    logging.info('creating parse table %s' % args.parse_table)
    parse_table.create()
    parse_insert = parse_table.insert()

    # do the following without database constraints
    with out_constraints(engine, parse_table):
        # batch items until we hit user given maximum batch size
        with batched(args.batch_size, lambda batch: engine.execute(parse_insert, batch)) as batch:
            # stats
            count_reads = 0
            count_hits  = 0
            count_chain = 0
            count_rc_hits = 0
            count_vj_hits = 0
            count_no_v_name = 0
            count_no_d_name = 0
            count_no_j_name = 0
            count_looked  = 0
            count_found   = 0

            count_added = 0

            for filename in args.parse_files:
                logging.info('processing file %s' % filename)
                for query, lines in IgBLASTAlignment(open(filename, 'r')):
                    #logging.info('processing read %s' % query)
                    try:
                        count_reads += 1

                        # output status at 100000 intervals
                        if count_reads % 100000 == 0:
                            logging.info('processed %s parses' % count_reads)

                        if lines.count('\n\n\n') == 4:
                            count_hits += 1

                            header, domain, summary, alignment, footer = lines.split('\n\n\n')
                            length, hits_table = ProcessHeader(header)

                            domain_class = ProcessDomin(domain)
                            assert domain_class == 'imgt'

                            best_v, best_d, best_j, fr_cdr_ranges, chain_type, stop_codon, v_j_in_frame, productive, strand, reverse_complemented, \
                            v_end_seq, n1_seq, d_seq, n2_seq, j_start_seq, n1_overlap, n2_overlap \
                                    = ProcessSummary(summary)

                            if chain_type == expect_chain_type:
                                count_chain += 1
                                if reverse_complemented:
                                    count_rc_hits += 1

                                new_record = dict([(str(i.name), None) for i in parse_table.c if not i.primary_key])
                                new_record['trimmed_read_id'] = int(query)

                                if best_v is not None:
                                    new_record['v_score'] = hits_table[best_v][0]
                                if best_d is not None:
                                    new_record['d_score'] = hits_table[best_d][0]
                                if best_j is not None:
                                    new_record['j_score'] = hits_table[best_j][0]
                                new_record['stop_codon'] = stop_codon
                                new_record['v_j_in_frame'] = v_j_in_frame
                                new_record['productive'] = productive
                                new_record['strand'] = strand

                                new_record['v_end_sequence'] = v_end_seq
                                new_record['n1_sequence'] = n1_seq
                                new_record['d_sequence'] = d_seq
                                new_record['n2_sequence'] = n2_seq
                                new_record['j_start_sequence'] = j_start_seq
                                new_record['n1_overlap'] = n1_overlap
                                new_record['n2_overlap'] = n2_overlap

                                framework_align, query_seq_align, query_from, query_to, \
                                top_v_from, top_v_to, top_d_from, top_d_to, top_j_from, top_j_to,  \
                                best_v_align, best_d_align, best_j_align = ProcessAlignment(alignment, best_v, best_d, best_j)

                                # map the V segment names and store it
                                if v_name_map is not None:
                                    if best_v is not None and v_name_map[best_v] is None:
                                        count_no_v_name += 1
                                    best_v = v_name_map[best_v]
                                if best_v is not None:
                                    new_record['v_segment'] = args.name_prefix + best_v
                                else:
                                    new_record['v_segment'] = None

                                # map the D segment names and store it
                                if d_name_map is not None:
                                    if best_d is not None and d_name_map[best_d] is None:
                                        count_no_d_name += 1
                                    best_d = d_name_map[best_d]
                                if best_d is not None:
                                    new_record['d_segment'] = args.name_prefix + best_d
                                else:
                                    new_record['d_segment'] = None

                                # map the J segment names and store it
                                if j_name_map is not None:
                                    if best_j is not None and j_name_map[best_j] is None:
                                        count_no_j_name += 1
                                    best_j = j_name_map[best_j]
                                if best_j is not None:
                                    new_record['j_segment'] = args.name_prefix + best_j
                                else:
                                    new_record['j_segment'] = None

                                if best_v is not None and best_j is not None:
                                    count_vj_hits += 1

                                new_record['q_start'] = query_from
                                new_record['q_end']   = query_to

                                new_record['v_start'] = top_v_from
                                new_record['v_end']   = top_v_to

                                new_record['d_start'] = top_d_from
                                new_record['d_end']   = top_d_to

                                new_record['j_start'] = top_j_from
                                new_record['j_end']   = top_j_to

                                query_seq_align = map_mutations(query_seq_align, best_v_align, best_d_align, best_j_align)

                                # find the frame offset
                                if top_v_from % 3 == 0:
                                    offset = 1
                                elif top_v_from % 3 == 1:
                                    offset = 0
                                else:
                                    offset = 2
                                # translate
                                query_aa_align = (' ' * offset) + remove_dash_translate(query_seq_align[offset:])

                                looked_for_cdr3, found_motif = ProcessFRCDR(framework_align, query_seq_align, expect_chain_type, best_v_align, best_d_align, best_j_align, top_j_from, domain_class, query_aa_align, new_record)
                                if looked_for_cdr3:
                                    count_looked += 1
                                    if found_motif:
                                        count_found += 1

                                if new_record['trimmed_read_id'] not in exclude_ids:
                                    batch.add(new_record)
                        elif lines.count('\n\n\n') == 2:
                            assert '***** No hits found *****' in lines
                    except:
                        logging.error('error processing read %s in %s' % (query, filename))
                        raise

    logging.info('processed %d reads' % count_reads)
    logging.info('found hits for %d reads' % count_hits)
    logging.info('found %s hits for %d reads' % (expect_chain_type, count_chain))
    logging.info('found %d reverse complement hits' % count_rc_hits)
    logging.info('found V and J segments for %d reads' % count_vj_hits)
    logging.info('%d V-segments were set to none because of name map' % count_no_v_name)
    logging.info('%d D-segments were set to none because of name map' % count_no_d_name)
    logging.info('%d J-segments were set to none because of name map' % count_no_j_name)
    logging.info('looked for a CDR3 in %d reads' % count_looked)
    logging.info('found CDR3 motif in %d reads' % count_found)

if __name__ == '__main__':
    main()
