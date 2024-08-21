#!/bin/bash

makeblastdb -parse_seqids -in human_gl_V.fasta -dbtype nucl -title human_gl_V -out human_gl_V
makeblastdb -parse_seqids -in human_gl_D.fasta -dbtype nucl -title human_gl_D -out human_gl_D
makeblastdb -parse_seqids -in human_gl_J.fasta -dbtype nucl -title human_gl_J -out human_gl_J
