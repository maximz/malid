--- ##########
--- ##########
--- ########## iRECEPTOR:

select
    meta."study.study_id",
    count(1)
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    seq.c_call is not NULL
    AND seq.locus = 'IGH'
GROUP BY
    meta."study.study_id"
order by
    2 desc;

--  study.study_id |  count
-- ----------------+---------
--  PRJNA648677    | 3443432
--  PRJCA002413    |    9969
-- (2 rows)

-- so really only one study has a lot of isotyped sequences.
-- and in fact that's the study we process below because it also has other fields set.
-- also exclude PRJNA628125 because that's Boydlab.

select
    meta."study.study_id",
    count(1)
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    seq.productive = TRUE
    AND seq.c_call is not NULL
    AND seq.locus = 'IGH'
    AND meta."study.study_id" != 'PRJNA628125'
    AND seq.cdr1 is not NULL
    AND seq.cdr1_aa is not NULL
    AND seq.cdr2 is not NULL
    AND seq.cdr2_aa is not NULL
    AND seq.cdr3 is not NULL
    AND seq.cdr3_aa is not NULL
GROUP BY
    meta."study.study_id"
order by
    2 desc;


--  study.study_id |  count
-- ----------------+---------
--  PRJNA648677    | 3443432
-- (1 row)

-- #################

select
    meta."study.study_id",
    count(1)
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    seq.sequence is not NULL
    AND meta."study.study_id" != 'PRJNA628125'
GROUP BY
    meta."study.study_id"
order by
    2 desc;


--  study.study_id |  count
-- ----------------+---------
--  PRJNA630455    | 7667705
--  PRJNA648677    | 3443432
--  PRJCA002413    |    9994
-- (3 rows)


select
    meta."study.study_id",
    count(1)
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
GROUP BY
    meta."study.study_id"
order by
    2 desc;

--   study.study_id  |  count
-- ------------------+---------
--  PRJNA628125      | 8970217
--  PRJNA630455      | 7667705
--  IR-Binder-000001 | 6912193
--  PRJNA648677      | 3443432
--  PRJCA002413      |    9994
-- (5 rows)


-- So basically, besides Boydlab data (PRJNA628125), only PRJNA648677 is fully complete for us.
-- (Though still not including clone IDs. That field looks empty.)
-- But we could process the raw sequences from 3 other studies.

select "study.study_id", count(1) from ireceptor_data.covid_metadata group by "study.study_id";

--   study.study_id  | count
-- ------------------+-------
--  PRJNA628125      |    14
--  PRJCA002413      |    10
--  PRJNA630455      |    58
--  IR-Binder-000001 |    67
--  PRJNA648677      |    16
--  PRJNA645245      |    84
--  PRJNA638224      |    19
-- (7 rows)


-- For now we will go with PRJNA648677, which has 16 repertoires. But if you look closely, that's only from 7 patients:

select
    "repertoire_id",
    "subject.subject_id",
    "subject.sex",
    "subject.age_min",
    "sample.0.sample_id",
    "sample.0.collection_time_point_relative",
    "subject.diagnosis.0.disease_length",
    "subject.diagnosis.0.disease_stage"
from
    ireceptor_data.covid_metadata
where "study.study_id" = 'PRJNA648677';

--       repertoire_id       | subject.subject_id | subject.sex | subject.age_min | sample.0.sample_id | sample.0.collection_time_point_relative | subject.diagnosis.0.disease_length | subject.diagnosis.0.disease_stage
-- --------------------------+--------------------+-------------+-----------------+--------------------+-----------------------------------------+------------------------------------+-----------------------------------
--  5f21e814e1adeb2edc12613c | A                  | male        |            55.0 | A_d11              | d11                                     | 11 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e814e1adeb2edc12613d | A                  | male        |            55.0 | A_d17              | d17                                     | 17 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e815e1adeb2edc12613e | A                  | male        |            55.0 | A_d45              | d45                                     | 45 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e815e1adeb2edc12613f | B                  | male        |            55.0 | B_d10              | d10                                     | 10 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e815e1adeb2edc126140 | B                  | male        |            55.0 | B_d19              | d19                                     | 19 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e816e1adeb2edc126141 | C                  | female      |            53.0 | C_d6               | d6                                      | 6 days since symptom onset         | Limited Pneumonic infiltrates
--  5f21e816e1adeb2edc126142 | C                  | female      |            53.0 | C_d15              | d15                                     | 15 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e816e1adeb2edc126143 | D                  | male        |            24.0 | D_d6               | d6                                      | 6 days since symptom onset         | Limited Pneumonic infiltrates
--  5f21e817e1adeb2edc126144 | D                  | male        |            24.0 | D_d28              | d28                                     | 28 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e817e1adeb2edc126145 | E                  | male        |            48.0 | E_d23              | d23                                     | 23 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e817e1adeb2edc126146 | E                  | male        |            48.0 | E_d44              | d44                                     | 44 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e818e1adeb2edc126147 | E                  | male        |            48.0 | E_d99              | d99                                     | 99 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e818e1adeb2edc126148 | F                  | female      |            40.0 | F_d14              | d14                                     | 14 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e818e1adeb2edc126149 | F                  | female      |            40.0 | F_d36              | d36                                     | 36 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e818e1adeb2edc12614a | G                  | female      |            59.0 | G_d9               | d9                                      | 9 days since symptom onset         | Limited Pneumonic infiltrates
--  5f21e819e1adeb2edc12614b | G                  | female      |            59.0 | G_d22              | d22                                     | 22 days since symptom onset        | Limited Pneumonic infiltrates
-- (16 rows)


-- We probably want to choose the peak time points only (day 14-21 -- since seroconverting around day 11):
--       repertoire_id       | subject.subject_id | subject.sex | subject.age_min | sample.0.sample_id | sample.0.collection_time_point_relative | subject.diagnosis.0.disease_length | subject.diagnosis.0.disease_stage
-- --------------------------+--------------------+-------------+-----------------+--------------------+-----------------------------------------+------------------------------------+-----------------------------------
--  5f21e814e1adeb2edc12613d | A                  | male        |            55.0 | A_d17              | d17                                     | 17 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e815e1adeb2edc126140 | B                  | male        |            55.0 | B_d19              | d19                                     | 19 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e816e1adeb2edc126142 | C                  | female      |            53.0 | C_d15              | d15                                     | 15 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e817e1adeb2edc126144 | D                  | male        |            24.0 | D_d28              | d28                                     | 28 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e817e1adeb2edc126145 | E                  | male        |            48.0 | E_d23              | d23                                     | 23 days since symptom onset        | Extensive Pneumonic infiltrates
--  5f21e818e1adeb2edc126148 | F                  | female      |            40.0 | F_d14              | d14                                     | 14 days since symptom onset        | Limited Pneumonic infiltrates
--  5f21e819e1adeb2edc12614b | G                  | female      |            59.0 | G_d22              | d22                                     | 22 days since symptom onset        | Limited Pneumonic infiltrates


-- Find which isotypes we have in this study (temporarily using supplied igblast outputs in the filters here, though we will replace with our own):

select
    seq."c_call",
    count(1)
from
    ireceptor_data.covid_sequences seq
    INNER JOIN ireceptor_data.covid_metadata meta on seq.repertoire_id = meta.repertoire_id
where
    seq.productive = TRUE
    AND seq.c_call is not NULL
    AND seq.locus = 'IGH'
    AND seq.cdr1 is not NULL
    AND seq.cdr1_aa is not NULL
    AND seq.cdr2 is not NULL
    AND seq.cdr2_aa is not NULL
    AND seq.cdr3 is not NULL
    AND seq.cdr3_aa is not NULL
    AND seq.repertoire_id IN (
        '5f21e814e1adeb2edc12613d',
        '5f21e815e1adeb2edc126140',
        '5f21e816e1adeb2edc126142',
        '5f21e817e1adeb2edc126144',
        '5f21e817e1adeb2edc126145',
        '5f21e818e1adeb2edc126148',
        '5f21e819e1adeb2edc12614b'
    )
    AND meta."study.study_id" = 'PRJNA648677'
GROUP BY
    seq."c_call"
order by
    2 desc;

--  c_call | count
-- --------+--------
--  IGHM   | 820510
--  IGHG1  | 337781
--  IGHG2  | 146475
--  IGHA1  |  92062
--  IGHD   |  54842
--  IGHA2  |  46086
--  IGHG3  |  18198
--  IGHG4  |   1139
--  IGHE   |    112
-- (9 rows)


--- ##########
--- ##########
--- ########## VDJSERVER:

-- Montague et al study, which is known to be IgG only.
-- See table S1 in https://www.medrxiv.org/content/10.1101/2020.07.13.20153114v1.full.pdf -- subjects 1-2 are mild disease, 3-14 are moderate, 15-19 are severe.
-- "Day 1 of clinical onset was defined as the first day of the appearance of clinical symptoms."

select
    "repertoire_id",
    "subject.subject_id",
    "subject.sex",
    "subject.age_min",
    "sample.0.sample_id",
    "sample.0.collection_time_point_relative",
    "subject.diagnosis.0.disease_length",
    "subject.diagnosis.0.disease_stage"
from
    ireceptor_data.covid_metadata
where
    "study.study_id" = 'PRJNA645245';


--              repertoire_id             | subject.subject_id | subject.sex | subject.age_min | sample.0.sample_id | sample.0.collection_time_point_relative | subject.diagnosis.0.disease_length | subject.diagnosis.0.disease_stage
-- ---------------------------------------+--------------------+-------------+-----------------+--------------------+-----------------------------------------+------------------------------------+-----------------------------------
--  5549400184724853226-242ac116-0001-012 | 1                  | female      |            62.0 | IgG24-2            | 43 days                                 |                                    |
--  5563272929090933226-242ac116-0001-012 | 1                  | female      |            62.0 | IgG24-1            | 43 days                                 |                                    |
--  5578047616589173226-242ac116-0001-012 | 1                  | female      |            62.0 | IgG21-2            | 22 days                                 |                                    |
--  5594669140024693226-242ac116-0001-012 | 1                  | female      |            62.0 | IgG21-1            | 22 days                                 |                                    |
--  5609272028831093226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG6-0             | 2 days                                  |                                    |
--  5624519162731893226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG15-2            | 34 days                                 |                                    |
--  5638220108406133226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG15-1            | 34 days                                 |                                    |
--  5651706305715573226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG15-0            | 34 days                                 |                                    |
--  5665063654006133226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG9-2             | 15 days                                 |                                    |
--  5692895042084213226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG9-1             | 15 days                                 |                                    |
--  5705736994299253226-242ac116-0001-012 | 2                  | female      |            37.0 | IgG9-0             | 15 days                                 |                                    |
--  5718836644552053226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG5-2             | 8 days                                  |                                    |
--  5731506798075253226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG5-1             | 8 days                                  |                                    |
--  5744907096038773226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG5-0             | 8 days                                  |                                    |
--  5758264444329333226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-2            | 38 days                                 |                                    |
--  5771922440330613226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-1            | 38 days                                 |                                    |
--  5786783027174773226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-0            | 38 days                                 |                                    |
--  5801214117289333226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG23-2            | 34 days                                 |                                    |
--  5814829163617653226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG23-1            | 34 days                                 |                                    |
--  5828143562235253226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG19-2            | 18 days                                 |                                    |
--  5842102205947253226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG19-1            | 18 days                                 |                                    |
--  5856060849659253226-242ac116-0001-012 | 5                  | female      |            72.0 | IgG7-0             | 10 days                                 |                                    |
--  5869504097295733226-242ac116-0001-012 | 5                  | female      |            72.0 | IgG13-0            | 27 days                                 |                                    |
--  5884579432504693226-242ac116-0001-012 | 6                  | male        |            56.0 | IgG8-0             | 13 days                                 |                                    |
--  5907944054594933226-242ac116-0001-012 | 6                  | male        |            56.0 | IgG16-0            | 28 days                                 |                                    |
--  5928130400886133226-242ac116-0001-012 | 7                  | female      |            55.0 | IgG18-0            | 39 days                                 |                                    |
--  5946684659604853226-242ac116-0001-012 | 7                  | female      |            55.0 | IgG11-0            | 16 days                                 |                                    |
--  5962618988273013226-242ac116-0001-012 | 7                  | female      |            55.0 | IgG10-0            | 11 days                                 |                                    |
--  5977479575117173226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG27-2            | 37 days                                 |                                    |
--  5994186997898613226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG27-1            | 37 days                                 |                                    |
--  6009734779510133226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG22-2            | 32 days                                 |                                    |
--  6028546736266613226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG22-1            | 32 days                                 |                                    |
--  6045683655777653226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG20-2            | 14 days                                 |                                    |
--  6059341651778933226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG20-1            | 14 days                                 |                                    |
--  6074502886333813226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG28-2            | 18 days                                 |                                    |
--  6089320523505013226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG28-1            | 18 days                                 |                                    |
--  6103966361984373226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG26-2            | 8 days                                  |                                    |
--  6119428244249973226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG26-1            | 8 days                                  |                                    |
--  6135190774226293226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG25-2            | 5 days                                  |                                    |
--  6148719921208693226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG25-1            | 5 days                                  |                                    |
--  6162506766228853226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG40-2            | 14 days                                 |                                    |
--  6180202031488373226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG40-1            | 14 days                                 |                                    |
--  6199958881049973226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG39-2            | 8 days                                  |                                    |
--  6219758680284533226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG39-1            | 8 days                                  |                                    |
--  6235177612877173226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG45-2            | 15 days                                 |                                    |
--  6255707556552053226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG45-1            | 15 days                                 |                                    |
--  6272329079987573226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG44-2            | 10 days                                 |                                    |
--  6286545421737333226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG44-1            | 10 days                                 |                                    |
--  6301749605965173226-242ac116-0001-012 | 12                 | female      |            48.0 | IgG34-2            | 11 days                                 |                                    |
--  6318371129400693226-242ac116-0001-012 | 12                 | female      |            48.0 | IgG34-1            | 11 days                                 |                                    |
--  6333961860685173226-242ac116-0001-012 | 12                 | female      |            48.0 | IgG33-2            | 7 days                                  |                                    |
--  6347448057994613226-242ac116-0001-012 | 12                 | female      |            48.0 | IgG33-1            | 7 days                                  |                                    |
--  6361063104322933226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG36-2            | 13 days                                 |                                    |
--  6374978798361973226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG36-1            | 13 days                                 |                                    |
--  6392545214602613226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG35-2            | 7 days                                  |                                    |
--  6410541127572853226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG35-1            | 7 days                                  |                                    |
--  6428064594140533226-242ac116-0001-012 | 14                 | female      |            51.0 | IgG38-2            | 7 days                                  |                                    |
--  6445072664632693226-242ac116-0001-012 | 14                 | female      |            51.0 | IgG38-1            | 7 days                                  |                                    |
--  6458344113577333226-242ac116-0001-012 | 14                 | female      |            51.0 | IgG37-2            | 3 days                                  |                                    |
--  6472560455327093226-242ac116-0001-012 | 14                 | female      |            51.0 | IgG37-1            | 3 days                                  |                                    |
--  6486218451328373226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG43-4            | 71 days                                 |                                    |
--  6500348893732213226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG43-3            | 71 days                                 |                                    |
--  6514393436790133226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG43-2            | 71 days                                 |                                    |
--  6528180281810293226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG43-1            | 71 days                                 |                                    |
--  6543040868654453226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG42-3            | 41 days                                 |                                    |
--  6563399013637493226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG42-2            | 41 days                                 |                                    |
--  6577873053425013226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG42-1            | 41 days                                 |                                    |
--  6592003495828853226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG41-2            | 39 days                                 |                                    |
--  6606649334308213226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG41-1            | 39 days                                 |                                    |
--  6625633089756533226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG30-2            | 53 days                                 |                                    |
--  6639763532160373226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG30-1            | 53 days                                 |                                    |
--  6654108722929013226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG29-2            | 44 days                                 |                                    |
--  6667637869911413226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG29-1            | 44 days                                 |                                    |
--  6680952268529013226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG32-3            | 36 days                                 |                                    |
--  6696585949486453226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG32-2            | 36 days                                 |                                    |
--  6710329844833653226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG32-1            | 36 days                                 |                                    |
--  6724073740180853226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG31-2            | 22 days                                 |                                    |
--  6738676628987253226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG31-1            | 22 days                                 |                                    |
--  6753537215831413226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-2            | 30 days                                 |                                    |
--  6768183054310773226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-1            | 30 days                                 |                                    |
--  6781110905871733226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-0            | 30 days                                 |                                    |
--  6795112499256693226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG12-2            | 8 days                                  |                                    |
--  6821827195837813226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG12-1            | 8 days                                  |                                    |
--  6835699940203893226-242ac116-0001-012 | 19                 | male        |            56.0 | IgG4-0             | 6 days                                  |                                    |
-- (84 rows)


-- Let's start with using these timepoints:

--              repertoire_id             | subject.subject_id | subject.sex | subject.age_min | sample.0.sample_id | sample.0.collection_time_point_relative | subject.diagnosis.0.disease_length | subject.diagnosis.0.disease_stage
-- ---------------------------------------+--------------------+-------------+-----------------+--------------------+-----------------------------------------+------------------------------------+-----------------------------------
--  5786783027174773226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-0            | 38 days                                 |                                    |
--  5842102205947253226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG19-1            | 18 days                                 |                                    |
--  5869504097295733226-242ac116-0001-012 | 5                  | female      |            72.0 | IgG13-0            | 27 days                                 |                                    |
--  5907944054594933226-242ac116-0001-012 | 6                  | male        |            56.0 | IgG16-0            | 28 days                                 |                                    |
--  5946684659604853226-242ac116-0001-012 | 7                  | female      |            55.0 | IgG11-0            | 16 days                                 |                                    |
--  6059341651778933226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG20-1            | 14 days                                 |                                    |
--  6074502886333813226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG28-2            | 18 days                                 |                                    |
--  6180202031488373226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG40-1            | 14 days                                 |                                    |
--  6255707556552053226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG45-1            | 15 days                                 |                                    |
--  6374978798361973226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG36-1            | 13 days                                 |                                    |
--  6606649334308213226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG41-1            | 39 days                                 |                                    |
--  6667637869911413226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG29-1            | 44 days                                 |                                    |
--  6738676628987253226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG31-1            | 22 days                                 |                                    |
--  6781110905871733226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-0            | 30 days                                 |                                    |


-- TODO: in future: let's add these replicates (but they have different repertoire_ids, so need to start using patient_id column instead in our analysis...)

--              repertoire_id             | subject.subject_id | subject.sex | subject.age_min | sample.0.sample_id | sample.0.collection_time_point_relative | subject.diagnosis.0.disease_length | subject.diagnosis.0.disease_stage
-- ---------------------------------------+--------------------+-------------+-----------------+--------------------+-----------------------------------------+------------------------------------+-----------------------------------

--  5758264444329333226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-2            | 38 days                                 |                                    |
--  5771922440330613226-242ac116-0001-012 | 3                  | male        |            47.0 | IgG14-1            | 38 days                                 |                                    |

--  5828143562235253226-242ac116-0001-012 | 4                  | female      |            73.0 | IgG19-2            | 18 days                                 |                                    |

--  6045683655777653226-242ac116-0001-012 | 8                  | male        |            37.0 | IgG20-2            | 14 days                                 |                                    |

--  6089320523505013226-242ac116-0001-012 | 9                  | female      |            52.0 | IgG28-1            | 18 days                                 |                                    |

--  6162506766228853226-242ac116-0001-012 | 10                 | male        |            27.0 | IgG40-2            | 14 days                                 |                                    |

--  6235177612877173226-242ac116-0001-012 | 11                 | female      |            20.0 | IgG45-2            | 15 days                                 |                                    |

--  6361063104322933226-242ac116-0001-012 | 13                 | female      |            44.0 | IgG36-2            | 13 days                                 |                                    |

--  6592003495828853226-242ac116-0001-012 | 15                 | female      |            68.0 | IgG41-2            | 39 days                                 |                                    |

--  6654108722929013226-242ac116-0001-012 | 16                 | male        |            75.0 | IgG29-2            | 44 days                                 |                                    |

--  6724073740180853226-242ac116-0001-012 | 17                 | male        |            60.0 | IgG31-2            | 22 days                                 |                                    |

--  6753537215831413226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-2            | 30 days                                 |                                    |
--  6768183054310773226-242ac116-0001-012 | 18                 | female      |            62.0 | IgG17-1            | 30 days                                 |                                    |
