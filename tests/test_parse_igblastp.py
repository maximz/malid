from io import StringIO
from malid import parse_igblastp
import pandas as pd

sample_fasta = """>testsequence
QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQASGQGLEWMGWMNPNSANPGYAQKFQGRVTMTRNTSISTAFMELSSLRSDDTAVYYCARARVTIHYDILTGYYSNAFDIWGQGTMVAVSS
>testsequence2
QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQASGQGLEWMGWMNPNSANPGYAQKFQGRVTMTRNTSISTAFMELSSLRSDDTAVYYCARARVTIHYDILTGYYSNAFDIWGQGTMVAVSS
"""

sample_parse = """# IGBLASTP 2.2.29+
# Query: testsequence
# Database: human_gl_V
# Domain classification requested: imgt

# Alignment summary between query and top germline V gene hit (from, to, length, matches, mismatches, gaps, percent identity)
FR1-IMGT	1	25	25	25	0	0	100
CDR1-IMGT	26	33	8	8	0	0	100
FR2-IMGT	34	50	17	16	1	0	94.1
CDR2-IMGT	51	58	8	6	2	0	75
FR3-IMGT	59	96	38	36	2	0	94.7
CDR3-IMGT (germline)	97	98	2	2	0	0	100
Total	N/A	N/A	98	93	5	0	94.9

# Hit table (the first field indicates the chain type of the hit)
# Fields: query id, subject id
# 1 hits found
V	testsequence	IGHV1-8*01
# IGBLASTP 2.2.29+
# Query: testsequence2
# Database: human_gl_V
# Domain classification requested: imgt

# Alignment summary between query and top germline V gene hit (from, to, length, matches, mismatches, gaps, percent identity)
FR1-IMGT	1	25	25	25	0	0	100
CDR1-IMGT	26	33	8	8	0	0	100
FR2-IMGT	34	50	17	16	1	0	94.1
CDR2-IMGT	51	58	8	6	2	0	75
FR3-IMGT	59	96	38	36	2	0	94.7
CDR3-IMGT (germline)	97	98	2	2	0	0	100
Total	N/A	N/A	98	93	5	0	94.9

# Hit table (the first field indicates the chain type of the hit)
# Fields: query id, subject id
# 1 hits found
V	testsequence2	IGHV1-8*01
# BLAST processed 2 queries
"""


def test_igblastp_parsing():
    file_handle_fasta = StringIO(sample_fasta)
    file_handle_parse = StringIO(sample_parse)
    results = parse_igblastp.process(
        file_handle_fasta=file_handle_fasta, file_handle_parse=file_handle_parse
    )
    print(results)
    expected_df = pd.DataFrame.from_dict(
        {
            0: {
                "type": "V",
                "query_id": "testsequence",
                "v_gene": "IGHV1-8*01",
                "global_percent_identity": 94.88,
                "FR1-IMGT": "QVQLVQSGAEVKKPGASVKVSCKAS",
                "CDR1-IMGT": "GYTFTSYD",
                "FR2-IMGT": "INWVRQASGQGLEWMGW",
                "CDR2-IMGT": "MNPNSANP",
                "FR3-IMGT": "GYAQKFQGRVTMTRNTSISTAFMELSSLRSDDTAVYYC",
                "CDR3-IMGT (germline)": "AR",
            },
            1: {
                "type": "V",
                "query_id": "testsequence2",
                "v_gene": "IGHV1-8*01",
                "global_percent_identity": 94.88,
                "FR1-IMGT": "QVQLVQSGAEVKKPGASVKVSCKAS",
                "CDR1-IMGT": "GYTFTSYD",
                "FR2-IMGT": "INWVRQASGQGLEWMGW",
                "CDR2-IMGT": "MNPNSANP",
                "FR3-IMGT": "GYAQKFQGRVTMTRNTSISTAFMELSSLRSDDTAVYYC",
                "CDR3-IMGT (germline)": "AR",
            },
        },
        orient="index",
    )

    # Compare the actual results with the expected dataframe
    # Use pd.testing.assert_frame_equal with a tolerance level for the floating point comparison (check_less_precise=2 means compare only up to 2 decimal places).
    pd.testing.assert_frame_equal(results, expected_df, check_less_precise=2)
