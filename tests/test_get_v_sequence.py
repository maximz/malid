from malid import get_v_sequence


def test_get_tcrb_v_gene_annotations():
    generated_list = get_v_sequence.get_tcrb_v_gene_annotations()
    assert (
        generated_list[generated_list["v_call"] == "TRBV26*01"]["cdr1_aa"].iloc[0]
        == "MNHVT"
    )
    assert (
        generated_list[generated_list["v_call"] == "TRBV26*01"]["cdr2_aa"].iloc[0]
        == "SPGTGS"
    )
