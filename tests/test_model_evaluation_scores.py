import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score

import malid.external


def test_auprc():
    ## Generate sample data: "muddied final column" from test_adjust_model_decision_thresholds:

    # first, clear diagonal
    # how many entries per class are clear diagonal
    n_diagonal_clear = 100
    labels = np.array([0, 1, 2, 3])
    # make labels categorical
    labels = np.array(["class" + str(i) for i in labels])

    clear_diagonal_probas = np.vstack(
        [
            np.tile([0.7, 0.1, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.7, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.7, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.1, 0.7], n_diagonal_clear).reshape(-1, 4),
        ]
    )
    clear_diagonal_trues = np.hstack(
        [np.tile([lbl], n_diagonal_clear) for lbl in labels]
    )

    # now, muddy up the final column:
    # all predictions consistent, but ground truth is a toss up
    n_muddy = 100
    muddy_final_row_probas = np.tile([0.1, 0.1, 0.1, 0.7], n_muddy * 4).reshape(-1, 4)
    muddy_final_row_trues = np.hstack([np.tile([lbl], n_muddy) for lbl in labels])

    y_score = np.vstack([clear_diagonal_probas, muddy_final_row_probas])
    y_true = np.hstack([clear_diagonal_trues, muddy_final_row_trues])

    # Sanity check accuracy score
    y_score_argmax = labels[y_score.argmax(axis=1)]
    assert accuracy_score(y_true, y_score_argmax) == 0.625

    # make sure same result even if labels come in different order
    labels_reordered = np.array(["class3", "class0", "class1", "class2"])
    y_score_reordered = pd.DataFrame(y_score, columns=labels)[labels_reordered].values

    assert np.allclose(
        [
            malid.external.model_evaluation_scores.auprc(y_true, y_score),
            malid.external.model_evaluation_scores.auprc(
                y_true, y_score, labels=labels
            ),
            malid.external.model_evaluation_scores.auprc(
                y_true, y_score_reordered, labels=labels_reordered
            ),
        ],
        0.72916,
    )

    assert not np.allclose(
        [
            malid.external.model_evaluation_scores.auprc(
                y_true, y_score_reordered
            ),  # no label order provided, so the assumed one will be wrong
            malid.external.model_evaluation_scores.auprc(
                y_true, y_score, labels=labels_reordered
            ),  # wrong label order provided
            malid.external.model_evaluation_scores.auprc(
                y_true, y_score_reordered, labels=labels
            ),  # wrong label order provided
        ],
        0.72916,
    )

    # make sure it supports binary too, whether y_score is provided as 2d or 1d array (sklearn style)
    chosen_labels = labels[:2]
    y_true_subselect = pd.Series(y_true)
    subselect_mask = y_true_subselect.isin(chosen_labels)
    y_true_subselect = y_true_subselect[subselect_mask].values
    y_score_subselect = pd.DataFrame(y_score, columns=labels)
    y_score_subselect = y_score_subselect[chosen_labels].loc[subselect_mask].values

    assert np.allclose(
        [
            malid.external.model_evaluation_scores.auprc(
                y_true_subselect, y_score_subselect
            ),
            malid.external.model_evaluation_scores.auprc(
                y_true_subselect, y_score_subselect, labels=chosen_labels
            ),
            malid.external.model_evaluation_scores.auprc(
                y_true_subselect, y_score_subselect[:, 1]
            ),
            malid.external.model_evaluation_scores.auprc(
                y_true_subselect, y_score_subselect[:, 1], labels=chosen_labels
            ),
            average_precision_score(
                y_true_subselect, y_score_subselect[:, 1], pos_label=chosen_labels[1]
            ),
            average_precision_score(
                y_true_subselect, y_score_subselect[:, 0], pos_label=chosen_labels[0]
            ),
        ],
        0.75,
    )
