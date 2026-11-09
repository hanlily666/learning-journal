"""Evaluation scores from the judge's matches, on hand-worked examples."""

from evaluation.scoring import Judged, score


def test_precision_recall_per_field_classification_and_duplicates():
    rag = Judged(
        session_id="a", expected_label="ai", predicted_labels=["ai", "ai"],
        labels=[{"field": "concept", "text": "RAG"}, {"field": "concept", "text": "Embeddings"},
                {"field": "gap", "text": "How retrieval is scored"}],
        predicted=[{"field": "concept", "text": "Retrieval-augmented generation", "segment": 0},
                   {"field": "concept", "text": "What RAG is", "segment": 1},
                   {"field": "explore", "text": "Chunking", "segment": 1}],
        # predicted 0 matches label 0; predicted 1 also claims label 0 (ignored: one-to-one);
        # predicted 2 "matches" a label of another field (ignored).
        matches=[(0, 0), (1, 0), (2, 2)],
        duplicates=[(0, 1)],  # across the two segments of a split session
    )
    french = Judged(session_id="b", expected_label="french", predicted_labels=["messy"],
                    labels=[], predicted=[], matches=[], duplicates=[])

    report = score([rag, french])

    assert report["classification_accuracy"] == 0.6667  # 2 of 3 segments
    assert report["fields"]["concept"] == {"predicted": 2, "labelled": 2, "matched": 1,
                                           "precision": 0.5, "recall": 0.5}
    assert report["fields"]["gap"] == {"predicted": 0, "labelled": 1, "matched": 0,
                                       "precision": None, "recall": 0.0}
    assert report["fields"]["explore"]["precision"] == 0.0
    assert report["duplicates_across_segments"] == 1


def test_extraction_is_scored_on_ai_sessions_and_other_items_are_counted_apart():
    messy = Judged(session_id="m", expected_label="messy", predicted_labels=["ai"], labels=[],
                   predicted=[{"field": "concept", "text": "Pizza", "segment": 0}],
                   matches=[], duplicates=[])
    baseline = Judged(session_id="a", expected_label="ai", predicted_labels=[],
                      labels=[{"field": "concept", "text": "RAG"}],
                      predicted=[{"field": "concept", "text": "RAG", "segment": 0}],
                      matches=[(0, 0)], duplicates=[])

    report = score([messy, baseline])

    assert report["fields"]["concept"]["precision"] == 1.0
    assert report["items_from_non_ai_sessions"] == 1
    assert report["classification_accuracy"] == 0.0  # the baseline doesn't classify


def test_duplicates_within_one_segment_are_not_counted_as_split_duplicates():
    judged = Judged(session_id="a", expected_label="ai", predicted_labels=["ai"], labels=[],
                    predicted=[{"field": "concept", "text": "x", "segment": 0},
                               {"field": "concept", "text": "x again", "segment": 0}],
                    matches=[], duplicates=[(0, 1)])
    assert score([judged])["duplicates_across_segments"] == 0
