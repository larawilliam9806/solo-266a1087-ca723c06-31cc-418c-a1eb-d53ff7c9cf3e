"""Tests for block-level verdicts and output projection.

The incremental publish path reuses source-level verdicts
(:class:`engine.BlockItem`) and only re-projects finished-film timecodes, so
the two halves of that behaviour are tested here in isolation.
"""

from __future__ import annotations

from app import engine


def seg(reel, s, e, out):
    return engine.Segment(
        source_reel_id=reel, source_start_ms=s, source_end_ms=e, output_start_ms=out
    )


def line(n, s, e, text=None):
    return engine.Line(number=n, start_ms=s, end_ms=e, text=text or f"line {n}")


def test_items_for_block_marks_fully_covered_line_as_cue():
    block = engine.Block(
        source_reel_id=1, source_start_ms=0, source_end_ms=5_000, output_start_ms=0
    )
    items = engine.items_for_block(block, [line(1, 1_000, 2_000, "整句")])

    assert len(items) == 1
    item = items[0]
    assert item.kind == engine.CUE
    assert (item.line_number, item.source_start_ms, item.source_end_ms) == (1, 1_000, 2_000)
    assert item.text == "整句"


def test_items_for_block_marks_partial_intersection_as_review_with_source_range():
    block = engine.Block(
        source_reel_id=1, source_start_ms=0, source_end_ms=2_000, output_start_ms=0
    )
    items = engine.items_for_block(block, [line(7, 1_000, 3_000, "半句")])

    assert len(items) == 1
    item = items[0]
    assert item.kind == engine.REVIEW
    # kept on the source reel: [1000, 2000)
    assert (item.source_start_ms, item.source_end_ms) == (1_000, 2_000)


def test_items_for_block_ignores_exclusive_boundary_touches():
    block = engine.Block(
        source_reel_id=1, source_start_ms=1_000, source_end_ms=2_000, output_start_ms=0
    )
    items = engine.items_for_block(
        block, [line(1, 0, 1_000), line(2, 2_000, 3_000)]
    )
    assert items == []


def test_project_items_recomputes_output_timecode_per_position():
    items = [
        engine.BlockItem(
            kind=engine.CUE,
            line_number=1,
            text="整句",
            source_start_ms=1_000,
            source_end_ms=2_000,
        ),
        engine.BlockItem(
            kind=engine.REVIEW,
            line_number=2,
            text="半句",
            source_start_ms=1_500,
            source_end_ms=2_000,
        ),
    ]
    block = engine.Block(
        source_reel_id=1, source_start_ms=0, source_end_ms=4_000, output_start_ms=9_000
    )

    cues, reviews = engine.project_items(block, items)

    assert [(c.start_ms, c.end_ms) for c in cues] == [(10_000, 11_000)]
    assert [(r.keep_start_ms, r.keep_end_ms) for r in reviews] == [(10_500, 11_000)]


def test_same_verdict_projects_differently_after_a_position_shift():
    """The documentary intro case: source range unchanged, output shifts right."""

    items = engine.items_for_block(
        engine.Block(1, 0, 4_000, 0), [line(3, 500, 1_500)]
    )
    assert items[0].kind == engine.CUE

    first = engine.Block(1, 0, 4_000, output_start_ms=0)
    shifted = engine.Block(1, 0, 4_000, output_start_ms=6_000)

    cues_before, _ = engine.project_items(first, items)
    cues_after, _ = engine.project_items(shifted, items)

    assert [(c.start_ms, c.end_ms) for c in cues_before] == [(500, 1_500)]
    assert [(c.start_ms, c.end_ms) for c in cues_after] == [(6_500, 7_500)]


def test_refactored_compute_matches_block_item_pipeline():
    segments = [
        seg(1, 0, 1_000, 0),
        seg(1, 1_000, 2_000, 1_000),
        seg(2, 0, 1_000, 5_000),
        seg(1, 0, 1_000, 8_000),
    ]
    lines = {
        1: [line(1, 500, 1_500), line(2, 0, 400)],
        2: [line(1, 100, 900)],
    }

    cues, reviews = engine.compute(segments, lines)

    blocks = engine.build_blocks(segments)
    pipe_cues, pipe_reviews = [], []
    for block in blocks:
        block_cues, block_reviews = engine.project_items(
            block, engine.items_for_block(block, lines.get(block.source_reel_id, []))
        )
        pipe_cues.extend(block_cues)
        pipe_reviews.extend(block_reviews)
    pipe_cues.sort(key=lambda c: (c.start_ms, c.end_ms, c.source_reel_id, c.line_number))
    pipe_reviews.sort(
        key=lambda r: (r.keep_start_ms, r.keep_end_ms, r.source_reel_id, r.line_number)
    )

    assert pipe_cues == cues
    assert pipe_reviews == reviews
