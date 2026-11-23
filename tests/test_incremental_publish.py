"""Incremental publish: verdict reuse between successful publications.

Scenario behind these tests: after a documentary gets an intro added, the
material that follows only shifts on the finished-film timeline while its
source ranges stay the same.  A re-publish must then

* reuse the whole-line / partial-line verdict from the last successful
  publication for blocks whose identity (reel, subtitle version, source range)
  is unchanged,
* re-project every occurrence onto the new finished-film timecode,
* re-check blocks that changed,
* skip reading subtitle line detail for reels whose blocks are all reused,
* keep serving the previous successful publication on rejection or write
  failure.
"""

from __future__ import annotations

import contextlib
import sqlite3
import uuid

import pytest

from app import db, store
from tests.test_api import full_setup, make_film, make_reel, put_edit, put_subtitles, seg


def _publish(client, delivery_id):
    response = client.post(f"/deliveries/{delivery_id}/publish")
    assert response.status_code == 200, response.text
    return response.json()


def _new_delivery_and_publish(client, film_id):
    created = client.post(f"/films/{film_id}/deliveries")
    assert created.status_code == 201, created.text
    return _publish(client, created.json()["id"])


def test_output_shift_reuses_verdict_and_reprojects_timecode(client):
    reel, film = full_setup(client)
    first = _new_delivery_and_publish(client, film)

    # Same source ranges, everything moved 3s later on the finished film.
    put_edit(
        client,
        film,
        [
            seg(reel, 0, 4_000, 3_000),
            seg(reel, 5_000, 7_000, 7_000),
        ],
    )
    second = _new_delivery_and_publish(client, film)

    assert second["output_version"] == 2
    assert [c["text"] for c in second["subtitles"]] == ["第一句", "第二句", "第三句"]
    assert [(c["start_ms"], c["end_ms"]) for c in second["subtitles"]] == [
        (4_000, 5_000),
        (6_000, 7_000),
        (7_000, 8_000),
    ]
    assert second["reviews"] == []


def test_reused_partial_line_keeps_verdict_and_shifts_keep_window(client):
    reel = make_reel(client)
    put_subtitles(
        client,
        reel,
        [{"number": 7, "start_ms": 1_000, "end_ms": 3_000, "text": "半句台词"}],
    )
    film = make_film(client)
    put_edit(client, film, [seg(reel, 0, 2_000, 0)])
    first = _new_delivery_and_publish(client, film)
    assert first["reviews"][0]["line_number"] == 7

    put_edit(client, film, [seg(reel, 0, 2_000, 10_000)])
    second = _new_delivery_and_publish(client, film)

    assert len(second["subtitles"]) == 0
    assert second["reviews"] == [
        {
            "source_reel_id": reel,
            "source_reel_name": "reel-a",
            "line_number": 7,
            "text": "半句台词",
            "keep_start_ms": 11_000,
            "keep_end_ms": 12_000,
        }
    ]


def test_every_occurrence_is_reprojected_independently(client):
    reel = make_reel(client)
    put_subtitles(
        client,
        reel,
        [{"number": 1, "start_ms": 0, "end_ms": 2_000, "text": "重复"}],
    )
    film = make_film(client)
    put_edit(client, film, [seg(reel, 0, 2_000, 0)])
    _new_delivery_and_publish(client, film)

    # The same source range is cut in twice, both at brand-new positions.
    put_edit(
        client,
        film,
        [seg(reel, 0, 2_000, 4_000), seg(reel, 0, 2_000, 20_000)],
    )
    second = _new_delivery_and_publish(client, film)

    assert [(c["start_ms"], c["end_ms"]) for c in second["subtitles"]] == [
        (4_000, 6_000),
        (20_000, 22_000),
    ]
    assert [c["line_number"] for c in second["subtitles"]] == [1, 1]


def test_fully_reused_reel_skips_subtitle_line_reads(client, monkeypatch):
    reel, film = full_setup(client)
    _new_delivery_and_publish(client, film)

    put_edit(
        client,
        film,
        [
            seg(reel, 0, 4_000, 9_000),
            seg(reel, 5_000, 7_000, 13_000),
        ],
    )

    calls: list[tuple[int, int]] = []
    original = store.load_subtitle_lines

    def spy(conn, reel_id, version):
        calls.append((reel_id, version))
        return original(conn, reel_id, version)

    monkeypatch.setattr(store, "load_subtitle_lines", spy)
    second = _new_delivery_and_publish(client, film)

    assert calls == []  # every block reused: reel detail never read
    assert [c["text"] for c in second["subtitles"]] == ["第一句", "第二句", "第三句"]


def test_partially_reused_reel_reads_lines_only_for_fresh_blocks(client, monkeypatch):
    reel = make_reel(client)
    put_subtitles(
        client,
        reel,
        [
            {"number": 1, "start_ms": 0, "end_ms": 1_000, "text": "第一句"},
            {"number": 2, "start_ms": 2_000, "end_ms": 3_000, "text": "第二句"},
        ],
    )
    film = make_film(client)
    put_edit(
        client,
        film,
        [seg(reel, 0, 1_000, 0), seg(reel, 2_000, 3_000, 1_000)],
    )
    _new_delivery_and_publish(client, film)

    # First block keeps its source range; the second block covers new material.
    put_edit(
        client,
        film,
        [seg(reel, 0, 1_000, 5_000), seg(reel, 2_500, 3_000, 6_000)],
    )

    calls: list[int] = []
    original = store.load_subtitle_lines

    def spy(conn, reel_id, version):
        calls.append(reel_id)
        return original(conn, reel_id, version)

    monkeypatch.setattr(store, "load_subtitle_lines", spy)
    second = _new_delivery_and_publish(client, film)

    assert calls == [reel]  # one read for the reel, reused block still shifts
    shifted = {c["line_number"]: (c["start_ms"], c["end_ms"]) for c in second["subtitles"]}
    assert shifted[1] == (5_000, 6_000)  # reused verdict, new position
    # line 2 [2000,3000) intersects the fresh block [2500,3000) only partially
    review = second["reviews"][0]
    assert review["line_number"] == 2
    assert (review["keep_start_ms"], review["keep_end_ms"]) == (6_000, 6_500)


def test_changed_subtitle_version_forces_full_recheck(client):
    reel, film = full_setup(client)
    _new_delivery_and_publish(client, film)

    # Same edit table, but a new subtitle version invalidates the cached keys.
    put_subtitles(
        client,
        reel,
        [
            {"number": 1, "start_ms": 1_000, "end_ms": 2_000, "text": "第一句 v2"},
            {"number": 2, "start_ms": 3_000, "end_ms": 4_000, "text": "第二句 v2"},
            {"number": 3, "start_ms": 5_000, "end_ms": 6_000, "text": "第三句 v2"},
        ],
    )
    second = _new_delivery_and_publish(client, film)

    assert [c["text"] for c in second["subtitles"]] == [
        "第一句 v2",
        "第二句 v2",
        "第三句 v2",
    ]


def test_legacy_publication_without_blocks_triggers_full_recheck(client, monkeypatch):
    reel, film = full_setup(client)
    first = _new_delivery_and_publish(client, film)

    # Simulate a pre-upgrade publication: no frozen block verdicts.
    with db.write_conn() as conn:
        conn.execute(
            "DELETE FROM publication_blocks WHERE publication_id = ?",
            (first["id"],),
        )

    put_edit(
        client,
        film,
        [seg(reel, 0, 4_000, 8_000), seg(reel, 5_000, 7_000, 12_000)],
    )

    calls: list[int] = []
    original = store.load_subtitle_lines

    def spy(conn, reel_id, version):
        calls.append(reel_id)
        return original(conn, reel_id, version)

    monkeypatch.setattr(store, "load_subtitle_lines", spy)
    second = _new_delivery_and_publish(client, film)

    assert calls == [reel]  # nothing reusable: full re-check
    assert [(c["start_ms"], c["end_ms"]) for c in second["subtitles"]] == [
        (9_000, 10_000),
        (11_000, 12_000),
        (12_000, 13_000),
    ]

    # The old publication remains fully queryable.
    legacy = client.get(f"/publications/{first['id']}")
    assert legacy.status_code == 200
    assert [c["text"] for c in legacy.json()["subtitles"]] == [
        "第一句",
        "第二句",
        "第三句",
    ]


def test_resegmented_but_identical_merged_block_is_reused(client, monkeypatch):
    """The cache key is the merged block, so re-splitting segments is invisible."""

    reel = make_reel(client)
    put_subtitles(
        client,
        reel,
        [{"number": 1, "start_ms": 500, "end_ms": 1_500, "text": "跨段台词"}],
    )
    film = make_film(client)
    put_edit(
        client,
        film,
        [seg(reel, 0, 1_000, 0), seg(reel, 1_000, 2_000, 1_000)],
    )
    first = _new_delivery_and_publish(client, film)
    assert [(c["start_ms"], c["end_ms"]) for c in first["subtitles"]] == [(500, 1_500)]

    # Same merged source range [0, 2000), submitted as a single segment shifted
    # by 4s on the finished film.
    put_edit(client, film, [seg(reel, 0, 2_000, 4_000)])

    calls: list[int] = []
    original = store.load_subtitle_lines

    def spy(conn, reel_id, version):
        calls.append(reel_id)
        return original(conn, reel_id, version)

    monkeypatch.setattr(store, "load_subtitle_lines", spy)
    second = _new_delivery_and_publish(client, film)

    assert calls == []  # merged block identity unchanged -> fully reused
    assert [(c["start_ms"], c["end_ms"]) for c in second["subtitles"]] == [
        (4_500, 5_500)
    ]


def test_verdicts_survive_restart(tmp_path, monkeypatch):
    db_path = tmp_path / f"restart-{uuid.uuid4().hex}.db"
    monkeypatch.setenv("SUBTITLE_DB_PATH", str(db_path))

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as first_client:
        reel, film = full_setup(first_client)
        _new_delivery_and_publish(first_client, film)
        put_edit(
            first_client,
            film,
            [seg(reel, 0, 4_000, 2_000), seg(reel, 5_000, 7_000, 6_000)],
        )

    # A brand-new client/process against the same database file.
    with TestClient(app) as restarted:
        second = _new_delivery_and_publish(restarted, film)
        assert [(c["start_ms"], c["end_ms"]) for c in second["subtitles"]] == [
            (3_000, 4_000),
            (5_000, 6_000),
            (6_000, 7_000),
        ]


def test_write_failure_rolls_back_and_keeps_last_publication(client, monkeypatch):
    reel, film = full_setup(client)
    first = _new_delivery_and_publish(client, film)

    put_edit(
        client,
        film,
        [seg(reel, 0, 4_000, 4_000), seg(reel, 5_000, 7_000, 8_000)],
    )
    delivery = client.post(f"/films/{film}/deliveries").json()

    real_write_conn = db.write_conn

    @contextlib.contextmanager
    def failing_write_conn(path=None):
        with real_write_conn(path) as conn:
            real_execute = conn.execute

            def execute(sql, params=()):
                if sql.lstrip().startswith("INSERT INTO publication_blocks"):
                    raise sqlite3.OperationalError("simulated disk failure")
                return real_execute(sql, params)

            # sqlite3 connections are immutable, so intercept via a thin proxy.
            class Proxy:
                def __getattr__(self, name):
                    return execute if name == "execute" else getattr(conn, name)

            yield Proxy()

    monkeypatch.setattr(db, "write_conn", failing_write_conn)

    with pytest.raises(sqlite3.OperationalError):
        client.post(f"/deliveries/{delivery['id']}/publish")

    # The failed delivery is still open and the broadcast side keeps version 1.
    state = client.get(f"/deliveries/{delivery['id']}").json()
    assert state["state"] == "open"
    assert state["output_version"] is None

    latest = client.get(f"/films/{film}/subtitles/latest").json()
    assert latest["output_version"] == first["output_version"] == 1
    assert [c["text"] for c in latest["subtitles"]] == ["第一句", "第二句", "第三句"]


def test_reel_skips_lines_when_all_its_blocks_reused_but_other_reel_checks(
    client, monkeypatch
):
    reel_a = make_reel(client, "reel-a")
    reel_b = make_reel(client, "reel-b")
    put_subtitles(
        client,
        reel_a,
        [{"number": 1, "start_ms": 0, "end_ms": 1_000, "text": "A卷台词"}],
    )
    put_subtitles(
        client,
        reel_b,
        [{"number": 1, "start_ms": 0, "end_ms": 1_000, "text": "B卷台词"}],
    )
    film = make_film(client)
    put_edit(
        client,
        film,
        [
            seg(reel_a, 0, 1_000, 0),
            seg(reel_b, 0, 1_000, 1_000),
        ],
    )
    _new_delivery_and_publish(client, film)

    # A keeps its source range (only output shifts); B covers new source material.
    put_edit(
        client,
        film,
        [
            seg(reel_a, 0, 1_000, 9_000),
            seg(reel_b, 500, 1_000, 10_000),
        ],
    )

    read_reels: list[int] = []
    original = store.load_subtitle_lines

    def spy(conn, reel_id, version):
        read_reels.append(reel_id)
        return original(conn, reel_id, version)

    monkeypatch.setattr(store, "load_subtitle_lines", spy)
    second = _new_delivery_and_publish(client, film)

    assert read_reels == [reel_b]  # A fully reused, B re-checked
    a_cue = next(c for c in second["subtitles"] if c["source_reel_id"] == reel_a)
    assert (a_cue["start_ms"], a_cue["end_ms"]) == (9_000, 10_000)
    # B line [0,1000) now only partially retained [500,1000) -> review
    b_review = next(r for r in second["reviews"] if r["source_reel_id"] == reel_b)
    assert (b_review["keep_start_ms"], b_review["keep_end_ms"]) == (10_000, 10_500)


def test_rejection_leaves_last_successful_publication_untouched(client):
    reel, film = full_setup(client)
    first = _new_delivery_and_publish(client, film)

    delivery = client.post(f"/films/{film}/deliveries").json()
    put_subtitles(
        client,
        reel,
        [{"number": 1, "start_ms": 1_000, "end_ms": 2_000, "text": "换版了"}],
    )

    rejected = client.post(f"/deliveries/{delivery['id']}/publish")
    assert rejected.status_code == 409

    latest = client.get(f"/films/{film}/subtitles/latest").json()
    assert latest["output_version"] == 1
    assert latest["publication_id"] == first["id"]
    assert [c["text"] for c in latest["subtitles"]] == ["第一句", "第二句", "第三句"]
