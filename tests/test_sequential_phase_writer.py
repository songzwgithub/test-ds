import numpy as np
import pytest

from pypsds.phase_linking.sequential_phase_writer import (
    SequentialPhaseWriter,
)


def _phase(a):
    return np.exp(
        1j
        *
        np.asarray(
            a,
            dtype=np.float32,
        )
    ).astype(
        np.complex64
    )


def test_writer_m19_style_two_stages(
    tmp_path,
):

    path = (
        tmp_path
        /
        "linked_phase.npy"
    )

    # Small analogue of:
    #
    # N=38
    # stage0 real 0:19
    # stage1 real 19:38
    #
    # Here:
    # N=6
    # stage0 = 0:3
    # stage1 = 3:6

    rr = np.array(
        [1, 2],
        dtype=np.int32,
    )

    cc = np.array(
        [2, 3],
        dtype=np.int32,
    )

    p0 = _phase(
        [
            [0.0, 0.2, 0.4],
            [0.0, 0.5, 0.8],
        ]
    )

    p1 = _phase(
        [
            [0.6, 0.9, 1.1],
            [1.0, 1.2, 1.4],
        ]
    )

    with SequentialPhaseWriter(
        path,
        ndate=6,
        rows=4,
        cols=5,
        overwrite=True,
    ) as writer:

        assert writer.shape == (
            6,
            4,
            5,
        )

        assert (
            writer.dtype
            ==
            np.dtype(
                np.complex64
            )
        )

        writer(
            stage_index=0,
            real_indices=(0, 1, 2),
            rows=rr,
            cols=cc,
            phase=p0,
        )

        writer(
            stage_index=1,
            real_indices=(3, 4, 5),
            rows=rr,
            cols=cc,
            phase=p1,
        )

        writer.flush()

        np.testing.assert_array_equal(
            writer.written_counts,
            np.array(
                [2, 2, 2, 2, 2, 2],
                dtype=np.int64,
            ),
        )

    got = np.load(
        path,
        mmap_mode="r",
    )

    assert got.shape == (
        6,
        4,
        5,
    )

    for j in range(3):

        np.testing.assert_allclose(
            got[
                j,
                rr,
                cc,
            ],
            p0[:, j],
            rtol=0,
            atol=0,
        )

        np.testing.assert_allclose(
            got[
                j + 3,
                rr,
                cc,
            ],
            p1[:, j],
            rtol=0,
            atol=0,
        )

    # Unwritten background stays NaN.
    assert np.isnan(
        got[:, 0, 0].real
    ).all()


def test_writer_rejects_duplicate_write(
    tmp_path,
):

    path = (
        tmp_path
        /
        "linked_phase.npy"
    )

    rr = np.array(
        [1],
        dtype=np.int32,
    )

    cc = np.array(
        [1],
        dtype=np.int32,
    )

    ph = _phase(
        [[0.5]]
    )

    writer = SequentialPhaseWriter(
        path,
        ndate=2,
        rows=3,
        cols=3,
        overwrite=True,
    )

    writer(
        stage_index=0,
        real_indices=(0,),
        rows=rr,
        cols=cc,
        phase=ph,
    )

    with pytest.raises(
        RuntimeError,
        match="overwrite",
    ):
        writer(
            stage_index=0,
            real_indices=(0,),
            rows=rr,
            cols=cc,
            phase=ph,
        )

    writer.close()


def test_writer_resume_existing_cube(
    tmp_path,
):

    path = (
        tmp_path
        /
        "linked_phase.npy"
    )

    rr = np.array(
        [1],
        dtype=np.int32,
    )

    cc = np.array(
        [2],
        dtype=np.int32,
    )

    p0 = _phase(
        [[0.2]]
    )

    p1 = _phase(
        [[0.7]]
    )

    with SequentialPhaseWriter(
        path,
        ndate=2,
        rows=3,
        cols=4,
        overwrite=True,
    ) as w:

        w(
            stage_index=0,
            real_indices=(0,),
            rows=rr,
            cols=cc,
            phase=p0,
        )

    # Re-open without overwrite and write
    # another acquisition.
    with SequentialPhaseWriter(
        path,
        ndate=2,
        rows=3,
        cols=4,
        overwrite=False,
    ) as w:

        w(
            stage_index=1,
            real_indices=(1,),
            rows=rr,
            cols=cc,
            phase=p1,
        )

    got = np.load(
        path
    )

    np.testing.assert_allclose(
        got[0, rr, cc],
        p0[:, 0],
        rtol=0,
        atol=0,
    )

    np.testing.assert_allclose(
        got[1, rr, cc],
        p1[:, 0],
        rtol=0,
        atol=0,
    )
