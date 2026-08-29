from pathlib import Path

from pypsds.stages import apply_reference


def test_auto_reference_manifest_is_none_safe():
    source = Path(
        apply_reference.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "PYPSDS_AUTO_REFERENCE_MANIFEST_NONE_SAFE_V2" in source

    assert (
        "None if args.center_row is None "
        "else int(args.center_row)"
    ) in source

    assert (
        "None if args.center_col is None "
        "else int(args.center_col)"
    ) in source

    assert (
        "None if args.point_ids_file is not None "
        "else int(args.half_row)"
    ) in source

    assert (
        "None if args.point_ids_file is not None "
        "else int(args.half_col)"
    ) in source

    assert '"reference_method"' in source
    assert '"point_ids_file"' in source
