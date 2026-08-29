from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


CONFIGS = (
    ROOT
    / "pypsds"
    / "resources"
    / "default_config.yaml",
)


def test_public_config_contract():

    for path in CONFIGS:

        cfg = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

        assert (
            cfg["schema_version"]
            ==
            1
        )


        paths = cfg["paths"]

        assert (
            paths["rslc_dir"]
            is None
        )

        assert (
            paths["rslc_tab"]
            is None
        )

        assert (
            paths["dem_dir"]
            is None
        )

        assert (
            paths["gacos_dir"]
            is None
        )

        assert (
            paths["scratch_dir"]
            ==
            "output/.scratch"
        )

        assert (
            paths["products_dir"]
            ==
            "output/products"
        )


        # Public production defaults intentionally enable
        # the validated scientific correction chain.

        assert (
            cfg["corrections"]
            ["scla"]
            ["mode"]
            ==
            'stamps'
        )

        assert (
            cfg["corrections"]
            ["atmosphere"]
            ["mode"]
            ==
            'gacos'
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["mode"]
            ==
            'stamps'
        )


        assert (
            cfg["corrections"]
            ["atmosphere"]
            ["backend"]
            ==
            "gacos"
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["temporal_window_days"]
            ==
            365.0
        )

        assert (
            cfg["corrections"]
            ["scn"]
            ["wavelength_m"]
            ==
            100.0
        )


        products = cfg["products"]

        assert (
            products["point"]["enabled"]
            is True
        )

        assert (
            products["point"]["crs"]
            ==
            "EPSG:4326"
        )

        assert (
            products["quicklook"]["enabled"]
            is False
        )

        assert (
            products["quicklook"]["scientific_product"]
            is False
        )


def test_public_templates_contain_no_absolute_project_paths():

    forbidden = (
        "/home/ubuntu/",
        "/mnt/",
        "/media/",
    )

    for path in CONFIGS:

        text = path.read_text(
            encoding="utf-8"
        )

        for token in forbidden:

            assert token not in text
