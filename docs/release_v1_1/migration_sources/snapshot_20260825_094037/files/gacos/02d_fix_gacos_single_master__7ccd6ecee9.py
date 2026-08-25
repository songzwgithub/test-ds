from pathlib import Path
import re

p = Path("pystamps/pipeline/gacos_correction.py")
text = p.read_text(encoding="utf-8")


def replace_function(source, name, new_code):
    pattern = re.compile(
        rf"^def {re.escape(name)}\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        raise RuntimeError(
            f"function not found: {name}"
        )

    start = m.start()

    next_def = re.compile(
        r"^(?:def|class) [A-Za-z_]",
        re.MULTILINE,
    ).search(source, m.end())

    end = (
        next_def.start()
        if next_def
        else len(source)
    )

    return (
        source[:start]
        + new_code.rstrip()
        + "\n\n\n"
        + source[end:]
    )


# ------------------------------------------------------------
# load_sbas_network is no longer required for phuw2 correction.
# ------------------------------------------------------------

text = text.replace(
    "from pystamps.pipeline.stage6_sbas "
    "import load_sbas_network\n",
    "",
)


# ============================================================
# Sign QA for SINGLE-MASTER phuw2
# ============================================================

new_choose_sign = r'''
def _choose_sign(
    ph_sm: np.ndarray,
    los_delay: np.memmap,
    master0: int,
    wavelength_m: float,
    config: ResolvedGacosConfig,
) -> tuple[str, dict[str, float]]:

    if config.sign in {
        "subtract",
        "add",
    }:
        return config.sign, {
            "selection": "forced"
        }

    n_ps, n_image = ph_sm.shape

    ps_ix = np.linspace(
        0,
        n_ps - 1,
        min(config.qa_ps, n_ps),
        dtype=np.int64,
    )

    image_pool = np.setdiff1d(
        np.arange(
            n_image,
            dtype=np.int64,
        ),
        np.asarray(
            [master0],
            dtype=np.int64,
        ),
    )

    if image_pool.size == 0:
        raise GacosCorrectionError(
            "No non-master acquisitions "
            "available for GACOS sign QA"
        )

    image_ix = image_pool[
        np.linspace(
            0,
            image_pool.size - 1,
            min(
                config.qa_ifg,
                image_pool.size,
            ),
            dtype=np.int64,
        )
    ]

    phase_scale = (
        4.0
        * math.pi
        / wavelength_m
    )

    scores_raw = []
    scores_subtract = []
    scores_add = []
    correlations = []

    master_delay = np.asarray(
        los_delay[
            ps_ix,
            master0,
        ],
        dtype=np.float64,
    )

    for j in image_ix:

        atmospheric = (
            phase_scale
            * (
                np.asarray(
                    los_delay[
                        ps_ix,
                        j,
                    ],
                    dtype=np.float64,
                )
                - master_delay
            )
        )

        raw = np.asarray(
            ph_sm[
                ps_ix,
                j,
            ],
            dtype=np.float64,
        )

        valid = (
            np.isfinite(raw)
            & np.isfinite(atmospheric)
        )

        if np.count_nonzero(valid) < 100:
            continue

        r = raw[valid].copy()
        a = atmospheric[valid].copy()

        r -= np.nanmedian(r)
        a -= np.nanmedian(a)

        scores_raw.append(
            float(
                _robust_scale(
                    r,
                    axis=0,
                )
            )
        )

        scores_subtract.append(
            float(
                _robust_scale(
                    r - a,
                    axis=0,
                )
            )
        )

        scores_add.append(
            float(
                _robust_scale(
                    r + a,
                    axis=0,
                )
            )
        )

        if (
            np.nanstd(r) > 0
            and np.nanstd(a) > 0
        ):
            correlations.append(
                float(
                    np.corrcoef(
                        r,
                        a,
                    )[0, 1]
                )
            )

    if not scores_subtract:
        raise GacosCorrectionError(
            "Unable to determine GACOS sign "
            "from valid single-master samples"
        )

    raw_score = float(
        np.nanmedian(
            scores_raw
        )
    )

    subtract_score = float(
        np.nanmedian(
            scores_subtract
        )
    )

    add_score = float(
        np.nanmedian(
            scores_add
        )
    )

    chosen = (
        "subtract"
        if subtract_score <= add_score
        else "add"
    )

    best_score = min(
        subtract_score,
        add_score,
    )

    improvement = (
        100.0
        * (
            raw_score
            - best_score
        )
        / raw_score
        if raw_score > 0
        else 0.0
    )

    return chosen, {
        "selection":
            "auto_single_master_robust_spatial_scale",
        "raw_score_rad":
            raw_score,
        "subtract_score_rad":
            subtract_score,
        "add_score_rad":
            add_score,
        "chosen_improvement_percent":
            improvement,
        "median_raw_atmosphere_correlation":
            (
                float(
                    np.nanmedian(
                        correlations
                    )
                )
                if correlations
                else float("nan")
            ),
        "qa_image_count":
            len(scores_subtract),
        "qa_ps_count":
            int(ps_ix.size),
    }
'''

text = replace_function(
    text,
    "_choose_sign",
    new_choose_sign,
)


# ============================================================
# Write SINGLE-MASTER corrected phuw2
# ============================================================

new_writer = r'''
def _write_hdf5_mat(
    path: Path,
    raw_phase: np.ndarray,
    msd: np.ndarray,
    unwrap_ifg_index_sm: Any,
    los_delay: np.memmap,
    master0: int,
    wavelength_m: float,
    sign: str,
    chunk_ps: int,
) -> None:

    try:
        import h5py
    except Exception as exc:
        raise GacosCorrectionError(
            "Writing phuw2_gacos.mat "
            "requires h5py"
        ) from exc

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    n_ps, n_image = raw_phase.shape

    phase_scale = (
        4.0
        * math.pi
        / wavelength_m
    )

    sign_factor = (
        -1.0
        if sign == "subtract"
        else 1.0
    )

    with h5py.File(
        tmp,
        "w",
    ) as h5:

        dset = h5.create_dataset(
            "ph_uw",
            shape=(
                n_ps,
                n_image,
            ),
            dtype=np.float32,
            chunks=(
                min(
                    chunk_ps,
                    n_ps,
                ),
                min(
                    32,
                    n_image,
                ),
            ),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        dset.attrs[
            "PY_STAMPS_row_major"
        ] = np.asarray(
            1,
            dtype=np.uint8,
        )

        for start in range(
            0,
            n_ps,
            chunk_ps,
        ):

            stop = min(
                start + chunk_ps,
                n_ps,
            )

            delay = np.asarray(
                los_delay[
                    start:stop,
                    :,
                ],
                dtype=np.float64,
            )

            atmospheric = (
                phase_scale
                * (
                    delay
                    - delay[
                        :,
                        master0,
                    ][:, None]
                )
            )

            corrected = np.asarray(
                raw_phase[
                    start:stop,
                    :,
                ],
                dtype=np.float64,
            )

            corrected += (
                sign_factor
                * atmospheric
            )

            dset[
                start:stop,
                :,
            ] = corrected.astype(
                np.float32
            )

            print(
                "[GACOS][WRITE] "
                f"{stop}/{n_ps} "
                f"({100.0*stop/n_ps:.1f}%)",
                flush=True,
            )

        msd_arr = np.asarray(
            msd,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        )

        msd_dset = h5.create_dataset(
            "msd",
            data=msd_arr,
        )

        msd_dset.attrs[
            "PY_STAMPS_row_major"
        ] = np.asarray(
            1,
            dtype=np.uint8,
        )

        if (
            unwrap_ifg_index_sm is not None
            and np.asarray(
                unwrap_ifg_index_sm
            ).size
        ):
            unwrap_arr = np.asarray(
                unwrap_ifg_index_sm,
                dtype=np.int64,
            ).reshape(
                -1,
                1,
            )

            unwrap_dset = (
                h5.create_dataset(
                    "unwrap_ifg_index_sm",
                    data=unwrap_arr,
                )
            )

            unwrap_dset.attrs[
                "PY_STAMPS_row_major"
            ] = np.asarray(
                1,
                dtype=np.uint8,
            )

        h5.attrs[
            "gacos_corrected"
        ] = np.asarray(
            1,
            dtype=np.uint8,
        )

        h5.attrs[
            "gacos_sign"
        ] = sign

        h5.attrs[
            "gacos_reference_mode"
        ] = "single_master"

        h5.attrs[
            "gacos_master_index_1based"
        ] = int(master0 + 1)

        h5.attrs[
            "wavelength_m"
        ] = float(
            wavelength_m
        )

    os.replace(
        tmp,
        path,
    )
'''

text = replace_function(
    text,
    "_write_hdf5_mat",
    new_writer,
)


# ============================================================
# Replace production entrypoint
# ============================================================

new_ensure = r'''
def ensure_gacos_corrected_phuw(
    dataset_root: Path,
    settings: PipelineGacosConfig | None = None,
) -> Path:
    """
    Create/reuse phuw2_gacos.mat.

    phuw2 is a SINGLE-MASTER phase matrix:
        shape = n_ps x n_image

    GACOS correction for acquisition i is referenced
    to the same master acquisition:
        atm_i = 4*pi/lambda * (LOS_i - LOS_master)
    """

    root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
    )

    config = _load_config(
        root,
        settings,
    )

    output = (
        root
        / "phuw2_gacos.mat"
    )

    debug_path = (
        root
        / "gacos_correction_debug.json"
    )

    inventory_csv = (
        root
        / "gacos_date_inventory.csv"
    )

    work_dir = (
        root
        / "_gacos_work"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = (
        "ps2.mat",
        "phuw2.mat",
        "parms.mat",
    )

    missing = [
        name
        for name in required
        if not (
            root / name
        ).exists()
    ]

    if missing:
        raise GacosCorrectionError(
            "Missing GACOS correction inputs: "
            + ", ".join(missing)
        )

    # ========================================================
    # StaMPS geometry
    # ========================================================

    ps2 = read_mat(
        root / "ps2.mat"
    )

    n_ps = int(
        round(
            _scalar(
                ps2.get("n_ps"),
                0,
            )
        )
    )

    n_image = int(
        round(
            _scalar(
                ps2.get("n_image"),
                0,
            )
        )
    )

    n_ifg = int(
        round(
            _scalar(
                ps2.get("n_ifg"),
                0,
            )
        )
    )

    master_ix = int(
        round(
            _scalar(
                ps2.get("master_ix"),
                1,
            )
        )
    )

    if n_ps <= 0:
        raise GacosCorrectionError(
            "ps2.mat contains invalid n_ps"
        )

    if n_image <= 1:
        raise GacosCorrectionError(
            "ps2.mat contains invalid n_image"
        )

    if not (
        1 <= master_ix <= n_image
    ):
        raise GacosCorrectionError(
            "ps2.mat contains invalid master_ix"
        )

    master0 = (
        master_ix - 1
    )

    day = np.asarray(
        ps2.get("day"),
        dtype=np.float64,
    ).reshape(-1)

    if day.size != n_image:
        raise GacosCorrectionError(
            f"ps2.day length {day.size} "
            f"!= n_image {n_image}"
        )

    dates = _day_labels(
        day
    )

    lonlat = _as_matrix(
        ps2.get("lonlat"),
        n_ps,
        "ps2.lonlat",
        np.float64,
    )

    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    # ========================================================
    # phuw2 = n_ps x n_image
    # ========================================================

    phuw = read_mat_variables(
        root / "phuw2.mat",
        (
            "ph_uw",
            "msd",
            "unwrap_ifg_index_sm",
        ),
    )

    ph_sm = _as_matrix(
        phuw["ph_uw"],
        n_ps,
        "phuw2.ph_uw",
        np.float32,
    )

    if ph_sm.shape != (
        n_ps,
        n_image,
    ):
        raise GacosCorrectionError(
            "phuw2.ph_uw must be "
            "n_ps x n_image; "
            f"got {ph_sm.shape}, "
            f"expected ({n_ps}, {n_image})"
        )

    msd = np.asarray(
        phuw.get(
            "msd",
            np.zeros(
                n_image,
                dtype=np.float32,
            ),
        ),
        dtype=np.float32,
    ).reshape(-1)

    if msd.size != n_image:
        msd = np.zeros(
            n_image,
            dtype=np.float32,
        )

    unwrap_ifg_index_sm = (
        phuw.get(
            "unwrap_ifg_index_sm"
        )
    )

    # ========================================================
    # GACOS inventory
    # ========================================================

    products = discover_products(
        config.gacos_dir
    )

    missing_dates = [
        date
        for date in dates
        if date not in products
    ]

    if missing_dates:

        preview = ", ".join(
            missing_dates[:20]
        )

        if not config.strict_dates:
            raise GacosCorrectionError(
                "gacos.strict_dates=false "
                "does not enable temporal "
                "interpolation or partial "
                "correction. "
                f"Missing "
                f"{len(missing_dates)}/{n_image} "
                f"dates: {preview}"
            )

        raise GacosCorrectionError(
            f"Missing "
            f"{len(missing_dates)}/{n_image} "
            f"GACOS acquisition dates: "
            f"{preview}"
        )

    fingerprint = (
        _inventory_fingerprint(
            products,
            dates,
        )
    )

    source_stat = (
        root
        / "phuw2.mat"
    ).stat()

    cache_signature = {
        "phase_mode":
            "single_master",
        "phuw2_size":
            source_stat.st_size,
        "phuw2_mtime_ns":
            source_stat.st_mtime_ns,
        "inventory_fingerprint":
            fingerprint,
        "unit":
            config.product_unit,
        "projection":
            config.projection,
        "min_valid_fraction":
            config.min_valid_fraction,
        "sign_requested":
            config.sign,
        "incidence_tif":
            (
                str(config.incidence_tif)
                if config.incidence_tif
                else None
            ),
        "incidence_deg":
            config.incidence_deg,
        "master_ix":
            master_ix,
        "n_image":
            n_image,
    }

    if (
        output.exists()
        and debug_path.exists()
        and not config.rebuild
    ):
        try:
            existing = json.loads(
                debug_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                existing.get("status")
                == "completed"
                and existing.get(
                    "cache_signature"
                )
                == cache_signature
            ):
                print(
                    "[GACOS] Reusing "
                    f"completed correction: "
                    f"{output}",
                    flush=True,
                )
                return output
        except Exception:
            pass

    started = (
        time.perf_counter()
    )

    parms = read_mat(
        root / "parms.mat"
    )

    wavelength_m = _scalar(
        parms.get("lambda"),
        0.0555,
    )

    if not (
        0.001
        < wavelength_m
        < 1.0
    ):
        raise GacosCorrectionError(
            "Invalid radar wavelength: "
            f"{wavelength_m}"
        )

    incidence, incidence_source = (
        _resolve_incidence(
            config,
            ps2,
            parms,
            lon,
            lat,
        )
    )

    if (
        config.projection
        == "zenith"
    ):
        cosine = np.cos(
            incidence
        )

        if (
            np.any(
                ~np.isfinite(
                    cosine
                )
            )
            or np.nanmin(
                cosine
            )
            <= 0.05
        ):
            raise GacosCorrectionError(
                "Invalid incidence angles "
                "for zenith-to-LOS "
                "projection"
            )

    else:
        cosine = np.ones(
            n_ps,
            dtype=np.float64,
        )

    ref_ix, reference_source = (
        _reference_indices(
            ps2,
            parms,
            n_ps,
        )
    )

    # Delay per PS per acquisition.
    delay_path = (
        work_dir
        / "gacos_los_ref.f32"
    )

    los_delay = np.memmap(
        delay_path,
        dtype=np.float32,
        mode="w+",
        shape=(
            n_ps,
            n_image,
        ),
    )

    los_delay[:] = np.nan
    los_delay.flush()

    valid_counts = {}
    metadata_by_kind = {}

    for index, date in enumerate(
        dates
    ):

        product = products[date]

        sampled, meta = (
            sample_product(
                product,
                lon,
                lat,
                config.product_unit,
            )
        )

        metadata_by_kind.setdefault(
            product.kind,
            meta,
        )

        los = (
            sampled
            / cosine
        )

        finite_los = (
            np.isfinite(los)
        )

        valid_fraction = float(
            np.count_nonzero(
                finite_los
            )
            / n_ps
        )

        if (
            valid_fraction
            < config.min_valid_fraction
        ):
            raise GacosCorrectionError(
                f"GACOS coverage for "
                f"{date} is only "
                f"{100*valid_fraction:.2f}% "
                f"(< "
                f"{100*config.min_valid_fraction:.2f}%)"
            )

        if not np.all(
            finite_los
        ):
            from scipy.spatial import (
                cKDTree,
            )

            valid_ix = np.flatnonzero(
                finite_los
            )

            missing_ix = np.flatnonzero(
                ~finite_los
            )

            cos_lat = math.cos(
                math.radians(
                    float(
                        np.nanmedian(
                            lat
                        )
                    )
                )
            )

            xy = np.column_stack(
                (
                    lon * cos_lat,
                    lat,
                )
            )

            tree = cKDTree(
                xy[
                    valid_ix,
                    :,
                ]
            )

            _distance, nearest = (
                tree.query(
                    xy[
                        missing_ix,
                        :,
                    ],
                    k=1,
                )
            )

            los[
                missing_ix
            ] = los[
                valid_ix[
                    np.asarray(
                        nearest,
                        dtype=np.int64,
                    )
                ]
            ]

        valid_ref = ref_ix[
            np.isfinite(
                los[ref_ix]
            )
        ]

        if (
            valid_ref.size
            == 0
        ):
            raise GacosCorrectionError(
                "No finite GACOS values "
                "in reference region "
                f"for {date}"
            )

        reference_value = float(
            np.nanmedian(
                los[
                    valid_ref
                ]
            )
        )

        # Same spatial reference as InSAR.
        los -= reference_value

        los_delay[
            :,
            index,
        ] = los.astype(
            np.float32
        )

        los_delay.flush()

        valid_counts[date] = int(
            np.count_nonzero(
                np.isfinite(
                    sampled
                )
            )
        )

        print(
            "[GACOS][SAMPLE] "
            f"{index+1}/{n_image} "
            f"{date} "
            f"kind={product.kind} "
            f"valid="
            f"{valid_counts[date]}/{n_ps}",
            flush=True,
        )

    _write_inventory_csv(
        inventory_csv,
        dates,
        products,
        valid_counts,
    )

    valid_date_columns = np.all(
        np.isfinite(
            los_delay
        ),
        axis=0,
    )

    if not np.all(
        valid_date_columns
    ):
        bad = [
            dates[i]
            for i
            in np.flatnonzero(
                ~valid_date_columns
            )
        ]

        raise GacosCorrectionError(
            "GACOS PS sampling "
            f"incomplete for "
            f"{len(bad)} dates; "
            f"first: {bad[:10]}"
        )

    # ========================================================
    # Sign selection from single-master phase
    # ========================================================

    sign, qa = _choose_sign(
        ph_sm,
        los_delay,
        master0,
        wavelength_m,
        config,
    )

    print(
        "[GACOS] correction sign: "
        f"{sign}; QA={qa}",
        flush=True,
    )

    _write_hdf5_mat(
        output,
        ph_sm,
        msd,
        unwrap_ifg_index_sm,
        los_delay,
        master0,
        wavelength_m,
        sign,
        config.chunk_ps,
    )

    kind_counts = {
        "tif": sum(
            1
            for date in dates
            if products[
                date
            ].kind
            == "tif"
        ),
        "ztd": sum(
            1
            for date in dates
            if products[
                date
            ].kind
            == "ztd"
        ),
    }

    debug = {
        "status":
            "completed",
        "dataset_root":
            str(root),
        "gacos_dir":
            str(
                config.gacos_dir
            ),
        "output":
            str(output),
        "source_phuw":
            str(
                root
                / "phuw2.mat"
            ),
        "phase_mode":
            "single_master",
        "n_ps":
            n_ps,
        "n_ifg":
            n_ifg,
        "n_image":
            n_image,
        "master_ix":
            master_ix,
        "master_date":
            dates[master0],
        "dates_start":
            dates[0],
        "dates_end":
            dates[-1],
        "product_kind_counts":
            kind_counts,
        "missing_dates":
            missing_dates,
        "product_unit_requested":
            config.product_unit,
        "min_valid_fraction":
            config.min_valid_fraction,
        "projection":
            config.projection,
        "incidence_source":
            incidence_source,
        "incidence_deg_median":
            (
                float(
                    np.nanmedian(
                        np.rad2deg(
                            incidence
                        )
                    )
                )
                if config.projection
                == "zenith"
                else None
            ),
        "reference_source":
            reference_source,
        "reference_ps":
            int(
                ref_ix.size
            ),
        "correction_sign":
            sign,
        "sign_qa":
            qa,
        "wavelength_m":
            wavelength_m,
        "cache_signature":
            cache_signature,
        "sample_metadata":
            metadata_by_kind,
        "work_delay_file":
            str(delay_path),
        "inventory_csv":
            str(
                inventory_csv
            ),
        "duration_sec":
            (
                time.perf_counter()
                - started
            ),
        "phase_formula":
            (
                "ph_corr(i) = ph_raw(i) "
                "+ sign * 4*pi/lambda * "
                "[(LOS_i-ref_i) - "
                "(LOS_master-ref_master)]"
            ),
        "note":
            (
                "GACOS correction is "
                "applied to single-master "
                "phuw2 after Stage 6; "
                "original phuw2.mat is "
                "preserved."
            ),
    }

    _write_json(
        debug_path,
        debug,
    )

    print(
        "[GACOS] completed: "
        f"{output}",
        flush=True,
    )

    return output
'''

text = replace_function(
    text,
    "ensure_gacos_corrected_phuw",
    new_ensure,
)

p.write_text(
    text,
    encoding="utf-8",
)

print(
    "02d SINGLE-MASTER GACOS FIX: PASS"
)
