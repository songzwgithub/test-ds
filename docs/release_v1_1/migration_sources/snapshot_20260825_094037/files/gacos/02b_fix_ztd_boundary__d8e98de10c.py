from pathlib import Path

p = Path("pystamps/pipeline/gacos_correction.py")
text = p.read_text(encoding="utf-8")

old = '''    grid = np.memmap(product.path, dtype=dtype, mode="r", shape=(length, width))
    row = (lat - y_first) / y_step
    col = (lon - x_first) / x_step
    sampled = _sample_normalized(grid, row, col)
'''

new = '''    grid = np.memmap(
        product.path,
        dtype=dtype,
        mode="r",
        shape=(length, width),
    )

    row = (
        np.asarray(lat, dtype=np.float64)
        - y_first
    ) / y_step

    col = (
        np.asarray(lon, dtype=np.float64)
        - x_first
    ) / x_step

    # Floating-point arithmetic can move an exact boundary
    # coordinate a few ulps outside the valid grid, e.g.
    # 2.0 -> 2.00000000000003. Snap only coordinates that are
    # within a tiny tolerance of the raster boundary. Truly
    # out-of-coverage coordinates remain outside and therefore
    # still return NaN.
    boundary_tol = 1.0e-7

    row_near = (
        (row >= -boundary_tol)
        & (row <= (length - 1) + boundary_tol)
    )

    col_near = (
        (col >= -boundary_tol)
        & (col <= (width - 1) + boundary_tol)
    )

    row = np.where(
        row_near,
        np.clip(
            row,
            0.0,
            float(length - 1),
        ),
        row,
    )

    col = np.where(
        col_near,
        np.clip(
            col,
            0.0,
            float(width - 1),
        ),
        col,
    )

    sampled = _sample_normalized(
        grid,
        row,
        col,
    )
'''

n = text.count(old)

if n != 1:
    raise RuntimeError(
        f"expected exactly 1 ZTD sampling block, found {n}"
    )

text = text.replace(old, new, 1)

p.write_text(
    text,
    encoding="utf-8",
)

print("02b ZTD BOUNDARY FIX: PASS")
