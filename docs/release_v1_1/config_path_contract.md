# pyPSDS-GAMMA v1.1 project configuration contract

## Design goal

A new study area must be runnable without modifying Python source code.

The project configuration is located in the study-area directory and all
relative paths are resolved from the project/data roots.

## Required inputs

- `RSLC/`
- `RSLC_tab`

## Optional auxiliary inputs

- `DEM_prep/`
- `GACOS/`

The auxiliary directories are only required when the corresponding
processing feature is enabled.

## Managed outputs

- `output/`
- `output/.scratch/`
- `output/products/`

## Scientific correction defaults

For backward compatibility and reproducibility, v1.1 does not silently
enable new corrections.

The public defaults remain:

- SCLA: disabled
- atmospheric correction: disabled
- SCN: disabled

A production configuration must enable these explicitly.

## Point-first product policy

The primary geodetic product is the original PS/DS point product.

Supported public point formats:

- Parquet
- GeoPackage
- CSV

Raster products are optional quicklooks and are not the authoritative
scientific product.

## Stage naming

The existing computational stage names remain unchanged during the v1.1
migration.

Development identifiers are removed from the public interface only after
the corresponding tests and production policies have been migrated.
