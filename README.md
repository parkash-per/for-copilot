# mooring_proc

Minimal scaffold for a mooring data processing project.

## Processing phases

- `proc_1`: initial ingestion, instrument-specific parsing, and early QC setup.
- `proc_2`: derived processing, harmonisation, and preparation for downstream delivery.
- `imos_delivery`: packaging processed outputs for IMOS-style publication and compliance checks.

## Reference instrument

AQD is treated as the reference instrument for the initial project layout and pipeline naming. Other instrument pipelines follow the same placeholder structure so processing patterns can stay aligned where appropriate.

## FAIR and reproducibility goals

This repository is structured to support:

- findable and reusable metadata through schema-driven configuration,
- reproducible processing steps via explicit workflow modules,
- auditable QC through manual flagging and validation placeholders,
- consistent delivery outputs for future compliance and publishing tasks.

At this stage the repository contains lightweight placeholders only; implementation logic is intentionally deferred.
