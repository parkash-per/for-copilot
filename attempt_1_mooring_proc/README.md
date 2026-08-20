# Instrument workflows

```python
from mooring_proc.tools.workflows import (
    run_aqd_proc1, run_aqd_proc2, run_aqd_delivery,
    run_sbe26_proc1, run_sbe26_proc2, run_sbe26_delivery,
    run_sbe37_proc1, run_sbe37_proc2, run_sbe37_delivery,
    run_rbrq_proc1, run_rbrq_proc2, run_rbrq_delivery,
    run_sig500_proc1, run_sig500_proc2, run_sig500_delivery,
)

config = {
    "metadata_csv": "/absolute/path/to/metadata.csv",
    "inst_deploy_ID": "AQD-EXAMPLE",
    "manual_qc_flags": [
        {
            "qc_vars": ["DEPTH_quality_control"],
            "flag": 4,
            "start": "2024-07-31 06:00:00",
            "end": "2024-08-01 00:00:00",
            "comment": "sensor failed",
        }
    ],
}

proc_1_result = run_aqd_proc1(config)
proc_2_result = run_aqd_proc2(config, input_dataset=proc_1_result["output_path"])
delivery_result = run_aqd_delivery(config)

print(proc_2_result["manual_qc_log"])
print(delivery_result["imos_deliverables_file"])
```

Each instrument follows the same 3 entry points:

- `run_<instrument>_proc1`: parse + deployment trim (writes `proc_1`)
- `run_<instrument>_proc2`: applies manual QC windows (writes `proc_2`)
- `run_<instrument>_delivery`: shared IMOS delivery
  - `proc_1` publishes as FV00
  - `proc_2` publishes as FV01 when present
