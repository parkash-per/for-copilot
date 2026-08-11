# AQD workflow

```python
from mooring_proc.tools.workflows.run_proc1 import run_proc1
from mooring_proc.tools.workflows.run_proc2 import run_proc2
from mooring_proc.tools.workflows.run_imos_delivery import run_imos_delivery

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

proc_1_result = run_proc1(config)
proc_2_result = run_proc2(config, input_dataset=proc_1_result["output_path"])
delivery_result = run_imos_delivery(config)

print(proc_2_result["manual_qc_log"])
print(delivery_result["imos_deliverables_file"])
```
