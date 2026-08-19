"""Explicit AQD workflow entry points."""

from .run_imos_delivery import run_imos_delivery
from .run_proc1 import run_proc1
from .run_proc2 import run_proc2


def run_aqd_proc1(config, instrument_id=None, source_path=None):
    return run_proc1(config, instrument_id=instrument_id, source_path=source_path)


def run_aqd_proc2(config, instrument_id=None, input_dataset=None):
    return run_proc2(config, instrument_id=instrument_id, input_dataset=input_dataset)


def run_aqd_delivery(config, instrument_id=None, input_dataset=None):
    return run_imos_delivery(config, instrument_id=instrument_id, input_dataset=input_dataset)
