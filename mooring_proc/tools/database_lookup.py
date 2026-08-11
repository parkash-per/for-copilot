"""Metadata lookup stubs for mooring_proc."""

# TODO: Connect metadata lookup functions to the project metadata source.


def load_metadata_table(source, table_name=None):
    """Load a metadata table from a configured source."""
    raise NotImplementedError("TODO: implement metadata table loading")


def get_instrument_context(metadata_table, instrument_id, deployment_id=None):
    """Return contextual metadata for a single instrument deployment."""
    raise NotImplementedError("TODO: implement instrument context lookup")


def update_metadata_file_fields(metadata_table, file_record, updates):
    """Update file-oriented metadata fields in a metadata table."""
    raise NotImplementedError("TODO: implement metadata field updates")


def list_instruments(metadata_table, instrument_type=None, active_only=False):
    """List available instruments from the metadata table."""
    raise NotImplementedError("TODO: implement instrument listing")
