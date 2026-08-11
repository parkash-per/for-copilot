"""
NetCDF Schema Manager - loads and caches YAML configuration files
defining variable schemas for each instrument type.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class NetCDFSchemaManager:
    """Loads and manages NetCDF variable schemas from YAML config files."""
    
    def __init__(self, config_dir: str = "configs"):
        # Resolve config_dir relative to this file's location if it's relative
        config_path = Path(config_dir)
        if not config_path.is_absolute():
            # Get the directory where config_manager.py is located
            manager_dir = Path(__file__).parent
            config_path = manager_dir / config_dir
        
        self.config_dir = config_path.resolve()
        self._schemas = {}
        self._global_attrs = None
    
    def load_schema(self, instrument_name: str) -> Dict[str, Any]:
        """Load schema from YAML, with caching."""
        if instrument_name in self._schemas:
            return self._schemas[instrument_name]
        
        config_file = self.config_dir / f"{instrument_name.lower()}_schema.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Schema not found: {config_file}")
        
        with open(config_file, "r") as f:
            schema = yaml.safe_load(f)
        
        self._schemas[instrument_name] = schema
        return schema
    
    def load_global_attributes(self) -> Dict[str, Any]:
        """Load global IMOS attributes from YAML, with caching."""
        if self._global_attrs is not None:
            return self._global_attrs
        
        global_attrs_file = self.config_dir / "global_attributes.yaml"
        if not global_attrs_file.exists():
            raise FileNotFoundError(f"Global attributes file not found: {global_attrs_file}")
        
        with open(global_attrs_file, "r") as f:
            self._global_attrs = yaml.safe_load(f)
        
        return self._global_attrs
    
    def get_input_variables(self, instrument_name: str) -> Dict[str, Dict[str, Any]]:
        """Get variables to extract from input NetCDF."""
        schema = self.load_schema(instrument_name)
        return schema.get("input_variables", {})
    
    def get_output_variables(self, instrument_name: str) -> Dict[str, Dict[str, Any]]:
        """Get variables to create in output NetCDF."""
        schema = self.load_schema(instrument_name)
        return schema.get("output_variables", {})
    
    def get_default_channel_code(self, instrument_name: str) -> str:
        """Get instrument-specific default channel code."""
        schema = self.load_schema(instrument_name)
        return schema.get("default_channel_code", "")
    
    def get_default_keywords(self, instrument_name: str) -> str:
        """Get instrument-specific default keywords."""
        schema = self.load_schema(instrument_name)
        return schema.get("default_keywords", "")
    
    def get_default_title(self, instrument_name: str) -> str:
        """Get instrument-specific default title template."""
        schema = self.load_schema(instrument_name)
        return schema.get("default_title", "")


# Global schema manager instance
_schema_manager = None


def get_schema_manager(config_dir: str = "configs") -> NetCDFSchemaManager:
    """Get or create global schema manager."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = NetCDFSchemaManager(config_dir)
    return _schema_manager