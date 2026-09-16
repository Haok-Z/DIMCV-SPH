import json
from pathlib import Path


def resolve_output_paths(scene_file_path, config_values, method_name):
    """Build per-scene result directories with optional configuration override.

    ``outputName`` can be placed in Configuration (or SegmentConfiguration for
    standalone Segment scenes). Without it, the JSON file stem is used.
    """
    scene_stem = Path(scene_file_path).stem
    output_name = str(config_values.get("outputName", scene_stem) or scene_stem)
    output_root = Path(config_values.get("outputRoot", "results"))
    method_root = output_root / output_name / method_name
    return method_root / "images", method_root / "ply"


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.scene_file_path = scene_file_path
        self.config = None
        with open(scene_file_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        print(self.config)
    
    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["Configuration"]
        if name not in self.config["Configuration"]:
            if enforce_exist:
                assert name in self.config["Configuration"]
            else:
                return None
        return self.config["Configuration"][name]
    
    def get_rigid_bodies(self):
        if "RigidBodies" in self.config:
            return self.config["RigidBodies"]
        else:
            return []
    
    def get_rigid_blocks(self):
        if "RigidBlocks" in self.config:
            return self.config["RigidBlocks"]
        else:
            return []
    
    def get_fluid_blocks(self):
        if "FluidBlocks" in self.config:
            return self.config["FluidBlocks"]
        else:
            return []
        
    def get_fluid_emitters(self):
        if "FluidEmitters" in self.config:
            return self.config["FluidEmitters"]
        else:
            return []