import json


class SegmentConfig:
    def __init__(self, scene_file_path: str) -> None:
        with open(scene_file_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def get_cfg(self, name: str, default=None):
        seg_cfg = self.config.get("SegmentConfiguration", {})
        return seg_cfg.get(name, default)

    def get_domain_start(self):
        return self.get_cfg("domainStart", [0.0, 0.0, 0.0])

    def get_domain_end(self):
        return self.get_cfg("domainEnd", [1.0, 1.0, 1.0])

    def get_obstacles(self):
        return self.config.get("RigidBlocks", [])

    def get_inflow(self):
        return self.config.get("FluidEmitters", [])
