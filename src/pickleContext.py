from dataclasses import dataclass

@dataclass
class PickleContext:
    run_idx: int
    cp_file: str