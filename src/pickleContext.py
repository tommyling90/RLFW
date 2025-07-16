from dataclasses import dataclass

@dataclass
class PickleContext:
    game_idx: int
    run_idx: int
    cp_file: str

    def reset_after_run(self):
        self.iter_idx = 0

    def reset_after_game(self):
        self.run_idx = 0
        self.iter_idx = 0