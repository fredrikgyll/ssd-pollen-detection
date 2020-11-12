from pathlib import Path


class Logger:
    def __init__(self, save_dir) -> None:
        self.log_file: Path = save_dir / 'log.txt'

    def __call__(self, msg: str) -> None:
        print(msg)
        with self.log_file.open('a') as f:
            f.write(msg + '\n')
