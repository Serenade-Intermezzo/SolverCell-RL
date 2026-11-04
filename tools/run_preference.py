"""Run the isolated preference extension: python -m tools.run_preference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvercell.preference_config import load_preference_config, debug_preference_config
from solvercell.preference_trainer import train_preference
from solvercell.storage import atomic_json, append_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-updates', type=int)
    args = p.parse_args()
    cfg = load_preference_config(args.config)
    if args.debug:
        cfg = debug_preference_config(cfg)
    started, code, error = time.time(), 1, None
    try:
        result = train_preference(cfg, args.output, resume=args.resume,
                                  stop_after_updates=args.stop_after_updates)
        code = 0 if result['status'] == 'complete' else 130
        return code
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        output = Path(args.output)
        identity = json.loads((output / 'worker.json').read_text()) if (output / 'worker.json').is_file() else {}
        start_ticks = Path('/proc/self/stat').read_text().split()[21]
        owns_run = identity.get('pid') == os.getpid() and identity.get('start_ticks') == start_ticks
        # A rejected duplicate/resume must not corrupt an earlier run's exit0.
        if owns_run:
            record = {'returncode': code, 'started_unix': started,
                      'finished_unix': time.time(), 'error': error, 'resume': args.resume}
            if (output / 'source.json').is_file():
                record['source_sha256'] = json.loads((output / 'source.json').read_text())['source_sha256']
            atomic_json(output / 'exit.json', record)
            append_json(output / 'exit_history.jsonl', record)


if __name__ == '__main__':
    raise SystemExit(main())
