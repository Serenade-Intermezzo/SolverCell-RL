"""Rejected entrypoint invocations preserve completed experiment evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from solvercell.preference_config import PreferenceConfig
from tools.run_preference import main


class PreferenceEntrypointTest(unittest.TestCase):
    def test_rejected_duplicate_and_completed_resume_preserve_exit(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = PreferenceConfig().to_dict()
            path, output = root / 'requested.json', root / 'complete'
            path.write_text(json.dumps(config))
            output.mkdir()
            (output / 'config.json').write_text(json.dumps(config))
            (output / 'metrics.json').write_text('{"status":"complete"}')
            (output / 'worker.json').write_text('{"pid":-1,"start_ticks":"old"}')
            evidence = '{"returncode":0,"source_sha256":"saved_source"}\n'
            (output / 'exit.json').write_text(evidence)
            for flags, error in (([], FileExistsError), (['--resume'], ValueError)):
                with self.subTest(flags=flags), patch('sys.argv', ['run_preference', '--config', str(path), '--output', str(output), *flags]):
                    with self.assertRaises(error):
                        main()
                    self.assertEqual((output / 'exit.json').read_text(), evidence)
                    self.assertFalse((output / 'exit_history.jsonl').exists())
