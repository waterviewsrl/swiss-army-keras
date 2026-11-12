"""Print the test files of one chunk of the suite.

The fast suite takes about 16 minutes on one machine; continuous
integration runs it as four parallel jobs, and the same chunks keep a local
run under ten minutes each:

    pytest -m "not slow" $(python scripts/ci_test_chunk.py backbones)

`rest` is every test file that no other chunk names, so a new test file is
run without having to be listed here.
"""

import glob
import os
import sys

CHUNKS = {
    'backbones': ['test_backbone_*.py'],
    'heads': ['test_classifiers.py', 'test_decoders.py', 'test_multi_task.py', 'test_new_heads.py'],
    'detection': ['test_detection.py', 'test_detection_utils.py', 'test_layer_utils.py', 'test_losses.py',
                  'test_training_strategies.py'],
}


def files(chunk, tests_dir):
    named = {name: sorted(path for pattern in patterns for path in glob.glob(os.path.join(tests_dir, pattern)))
             for name, patterns in CHUNKS.items()}
    if chunk in named:
        return named[chunk]
    if chunk == 'rest':
        taken = {path for paths in named.values() for path in paths}
        return sorted(path for path in glob.glob(os.path.join(tests_dir, 'test_*.py')) if path not in taken)
    raise SystemExit('Unknown chunk {!r}; choose from {}.'.format(chunk, ', '.join(list(CHUNKS) + ['rest'])))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
    print(' '.join(os.path.relpath(path) for path in files(sys.argv[1], tests)))
