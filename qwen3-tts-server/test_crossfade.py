"""
DEPRECATED: This test has been replaced by test_streaming.py

The crossfade approach has been replaced with progressive batch decoding,
which produces artifact-free audio without needing crossfade.

Run test_streaming.py instead.
"""

import sys
print(__doc__)
print("Running test_streaming.py instead...\n")

# Import and run the new test
from test_streaming import main
main()
