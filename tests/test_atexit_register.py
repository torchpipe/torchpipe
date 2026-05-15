# test/test_atexit_integration.py

import subprocess
import sys
import os
import tempfile


def test_atexit_no_crash():
    # Construct a standalone Python script content
    script = '''
import omniback
omniback.ffi.register_backend_group("test_backend", "test_group", lambda: None)
print("Registered, exiting normally...")
'''

    # Write to temp file (avoid long command line)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Launch subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=10  # Prevent hanging
        )

        # Check normal exit (return code 0)
        assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Optional: check stderr for exceptions/traceback
        assert "Traceback" not in result.stderr, f"Unexpected exception:\n{result.stderr}"

        print("atexit cleanup ran without crash")

    finally:
        os.unlink(script_path)  # Clean up temp file
