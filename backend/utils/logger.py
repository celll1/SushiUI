import sys
from io import StringIO
from collections import deque
from datetime import datetime
import threading

class LogCapture:
    """Capture stdout and stderr for console viewing"""

    def __init__(self, max_lines=1000):
        self.max_lines = max_lines
        self.logs = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, message):
        if message.strip():  # Only log non-empty messages
            with self.lock:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.logs.append(f"[{timestamp}] {message.strip()}")

        # Also write to original stdout
        self.original_stdout.write(message)
        self.original_stdout.flush()

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        """Check if the original stdout is a TTY"""
        return self.original_stdout.isatty() if hasattr(self.original_stdout, 'isatty') else False

    def fileno(self):
        """Return the file descriptor of the original stdout"""
        return self.original_stdout.fileno() if hasattr(self.original_stdout, 'fileno') else -1

    def get_logs(self, last_n=None):
        with self.lock:
            if last_n:
                return list(self.logs)[-last_n:]
            return list(self.logs)

    def clear(self):
        with self.lock:
            self.logs.clear()

# Global log capture instance
log_capture = LogCapture()

def setup_logging():
    """Redirect stdout to log capture"""
    sys.stdout = log_capture
    # Note: We're not redirecting stderr to avoid interfering with uvicorn logging
