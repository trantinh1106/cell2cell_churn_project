"""
utils/logger.py
-----------------
Tiện ích in log song song ra Console và file txt (output/training_log.txt).
Hỗ trợ UTF-8 an toàn trên Windows Console.
"""

import sys
import os

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class Tee:
    """Ghi cùng lúc vào nhiều luồng (console và file log) với xử lý mã hoá ký tự an toàn."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except UnicodeEncodeError:
                safe_data = data.encode("ascii", errors="replace").decode("ascii")
                s.write(safe_data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def start_logging(log_path: str = os.path.join("output", "training_log.txt")):
    """Bắt đầu ghi song song vào console và file log_path."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(sys.stdout, log_file)
    return log_file


def stop_logging(log_file):
    """Kết thúc ghi log, khôi phục luồng stdout chuẩn."""
    sys.stdout = sys.__stdout__
    try:
        log_file.close()
    except Exception:
        pass


def print_header(title: str, width: int = 70, char: str = "="):
    """In tiêu đề dạng khung viền ngăn cách các giai đoạn."""
    line = char * width
    print(f"\n{line}\n{title}\n{line}")
