import os
import time
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")


def benchmark_and_log(step_name: str):
    """Parameterized decorator to measure execution time and log audit trails."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            os.makedirs(LOG_DIR, exist_ok=True)
            start_time = time.perf_counter()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "SUCCESS"
            error_msg = ""

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                status = "FAILED"
                error_msg = f" | Details: {str(e)}"
                raise
            else:
                return result
            finally:
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                log_entry = (
                    f"[{timestamp}] | Step: {step_name:<22} | "
                    f"Status: {status:<7} | Time: {elapsed_ms:>8.3f} ms{error_msg}\n"
                )
                with open(LOG_FILE, mode="a", encoding="utf-8") as f:
                    f.write(log_entry)
                print(f"⚡ {step_name:<22} -> {elapsed_ms:>8.3f} ms [{status}]")

        return wrapper
    return decorator