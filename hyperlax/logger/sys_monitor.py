import csv
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import pandas as pd
import psutil
from nvitop import Device

logger = logging.getLogger(__name__)


class SystemMonitor:
    def __init__(
        self,
        log_interval: float = 0.5,
        log_dir: str = "./system_logs",
        max_in_memory_samples: int = 10000,
    ):
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.max_in_memory_samples = max_in_memory_samples

        self.is_running = False
        self.system_logs: list[dict[str, float]] = []
        self.thread: Thread | None = None
        self.lock = Lock()

        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

        self.gpu_metric_names = [
            "utilization",
            "memory_utilization",
            "memory_used_gb",
            "temperature",
            "power_usage",
        ]
        self.system_metric_names = [
            "cpu_percent",
            "memory_percent",
            "disk_usage_percent",
        ]
        self.process_metric_names = [
            "cpu_percent",
            "memory_percent",
            "memory_used_mb",
            "num_threads",
            "num_children",
        ]
        self.header: list[str] | None = None
        self.devices: list[Device] = []

        self.process = psutil.Process()
        self.last_process_cpu_times = self.process.cpu_times()
        self.last_process_cpu_time = time.time()

    def start(self):
        if self.is_running:
            self.log_str("System monitor is already running.", level="warning")
            return
        try:
            self.devices = Device.all()  # We'll handle if this is empty later
            if not self.devices:  # Allow running without GPU for CPU-only monitoring
                self.log_str(
                    "No GPU devices found. GPU monitoring will be skipped.",
                    level="warning",
                )

            self._setup_csv_logging()  # CSV logging setup first

            self.is_running = True
            self.thread = Thread(target=self._monitor_system, daemon=True)
            self.thread.start()

            startup_msg = f"System monitor started. Monitoring process PID: {self.process.pid}"
            if self.devices:
                startup_msg += f"\nGPU Driver Version: {Device.driver_version()}"
            else:
                startup_msg += "\nNo GPU devices found by nvitop."
            self.log_str(startup_msg, level="debug")

        except Exception as e:
            error_msg = f"Failed to start system monitor: {str(e)}"
            self.log_str(error_msg, level="error")
            self.is_running = False

    def _setup_csv_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        existing_files = [
            f
            for f in os.listdir(self.log_dir)
            if f.startswith("system_log_run_") and f.endswith(".csv")
        ]
        run_numbers = [
            int(re.search(r"system_log_run_(\d+)\.csv", f).group(1))
            for f in existing_files
            if re.search(r"system_log_run_(\d+)\.csv", f)
        ]
        next_run_number = max(run_numbers, default=0) + 1
        self.csv_path = os.path.join(self.log_dir, f"system_log_run_{next_run_number:05d}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.header = ["timestamp"]
        if self.devices:
            for i, _ in enumerate(self.devices):
                for metric in self.gpu_metric_names:
                    self.header.append(f"gpu_{i}_{metric}")
        for metric in self.system_metric_names:
            self.header.append(f"system_{metric}")
        for metric in self.process_metric_names:
            self.header.append(f"process_{metric}")
        self.csv_writer.writerow(self.header)
        self.log_str(f"Saving system monitor CSV at: {Path(self.csv_path).resolve()}")

    def _monitor_system(self):
        while self.is_running:
            try:
                log_entry = {"timestamp": time.time()}
                if self.devices:
                    for i, device in enumerate(self.devices):
                        try:
                            log_entry[f"gpu_{i}_utilization"] = device.gpu_utilization()
                            log_entry[f"gpu_{i}_memory_utilization"] = device.memory_utilization()
                            log_entry[f"gpu_{i}_memory_used_gb"] = device.memory_used() / (1024**3)
                            log_entry[f"gpu_{i}_temperature"] = device.temperature()
                            log_entry[f"gpu_{i}_power_usage"] = device.power_usage()
                        except Exception as e:  # Handle if a device becomes unresponsive
                            self.log_str(f"Error reading GPU {i} metrics: {e}", level="warning")
                            for metric in self.gpu_metric_names:
                                log_entry[f"gpu_{i}_{metric}"] = None
                log_entry["system_cpu_percent"] = psutil.cpu_percent()
                log_entry["system_memory_percent"] = psutil.virtual_memory().percent
                log_entry["system_disk_usage_percent"] = psutil.disk_usage("/").percent

                try:
                    process = psutil.Process(self.process.pid)
                    current_time = time.time()
                    current_cpu_times = process.cpu_times()
                    cpu_time_delta = (
                        current_cpu_times.user - self.last_process_cpu_times.user
                    ) + (current_cpu_times.system - self.last_process_cpu_times.system)
                    wall_time_delta = current_time - self.last_process_cpu_time
                    log_entry["process_cpu_percent"] = (
                        (cpu_time_delta / wall_time_delta) * 100 if wall_time_delta > 0 else 0.0
                    )
                    self.last_process_cpu_times = current_cpu_times
                    self.last_process_cpu_time = current_time
                    log_entry["process_memory_percent"] = process.memory_percent()
                    log_entry["process_memory_used_mb"] = process.memory_info().rss / (1024**2)
                    log_entry["process_num_threads"] = process.num_threads()
                    log_entry["process_num_children"] = len(process.children(recursive=False))
                except psutil.NoSuchProcess:
                    # self.log_str(f"Process with PID {self.process.pid} no longer exists. Process monitoring stopped.", level="warning")
                    for metric in self.process_metric_names:
                        log_entry[f"process_{metric}"] = None
                except Exception as e:
                    self.log_str(
                        f"Error monitoring process {self.process.pid}: {e}",
                        level="warning",
                    )
                    for metric in self.process_metric_names:
                        log_entry[f"process_{metric}"] = None

                with self.lock:
                    self.system_logs.append(log_entry)
                    if len(self.system_logs) > self.max_in_memory_samples:
                        self.system_logs.pop(0)
                self._write_to_csv(log_entry)
            except Exception as e:
                self.log_str(f"Error in system monitoring loop: {str(e)}", level="error")
            finally:
                time.sleep(self.log_interval)

    def _write_to_csv(self, log_entry: dict[str, float]):
        if self.csv_writer and self.header:  # Ensure header is initialized
            try:
                row = [
                    log_entry.get(key, None) for key in self.header
                ]  # Use None for missing values
                self.csv_writer.writerow(row)
                self.csv_file.flush()
            except Exception as e:
                self.log_str(f"Failed to write to CSV: {str(e)}", level="error")

    def get_logs(
        self, metrics: str | list[str] | None = None, include_timestamp: bool = False
    ) -> list[dict[str, float]]:
        with self.lock:
            if metrics is None:
                # Create a comprehensive list of all possible metric keys
                all_possible_keys = set(["timestamp"])
                if self.devices:
                    for i in range(len(self.devices)):
                        for gm in self.gpu_metric_names:
                            all_possible_keys.add(f"gpu_{i}_{gm}")
                for sm in self.system_metric_names:
                    all_possible_keys.add(f"system_{sm}")
                for pm in self.process_metric_names:
                    all_possible_keys.add(f"process_{pm}")

                # If metrics is None, we want to return all keys present in the logs
                # For simplicity here, let's stick to the previous behavior of needing explicit keys
                # or a predefined set, but ideally it would discover from logs.
                # This part needs careful thought if 'metrics=None' means "all ever logged".
                # For now, if metrics is None, let's use all *defined* metric names
                metrics_to_fetch = list(all_possible_keys)
                if not include_timestamp and "timestamp" in metrics_to_fetch:
                    metrics_to_fetch.remove("timestamp")

            elif isinstance(metrics, str):
                metrics_to_fetch = [metrics]
                if include_timestamp and "timestamp" not in metrics_to_fetch:
                    metrics_to_fetch.append("timestamp")
            else:  # isinstance(metrics, list):
                metrics_to_fetch = list(metrics)  # copy
                if include_timestamp and "timestamp" not in metrics_to_fetch:
                    metrics_to_fetch.append("timestamp")

            # Validation is tricky if metrics=None implies all possible keys
            # For now, let's assume metrics_to_fetch contains specific keys the user wants
            # This validation logic might need adjustment based on how 'metrics=None' is handled above.
            # valid_base_metrics = set(self.gpu_metric_names + self.system_metric_names + self.process_metric_names)
            # if include_timestamp:
            #     valid_base_metrics.add('timestamp')
            # # This validation assumes `metrics` contains base names, not full names like `gpu_0_utilization`
            # # The current fetching logic below expects full names if specific GPU/system/process keys are desired
            # # This part of get_logs needs a clearer contract.
            # # For now, assuming metrics_to_fetch contains the exact keys as they appear in log_entry.

            filtered_logs = []
            for log_entry in self.system_logs:
                filtered_log_entry = {
                    key: log_entry[key] for key in metrics_to_fetch if key in log_entry
                }
                if filtered_log_entry:  # Add if we found any of the requested metrics
                    filtered_logs.append(filtered_log_entry)
            return filtered_logs

    def get_available_metrics(self) -> dict[str, list[str]]:
        # This should ideally list the full keys available, e.g., 'gpu_0_utilization'
        available = {
            "gpu": [],
            "system": list(self.system_metric_names),
            "process": list(self.process_metric_names),
        }
        if self.devices:
            for i in range(len(self.devices)):
                for metric_base in self.gpu_metric_names:
                    available["gpu"].append(f"gpu_{i}_{metric_base}")
        return available

    def get_final_statistics(self) -> dict[str, dict[str, float]]:
        if (
            not self.csv_path
            or not os.path.exists(self.csv_path)
            or os.path.getsize(self.csv_path) == 0
        ):
            self.log_str(
                "CSV file does not exist or is empty. No statistics available.",
                level="warning",
            )
            return {}
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                self.log_str("CSV file is empty. No statistics available.", level="warning")
                return {}

            num_gpus = len(self.devices) if self.devices else 0
            final_stats: dict[str, dict[str, float]] = {
                "gpu": {},
                "system": {},
                "process": {},
            }

            for i in range(num_gpus):
                for metric in self.gpu_metric_names:
                    col_name = f"gpu_{i}_{metric}"
                    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                        final_stats["gpu"][f"{col_name}_avg"] = df[col_name].mean()
                        final_stats["gpu"][f"{col_name}_max"] = df[col_name].max()
                    else:
                        final_stats["gpu"][f"{col_name}_avg"] = np.nan  # Or 0.0 or None
                        final_stats["gpu"][f"{col_name}_max"] = np.nan

            for metric in self.system_metric_names:
                col_name = f"system_{metric}"
                if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                    final_stats["system"][f"{col_name}_avg"] = df[col_name].mean()
                    final_stats["system"][f"{col_name}_max"] = df[col_name].max()
                else:
                    final_stats["system"][f"{col_name}_avg"] = np.nan
                    final_stats["system"][f"{col_name}_max"] = np.nan

            for metric in self.process_metric_names:
                col_name = f"process_{metric}"
                if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                    final_stats["process"][f"{col_name}_avg"] = df[col_name].mean()
                    final_stats["process"][f"{col_name}_max"] = df[col_name].max()
                else:
                    final_stats["process"][f"{col_name}_avg"] = np.nan
                    final_stats["process"][f"{col_name}_max"] = np.nan
            return final_stats
        except pd.errors.EmptyDataError:
            self.log_str(
                "CSV file is empty (pandas error). No statistics available.",
                level="warning",
            )
            return {}
        except Exception as e:
            self.log_str(f"Error calculating final statistics: {str(e)}", level="error")
            return {}

    def get_current_utilization(
        self,
    ) -> dict[str, dict[str, float | None]]:  # Ensure Optional for values
        current_stats: dict[str, dict[str, float | None]] = {
            "gpu": {},
            "system": {},
            "process": {},
        }
        if self.devices:
            for i, device in enumerate(self.devices):
                try:
                    current_stats["gpu"][f"gpu_{i}_utilization"] = device.gpu_utilization()
                    current_stats["gpu"][f"gpu_{i}_memory_utilization"] = (
                        device.memory_utilization()
                    )
                    mem_used_gb = device.memory_used() / (1024**3)
                    mem_total_gb = device.memory_total() / (1024**3)
                    current_stats["gpu"][f"gpu_{i}_memory_used_gb"] = mem_used_gb
                    current_stats["gpu"][f"gpu_{i}_total_memory_gb"] = mem_total_gb
                except Exception as e:
                    self.log_str(f"Error getting current stats for GPU {i}: {e}", level="warning")
                    # Set relevant fields to None or 0.0
                    current_stats["gpu"][f"gpu_{i}_utilization"] = None
                    current_stats["gpu"][f"gpu_{i}_memory_utilization"] = None
                    current_stats["gpu"][f"gpu_{i}_memory_used_gb"] = None
                    current_stats["gpu"][f"gpu_{i}_total_memory_gb"] = None
        else:  # No devices, fill with None for expected keys
            current_stats["gpu"] = {  # Assuming one GPU would have been gpu_0
                "gpu_0_utilization": None,
                "gpu_0_memory_utilization": None,
                "gpu_0_memory_used_gb": None,
                "gpu_0_total_memory_gb": None,
            }

        current_stats["system"]["cpu_utilization"] = psutil.cpu_percent()
        sys_mem = psutil.virtual_memory()
        current_stats["system"]["memory_utilization"] = sys_mem.percent
        current_stats["system"]["system_memory_used_gb"] = sys_mem.used / (1024**3)
        current_stats["system"]["system_total_memory_gb"] = sys_mem.total / (1024**3)

        try:
            process = psutil.Process(self.process.pid)
            # For current process CPU, psutil.Process.cpu_percent() needs to be called with an interval
            # or it returns 0 on first call. The _monitor_system loop calculates it over time.
            # Here, we can report the last calculated one or fetch a new one (might block briefly).
            # For simplicity, let's report None or rely on the logged process_cpu_percent if available in latest log.
            current_stats["process"]["cpu_percent"] = (
                None  # Or retrieve from last log if needed for "current"
            )
            current_stats["process"]["memory_percent"] = process.memory_percent()
        except psutil.NoSuchProcess:
            # self.log_str(f"Process with PID {self.process.pid} no longer exists for current_util.", level="warning")
            current_stats["process"]["cpu_percent"] = None
            current_stats["process"]["memory_percent"] = None
        return current_stats

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=self.log_interval * 2 + 1)  # Wait for thread
            if self.thread.is_alive():
                self.log_str("System monitor thread did not join in time.", level="warning")
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        self.log_str("System monitor stopped.")

    def log_str(self, message: str, level: str = "info"):
        log_fn = getattr(logger, level, logger.info)
        log_fn(message)

    def get_utilization_stats_for_interval(
        self,
        t_start: float,
        t_end: float,
        gpu_id: int = 0,
        utilization_threshold_pct: float = 90.0,
    ) -> dict[str, float | None]:
        if not self.is_running:
            self.log_str(
                "System monitor is not running. Cannot get interval stats.",
                level="warning",
            )
            return {"error": "Monitor not running"}

        with self.lock:
            if not self.system_logs:
                return {"error": "No logs available"}
            # min_log_ts = min(log.get('timestamp', float('inf')) for log in self.system_logs)
            # max_log_ts = max(log.get('timestamp', float('-inf')) for log in self.system_logs)
            # self.log_str(f"DEBUG SysMon.get_utilization_stats_for_interval: Interval [{t_start:.3f}, {t_end:.3f}]. Available log range: [{min_log_ts:.3f}, {max_log_ts:.3f}] with {len(self.system_logs)} logs.", level="debug")
            relevant_logs = [
                log for log in self.system_logs if t_start <= log.get("timestamp", 0) <= t_end
            ]

        stats: dict[str, float | None] = {
            # GPU Util
            f"gpu_{gpu_id}_util_mean": None,
            f"gpu_{gpu_id}_util_max": None,
            f"gpu_{gpu_id}_util_min": None,
            f"gpu_{gpu_id}_util_median": None,
            f"gpu_{gpu_id}_util_std": None,
            f"gpu_{gpu_id}_util_time_above_{utilization_threshold_pct:.0f}pct": None,
            # GPU Memory
            f"gpu_{gpu_id}_mem_util_mean": None,
            f"gpu_{gpu_id}_mem_used_gb_mean": None,
            f"gpu_{gpu_id}_total_memory_gb": None,
            # System CPU
            "system_cpu_util_mean": None,
            "system_cpu_util_max": None,
            "system_cpu_util_min": None,
            "system_cpu_util_median": None,
            "system_cpu_util_std": None,
            f"system_cpu_util_time_above_{utilization_threshold_pct:.0f}pct": None,
            # System Memory
            "system_mem_util_mean": None,
            "system_mem_used_gb_mean": None,
            "system_total_memory_gb": None,
            # Info
            "num_samples_in_interval": 0,
            "interval_duration_s": t_end - t_start,
            "actual_log_span_s": 0.0,
        }

        if not relevant_logs:
            self.log_str(
                f"No system monitor samples found for interval [{t_start:.3f}, {t_end:.3f}] ({stats['interval_duration_s']:.3f}s duration)",
                level="debug",
            )
            return stats

        stats["num_samples_in_interval"] = len(relevant_logs)
        log_timestamps = [log["timestamp"] for log in relevant_logs]
        if log_timestamps:
            stats["actual_log_span_s"] = max(log_timestamps) - min(log_timestamps)

        # GPU Metrics
        if self.devices and 0 <= gpu_id < len(self.devices):
            gpu_util_key = f"gpu_{gpu_id}_utilization"
            gpu_utils = [
                log.get(gpu_util_key) for log in relevant_logs if log.get(gpu_util_key) is not None
            ]
            if gpu_utils:
                stats[f"gpu_{gpu_id}_util_mean"] = np.mean(gpu_utils)
                stats[f"gpu_{gpu_id}_util_max"] = np.max(gpu_utils)
                stats[f"gpu_{gpu_id}_util_min"] = np.min(gpu_utils)
                stats[f"gpu_{gpu_id}_util_median"] = np.median(gpu_utils)
                stats[f"gpu_{gpu_id}_util_std"] = np.std(gpu_utils)
                above_thresh_count = sum(
                    1 for util in gpu_utils if util >= utilization_threshold_pct
                )
                stats[f"gpu_{gpu_id}_util_time_above_{utilization_threshold_pct:.0f}pct"] = (
                    above_thresh_count / len(gpu_utils)
                ) * 100.0

            gpu_mem_util_key = f"gpu_{gpu_id}_memory_utilization"
            gpu_mem_utils = [
                log.get(gpu_mem_util_key)
                for log in relevant_logs
                if log.get(gpu_mem_util_key) is not None
            ]
            if gpu_mem_utils:
                stats[f"gpu_{gpu_id}_mem_util_mean"] = np.mean(gpu_mem_utils)

            gpu_mem_used_key = f"gpu_{gpu_id}_memory_used_gb"
            gpu_mem_used_gbs = [
                log.get(gpu_mem_used_key)
                for log in relevant_logs
                if log.get(gpu_mem_used_key) is not None
            ]
            if gpu_mem_used_gbs:
                stats[f"gpu_{gpu_id}_mem_used_gb_mean"] = np.mean(gpu_mem_used_gbs)

            try:
                stats[f"gpu_{gpu_id}_total_memory_gb"] = self.devices[gpu_id].memory_total() / (
                    1024**3
                )
            except Exception:
                pass  # Already None

        # System CPU Utilization
        cpu_util_key = "system_cpu_percent"
        cpu_utils = [
            log.get(cpu_util_key) for log in relevant_logs if log.get(cpu_util_key) is not None
        ]
        if cpu_utils:
            stats["system_cpu_util_mean"] = np.mean(cpu_utils)
            stats["system_cpu_util_max"] = np.max(cpu_utils)
            stats["system_cpu_util_min"] = np.min(cpu_utils)
            stats["system_cpu_util_median"] = np.median(cpu_utils)
            stats["system_cpu_util_std"] = np.std(cpu_utils)
            cpu_above_thresh_count = sum(
                1 for util in cpu_utils if util >= utilization_threshold_pct
            )
            stats[f"system_cpu_util_time_above_{utilization_threshold_pct:.0f}pct"] = (
                cpu_above_thresh_count / len(cpu_utils)
            ) * 100.0

        # System Memory Utilization
        sys_mem_util_key = "system_memory_percent"
        sys_mem_utils = [
            log.get(sys_mem_util_key)
            for log in relevant_logs
            if log.get(sys_mem_util_key) is not None
        ]
        if sys_mem_utils:
            stats["system_mem_util_mean"] = np.mean(sys_mem_utils)
        try:
            vm = psutil.virtual_memory()
            stats["system_total_memory_gb"] = vm.total / (1024**3)
            # Calculate mean used GB from mean percent if possible
            if stats["system_mem_util_mean"] is not None:
                stats["system_mem_used_gb_mean"] = (stats["system_mem_util_mean"] / 100.0) * stats[
                    "system_total_memory_gb"
                ]
        except Exception:
            pass  # Already None

        return stats
