from dataclasses import dataclass


@dataclass(frozen=True)
class LoggerConfig:
    # Core Console Logger Settings
    enabled: bool = True  # Master switch for all logging from this config
    initialized: bool = False

    level: str = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
    show_timestamp: bool = True  # Whether ConsoleLogger prepends timestamps

    # File Logging for Console Output
    save_console_to_file: bool = False  # New: Master switch for saving console output to file
    console_log_filename: str = "hyperlax.log"  # New: Default filename for the console log

    # Experiment Output & Progress
    base_exp_path: str = "results"  # Root for experiment artifacts
    aggregate_metrics: bool = True  # For batched runs, affects how metrics are saved/processed
    enable_hyperparam_progress_bar: bool = False  # For the Phaser progress display

    # Debugging & Performance Monitoring Flags
    # These flags control whether specific diagnostic logs are generated or decorators are active.
    # The actual blocking for timing is handled by the timing/memory logging functions themselves.
    enable_jax_debug_prints: bool = False  # Controls JAXDebugFilter for jax.debug.print
    enable_timing_logs: bool = False  # Controls if timing decorators/phaser log execution times
    enable_gpu_memory_logging: bool = False  # Controls if GPU memory decorators log memory usage

    # Checkpointing (Simplified for now, can be expanded)
    checkpointing_enabled: bool = False
    # If more detailed checkpointing is needed:
    # from hyperlax.configs.logger.base_logger import Checkpointing # Make sure Checkpointing is defined
    # checkpointing: Checkpointing = field(default_factory=Checkpointing)

    enable_summarize_layout: bool = False
