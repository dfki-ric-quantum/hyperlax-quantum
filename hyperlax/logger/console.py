import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path

import colorama
from colorama import Back, Fore, Style

from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.logger.jax_debug import JAXDebugFilter


# --- ENUMS ---
class LogFormatMode(Enum):
    COMPACT = "compact"
    STANDARD = "standard"
    DETAILED = "detailed"
    CUSTOM = "custom"


class FilePathDisplay(Enum):
    NONE = "none"  # No file path shown
    ABSOLUTE = "absolute"  # /abs/path/to/file.py:123
    RELATIVE = "relative"  # relative/path/to/file.py:123
    FILENAME = "filename"  # file.py:123


# --- LEVEL TO COLOR MAP ---
LOG_LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.WHITE + Back.RED,
}


# --- FORMATTER ---
class ColorConsoleFormatter(logging.Formatter):
    def __init__(
        self,
        log_format_mode: LogFormatMode = LogFormatMode.STANDARD,
        show_header: bool = True,
        show_timestamp: bool = True,
        show_logger_name: bool = True,
        show_line_number: bool = True,
        show_level_text: bool = True,
        logger_name_width: int = 28,
        overall_log_prefix: str | None = None,
        compact_timestamp: bool = True,
        custom_format_str: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        strip_colors_for_file: bool = False,
        file_path_display: FilePathDisplay = FilePathDisplay.NONE,
    ):
        self.log_format_mode = log_format_mode
        self.show_timestamp = show_timestamp
        self.show_logger_name = show_logger_name
        self.show_line_number = show_line_number
        self.show_level_text = show_level_text
        self.logger_name_width = logger_name_width
        self.overall_log_prefix = f"[{overall_log_prefix.strip()}]" if overall_log_prefix else ""
        self.compact_timestamp = compact_timestamp
        self.custom_format_str = custom_format_str
        self._strip_colors_for_file = strip_colors_for_file
        self.file_path_display = file_path_display
        self.show_header = show_header

        # --- FORMAT STRING BUILDER ---
        # if log_format_mode == LogFormatMode.CUSTOM and self.custom_format_str:
        #     fmt_str = self.custom_format_str
        #     effective_datefmt = datefmt
        # else:
        if not self.show_header:
            fmt_str = "%(message)s"
            effective_datefmt = None
        elif log_format_mode == LogFormatMode.CUSTOM and self.custom_format_str:
            fmt_str = self.custom_format_str
            effective_datefmt = datefmt
        else:
            fmt_parts = []
            # --- File path header ---
            if self.file_path_display != FilePathDisplay.NONE:
                fmt_parts.append("%(logpath)s")

            # --- Timestamp ---
            if self.show_timestamp:
                effective_datefmt = (
                    "[%H:%M:%S]" if self.compact_timestamp else "[%Y-%m-%d %H:%M:%S,%f]"[:-3]
                )
                fmt_parts.append("%(asctime)s")
            else:
                effective_datefmt = None

            # --- Overall prefix ---
            if self.overall_log_prefix:
                fmt_parts.append(self.overall_log_prefix)

            # --- Level text ---
            if self.show_level_text:
                fmt_parts.append("[%(levelname)-8s]")

            # --- Logger name ---
            if self.show_logger_name:
                if self.log_format_mode == LogFormatMode.COMPACT:
                    fmt_parts.append("[%(short_name)s]")
                else:
                    fmt_parts.append(
                        f"[%(name)-{self.logger_name_width}.{self.logger_name_width}s]"
                    )

            # --- Line number, except for compact ---
            if self.show_line_number and self.log_format_mode != LogFormatMode.COMPACT:
                fmt_parts.append("L%(lineno)-4d:")

            # --- The message ---
            fmt_parts.append("%(message)s")

            fmt_str = " ".join(fmt_parts)

        super().__init__(fmt=fmt_str, datefmt=effective_datefmt, style=style)

    def format(self, record: logging.LogRecord) -> str:
        # For %(short_name)s
        record.short_name = record.name.rsplit(".", 1)[-1]

        # --- Set %(logpath)s for editor-clickable links ---
        if self.file_path_display == FilePathDisplay.ABSOLUTE:
            record.logpath = f"{record.pathname}:{record.lineno}"
        elif self.file_path_display == FilePathDisplay.RELATIVE:
            try:
                relpath = os.path.relpath(record.pathname)
            except Exception:
                relpath = record.pathname
            record.logpath = f"{relpath}:{record.lineno}"
        elif self.file_path_display == FilePathDisplay.FILENAME:
            record.logpath = f"{record.filename}:{record.lineno}"
        else:
            record.logpath = ""

        # Let base formatter do the work
        log_message_uncolored = super().format(record)

        if self._strip_colors_for_file:
            ansi_escape_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape_pattern.sub("", log_message_uncolored)
        else:
            log_color = LOG_LEVEL_COLORS.get(record.levelno, "")
            return f"{log_color}{log_message_uncolored}{Style.RESET_ALL}"


# --- MAIN CONFIGURE FUNCTION ---
def configure_global_logging(
    config: LoggerConfig,
    root_logger_name: str = "hyperlax",
    overall_log_prefix: str | None = None,
    show_header: bool = False,
    show_colored_levels: bool = False,
    console_format_mode: LogFormatMode = LogFormatMode.COMPACT,
    console_show_level_text: bool = False,
    console_logger_name_width: int = 28,
    file_format_mode: LogFormatMode = LogFormatMode.DETAILED,
    file_logger_name_width: int = 35,
    console_file_path_display: FilePathDisplay = FilePathDisplay.NONE,  # .NONE,
    file_file_path_display: FilePathDisplay = FilePathDisplay.NONE,  # .NONE,
    show_logger_name: bool = False,
) -> None:
    """
    Configure the global Python logging system for hyperlax application.
    Supports user choice of file path display in log headers for better editor integration.
    """

    if show_colored_levels:
        colorama.init(autoreset=True, strip=False)

    # 1. JAX Debug Filter Setup
    jax_filter = JAXDebugFilter.get_instance()
    if not config.enabled or not config.enable_jax_debug_prints:
        jax_filter.deactivate_to_noop()
    elif config.enabled and config.enable_jax_debug_prints:
        jax_filter.activate_filtering()

    app_root_logger = logging.getLogger(root_logger_name)

    if not config.enabled:
        app_root_logger.handlers.clear()
        app_root_logger.addHandler(logging.NullHandler())
        app_root_logger.propagate = False
        app_root_logger.setLevel(logging.CRITICAL + 1)
        return

    log_level_int = getattr(logging, config.level.upper(), logging.INFO)
    app_root_logger.setLevel(log_level_int)
    app_root_logger.handlers.clear()
    app_root_logger.propagate = False

    # 2. Console Handler Setup
    console_formatter = ColorConsoleFormatter(
        log_format_mode=console_format_mode,
        # show_timestamp=config.show_timestamp,
        show_timestamp=True,
        show_logger_name=False,  # show_logger_name,
        show_line_number=(console_format_mode != LogFormatMode.COMPACT),
        show_level_text=console_show_level_text,
        logger_name_width=console_logger_name_width,
        overall_log_prefix=overall_log_prefix,
        compact_timestamp=(
            console_format_mode == LogFormatMode.COMPACT
            or console_format_mode == LogFormatMode.STANDARD
        ),
        strip_colors_for_file=False,
        file_path_display=console_file_path_display,
        show_header=show_header,
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level_int)
    app_root_logger.addHandler(console_handler)

    # 3. File Handler Setup (if enabled)
    if config.save_console_to_file:
        if config.base_exp_path:
            log_file_dir = Path(config.base_exp_path)
            try:
                log_file_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = log_file_dir / config.console_log_filename

                file_formatter = ColorConsoleFormatter(
                    log_format_mode=file_format_mode,
                    show_timestamp=True,
                    show_logger_name=True,
                    show_line_number=True,
                    show_level_text=True,
                    logger_name_width=file_logger_name_width,
                    overall_log_prefix=overall_log_prefix,
                    compact_timestamp=False,
                    strip_colors_for_file=True,
                    file_path_display=file_file_path_display,
                )
                file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(log_level_int)
                app_root_logger.addHandler(file_handler)

                app_root_logger.info(
                    f"Console output is also being logged to: {log_file_path.resolve()}"
                )
            except Exception as e:
                app_root_logger.error(
                    f"Failed to configure file logging to '{config.base_exp_path}/{config.console_log_filename}': {e}. "
                    "File logging for console output will be disabled for this session."
                )
        else:
            app_root_logger.warning(
                "File logging for console output (`save_console_to_file=True`) requested, "
                "but 'base_exp_path' is not set in LoggerConfig. Skipping file logging for console output."
            )

    app_root_logger.info(
        f"hyperlax global logger '{root_logger_name}' configured. Level: {logging.getLevelName(app_root_logger.level)}."
    )


# Example usage:
# configure_global_logging(
#     config=my_logger_config,
#     console_file_path_display=FilePathDisplay.RELATIVE,  # Or ABSOLUTE, FILENAME, NONE
#     file_file_path_display=FilePathDisplay.ABSOLUTE,
# )
