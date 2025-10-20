import logging
import sys
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from hyperlax.base_types import HPRuntimeState
from hyperlax.logger.sys_monitor import SystemMonitor

logger = logging.getLogger(__name__)


def _generate_progress_bar(
    percentage,  # current progress percentage for this HP
    length=20,
    char="=",
    tip_char_normal=">",
    tip_char_overshoot="*",  # For when current_steps > current_target_milestone_value
    unfill_char=".",
    milestone_char_future="|",
    milestone_char_passed="+",
    milestone_char_current_target_overshot="!",  # If current_steps > current_target_milestone_value
    all_milestone_bar_indices: list[int] | None = None,  # char indices for ALL milestones
    current_target_milestone_bar_idx: int = -1,  # char idx of the current HP's target milestone
    is_generally_overshooting_current_target: bool = False,  # For the tip character
):
    percentage = max(0, min(100, percentage))
    # How many characters are filled based on the HP's current progress percentage
    filled_progress_length = int(length * percentage / 100)

    bar_list = [unfill_char] * length

    # Fill the bar with the main progress character
    for i in range(filled_progress_length):
        if i < length:
            bar_list[i] = char

    # Determine and place the tip character
    tip_to_use = (
        tip_char_overshoot if is_generally_overshooting_current_target else tip_char_normal
    )
    tip_pos = -1

    if 0 <= filled_progress_length < length:  # Tip is applicable if bar is not full
        bar_list[filled_progress_length] = tip_to_use
        tip_pos = filled_progress_length
    elif filled_progress_length == length and length > 0:  # Bar is full
        if (
            is_generally_overshooting_current_target
        ):  # If full AND overshooting current eval target
            # Overwrite last char with overshoot tip only if it's genuinely overshooting.
            # If it's just full and completed, no special tip needed unless specified.
            bar_list[length - 1] = tip_to_use
            tip_pos = length - 1
        # Otherwise (full but not overshooting its *current* target, or completed),
        # the bar is just full of `char`.

    # Overlay milestone markers
    if all_milestone_bar_indices:
        for m_bar_idx in all_milestone_bar_indices:
            if 0 <= m_bar_idx < length:
                chosen_milestone_char = milestone_char_future  # Default to future

                if m_bar_idx == current_target_milestone_bar_idx:
                    # This is the specific milestone this HP is currently aiming for.
                    # Has the progress (filled_length) passed this milestone's position on the bar?
                    if filled_progress_length > m_bar_idx:
                        chosen_milestone_char = milestone_char_current_target_overshot
                    # else, it's the current target, but progress hasn't reached its bar position yet.
                    # It will remain `milestone_char_future` unless `filled_progress_length == m_bar_idx`,
                    # in which case the tip might overwrite it.
                elif filled_progress_length > m_bar_idx:
                    # Progress has definitely passed this non-target milestone's position.
                    chosen_milestone_char = milestone_char_passed
                # Otherwise, it's a future milestone not yet reached and not the current target.

                # Place the chosen milestone char, ensuring tip takes precedence
                if m_bar_idx == tip_pos:
                    pass  # Tip (normal or general overshoot) takes precedence
                else:
                    bar_list[m_bar_idx] = chosen_milestone_char

    bar_str = "".join(bar_list)
    return f"[{bar_str}]"


def display_hp_progress(
    num_hyperparams,
    current_steps,
    active_status,
    target_total_steps,
    milestone_targets: list[int] | None = None,
    current_milestone_indices: list[int] | None = None,
    segment_info: dict[str, Any] | None = None,
    current_returns: list[float | None] | None = None,
    clear_previous: bool = True,
    learning_indicators: list[str] | None = None,
    resource_info: dict[str, Any] | None = None,
    display_completed_hps_flag: bool = True,
):
    if clear_previous and sys.stdout.isatty():
        # Only try to clear the screen if stdout is a tty (an interactive terminal)
        print("\033[2J\033[H", end="")
        sys.stdout.flush()

    logger.info("\n=== TRAINING PROGRESS ===\n")

    if segment_info:
        seg_num = segment_info.get("segment_num", 0)
        scan_len = segment_info.get("scan_length", 0)
        eta_next_eval = segment_info.get("eta_next_eval", "N/A")
        eta_next_rephase = segment_info.get("eta_next_rephase", "N/A")
        jit_train = segment_info.get("jit_train_s", "N/A")
        jit_eval = segment_info.get("jit_eval_s", "N/A")
        logger.info(
            f"Segment: {seg_num} | Scan: {scan_len} | ETA Eval: {eta_next_eval} | ETA Rephase: {eta_next_rephase} | JIT (T/E): {jit_train}/{jit_eval}"
        )
        logger.info("")  # For newline

    logger.info("--- HYPERRRRRRRRRRPARAMETER ---")
    MIN_PROGRESS_BAR_LENGTH, MAX_PROGRESS_BAR_LENGTH, CHARS_PER_MILESTONE_IDEAL = (
        20,
        80,
        2,
    )
    progress_bar_actual_length = MIN_PROGRESS_BAR_LENGTH
    if milestone_targets and (num_milestones := len(milestone_targets)) > 0:
        progress_bar_actual_length = max(
            MIN_PROGRESS_BAR_LENGTH,
            min(num_milestones * CHARS_PER_MILESTONE_IDEAL, MAX_PROGRESS_BAR_LENGTH),
        )
    else:
        progress_bar_actual_length = 30

    max_hp_idx_digits = len(str(num_hyperparams - 1)) if num_hyperparams > 0 else 1
    formatted_target_steps = f"{target_total_steps:,}"
    max_step_digits_with_commas = len(formatted_target_steps)
    percent_width_for_num, return_text_width = 5, 8
    line_format = (
        f"HP {{hp_idx:<{max_hp_idx_digits}}} : {{bar}} "
        f"{{progress_pct:>{percent_width_for_num}.1f}}% "
        f"{{current_hp_steps:>{max_step_digits_with_commas},}}/"
        f"{{target_total_steps:>{max_step_digits_with_commas},}} "
        f"Return: {{return_text:<{return_text_width}}} {{trend_indicator}}"
    )
    any_hp_displayed = False
    for hp_idx in range(num_hyperparams):
        is_active_hp = active_status[hp_idx]
        if not is_active_hp and not display_completed_hps_flag:
            continue
        any_hp_displayed = True
        hp_current_steps = current_steps[hp_idx]
        progress_pct = (
            (hp_current_steps / target_total_steps) * 100 if target_total_steps > 0 else 0.0
        )
        all_milestones_on_bar: list[int] | None = None
        if milestone_targets and target_total_steps > 0:
            all_milestones_on_bar = sorted(
                list(
                    set(
                        max(
                            0,
                            min(
                                int((m_val / target_total_steps) * progress_bar_actual_length),
                                progress_bar_actual_length - 1,
                            ),
                        )
                        for m_val in milestone_targets
                    )
                )
            )
        is_generally_overshooting_current_target_for_tip = False
        current_target_milestone_char_idx_for_bar = -1
        if (
            milestone_targets
            and current_milestone_indices
            and hp_idx < len(current_milestone_indices)
        ):
            hp_cmi_list_idx = current_milestone_indices[hp_idx]
            if hp_cmi_list_idx < len(milestone_targets):
                current_target_m_val = milestone_targets[hp_cmi_list_idx]
                if hp_current_steps > current_target_m_val:
                    is_generally_overshooting_current_target_for_tip = True
                if target_total_steps > 0:
                    current_target_milestone_char_idx_for_bar = max(
                        0,
                        min(
                            int(
                                (current_target_m_val / target_total_steps)
                                * progress_bar_actual_length
                            ),
                            progress_bar_actual_length - 1,
                        ),
                    )
        bar_visual = _generate_progress_bar(
            progress_pct,
            progress_bar_actual_length,
            all_milestone_bar_indices=all_milestones_on_bar,
            current_target_milestone_bar_idx=current_target_milestone_char_idx_for_bar,
            is_generally_overshooting_current_target=is_generally_overshooting_current_target_for_tip,
        )
        return_text_str = (
            f"{current_returns[hp_idx]:>{return_text_width}.2f}"
            if current_returns
            and hp_idx < len(current_returns)
            and current_returns[hp_idx] is not None
            else f"{'N/A':>{return_text_width}}"
        )
        trend_indicator_char = (
            learning_indicators[hp_idx]
            if learning_indicators
            and hp_idx < len(learning_indicators)
            and learning_indicators[hp_idx] is not None
            else " "
        )
        logger.info(
            line_format.format(
                hp_idx=hp_idx,
                bar=bar_visual,
                progress_pct=progress_pct,
                current_hp_steps=hp_current_steps,
                target_total_steps=target_total_steps,
                return_text=return_text_str.strip(),
                trend_indicator=trend_indicator_char,
            )
        )

    active_count_summary = sum(1 for is_active in active_status if is_active)
    completed_count_summary = num_hyperparams - active_count_summary
    if not any_hp_displayed:
        if completed_count_summary == num_hyperparams and num_hyperparams > 0:
            logger.info("  All hyperparameter configurations have completed training.")
        elif num_hyperparams == 0:
            logger.info("  No hyperparameter configurations to train.")
        else:
            logger.info("  No active hyperparameter configurations to display (unexpected state).")
    logger.info(
        f"\nActive: {active_count_summary}  |  Completed: {completed_count_summary}  |  Total: {num_hyperparams}"
    )

    if resource_info:
        logger.info("\n--- RESOURCE UTILIZZZZZZZZZZATION ---")
        interval_info = resource_info.get("interval_info", {})
        status_msg = interval_info.get("status_msg", "")
        if interval_info.get("num_samples_in_interval") is not None:
            logger.debug(
                f"Interval: {interval_info.get('duration_s', 0.0):.2f}s, Samples: {interval_info.get('num_samples_in_interval')}"
            )
        elif status_msg:
            logger.debug(f"Interval: ({status_msg})")

        def fmt_f(val: float | None, precision: int = 0) -> str:
            # Safely handle None and potential non-float types if they sneak in, though type hints should prevent
            if val is None:
                return "N/A"
            try:
                return f"{float(val):3.{precision}f}"
            except (ValueError, TypeError):
                return "N/A"

        if resource_info.get("gpus"):
            for gpu_data in resource_info.get("gpus", []):
                gpu_id = gpu_data.get("id", 0)

                # GPU Memory
                mem_used_gb_mean = gpu_data.get("mem_used_gb_mean")
                mem_total_gb = gpu_data.get("mem_total_gb")
                mem_util_mean = gpu_data.get("mem_util_mean")

                gpu_mem_percent_str = "N/A"  # Default
                if mem_used_gb_mean is not None and mem_total_gb is not None and mem_total_gb > 0:
                    # Prioritize calculation from displayed absolute values
                    calculated_percent = (mem_used_gb_mean / mem_total_gb) * 100
                    gpu_mem_percent_str = fmt_f(calculated_percent)
                elif mem_util_mean is not None:
                    # Fallback to the averaged percentage
                    # if calculation from absolutes isn't possible
                    gpu_mem_percent_str = fmt_f(mem_util_mean)

                mem_str = f"  MEM {gpu_mem_percent_str:>3}% ({mem_used_gb_mean:>5.1f}/{mem_total_gb:>5.1f}GB)"

                # GPU Utilization
                util_mean = gpu_data.get("util_mean")
                util_max = gpu_data.get("util_max")
                util_min = gpu_data.get("util_min")
                util_median = gpu_data.get("util_median")
                util_time_above = gpu_data.get("util_time_above_90pct")

                util_str = (
                    f"  UTL: Mean {fmt_f(util_mean)}% | "
                    f"Max {fmt_f(util_max)} | "
                    f"Min {fmt_f(util_min)} | "
                    f"Med {fmt_f(util_median)} | "
                    f">90% {fmt_f(util_time_above)}"
                )

                logger.info(f"GPU{gpu_id}:")
                logger.info(mem_str)
                logger.info(util_str)
                logger.info("")  # Blank line between devices
        else:
            logger.info("GPU - Not Available / No GPU Data\n")

        cpu_data = resource_info.get("cpu")
        if cpu_data:
            cpu_mem_used_gb_mean = cpu_data.get("mem_used_gb_mean")
            cpu_mem_total_gb = cpu_data.get("mem_total_gb")
            cpu_mem_util_mean = cpu_data.get("mem_util_mean")

            cpu_mem_percent_str = "N/A"
            if (
                cpu_mem_used_gb_mean is not None
                and cpu_mem_total_gb is not None
                and cpu_mem_total_gb > 0
            ):
                calculated_cpu_percent = (cpu_mem_used_gb_mean / cpu_mem_total_gb) * 100
                cpu_mem_percent_str = fmt_f(calculated_cpu_percent)
            elif cpu_mem_util_mean is not None:
                cpu_mem_percent_str = fmt_f(cpu_mem_util_mean)

            # mem_str = f"  MEM: {cpu_mem_percent_str}% ({cpu_mem_used_gb_mean}/{cpu_mem_total_gb}GB)"
            mem_str = f"  MEM {cpu_mem_percent_str:>3}% ({cpu_mem_used_gb_mean:>5.1f}/{cpu_mem_total_gb:>5.1f}GB)"

            cpu_util_mean = cpu_data.get("util_mean")
            cpu_util_max = cpu_data.get("util_max")
            cpu_util_min = cpu_data.get("util_min")
            cpu_util_median = cpu_data.get("util_median")
            cpu_util_time_above_thresh = cpu_data.get("util_time_above_90pct")

            util_str = (
                f"  UTL: Mean {fmt_f(cpu_util_mean)}% | "
                f"Max {fmt_f(cpu_util_max)} | "
                f"Min {fmt_f(cpu_util_min)} | "
                f"Med {fmt_f(cpu_util_median)} | "
                f">90% {fmt_f(cpu_util_time_above_thresh)}"
            )

            logger.info("CPU:")
            logger.info(mem_str)
            logger.info(util_str)
            logger.info("")
        else:
            logger.info("CPU - Not Available / No CPU Data\n")

    # logger.info("\n" + "=" * 70)


@runtime_checkable
class TrainingObserver(Protocol):
    def on_segment_start(
        self, segment_num: int, scan_length: int, segment_display_info: dict[str, Any]
    ) -> None:
        pass

    def on_training_progress(
        self,
        hp_runtime_states: list[HPRuntimeState],
        segment_info_for_display_from_trainer: dict[str, Any],
        current_returns: list[float | None] = None,
        learning_indicators: list[str] | None = None,
    ) -> None:
        pass

    def on_evaluation_complete(self, segment_num: int, num_evaluated: int) -> None:
        pass


class ProgressDisplayObserver(TrainingObserver):
    def __init__(
        self,
        initial_num_hyperparams: int,
        target_total_steps: int,
        all_possible_milestones: list[int],
        display_fn: Callable = display_hp_progress,
        display_completed_hps_flag: bool = True,
        system_monitor: SystemMonitor | None = None,
        gpu_id_to_display: int = 0,
        util_threshold_pct_display: float = 90.0,  # New params
    ):
        self.initial_num_hyperparams = initial_num_hyperparams
        self.target_total_steps = target_total_steps
        self.all_possible_milestones = all_possible_milestones
        self.display_fn = display_fn
        self.display_completed_hps_flag = display_completed_hps_flag
        self._current_segment_display_info: dict[str, Any] = {
            "segment_num": 0,
            "scan_length": 0,
            "eta_next_eval": "N/A",
            "eta_next_rephase": "N/A",
            "jit_train_s": "N/A",
            "jit_eval_s": "N/A",
            "last_segment_duration_s": 0.0,
        }
        self.system_monitor = system_monitor
        self.gpu_id_to_display = gpu_id_to_display
        self.util_threshold_pct_display = util_threshold_pct_display

    def on_segment_start(
        self, segment_num: int, scan_length: int, segment_display_info: dict[str, Any]
    ) -> None:
        self._current_segment_display_info = segment_display_info

    def on_training_progress(
        self,
        hp_runtime_states: list[HPRuntimeState],
        segment_info_for_display_from_trainer: dict[str, Any],
        current_returns: list[float | None] = None,
        learning_indicators: list[str] | None = None,
    ) -> None:
        current_steps = [hp.env_steps for hp in hp_runtime_states]
        active_status = [hp.is_active for hp in hp_runtime_states]
        milestone_indices = [hp.current_milestone_target_idx for hp in hp_runtime_states]

        resource_info_for_display = None
        interval_status_msg = "SysMon inactive or interval times missing"

        if self.system_monitor and self.system_monitor.is_running:
            t_start = segment_info_for_display_from_trainer.get("segment_work_t_start")
            t_end = segment_info_for_display_from_trainer.get("segment_work_t_end")

            if t_start is not None and t_end is not None and t_end > t_start:
                # logger.debug(f"ProgressObserver: Fetching interval stats [{t_start:.3f} - {t_end:.3f}]")
                interval_stats = self.system_monitor.get_utilization_stats_for_interval(
                    t_start=t_start,
                    t_end=t_end,
                    gpu_id=self.gpu_id_to_display,
                    utilization_threshold_pct=self.util_threshold_pct_display,
                )
                # logger.debug(f"ProgressObserver: Raw interval_stats: {interval_stats}")

                num_samples = interval_stats.get("num_samples_in_interval", 0)
                if num_samples > 0:
                    # interval_status_msg = f"OK ({num_samples} samples)"
                    resource_info_for_display = {
                        "gpus": [],
                        "cpu": {},
                        "interval_info": {
                            "num_samples_in_interval": num_samples,
                            "duration_s": interval_stats.get("interval_duration_s"),
                            "actual_log_span_s": interval_stats.get("actual_log_span_s"),
                            # "status_msg": interval_status_msg
                            "status_msg": "",
                        },
                    }
                    # GPU data
                    resource_info_for_display["gpus"].append(
                        {
                            "id": self.gpu_id_to_display,
                            "util_mean": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_mean"
                            ),
                            "util_max": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_max"
                            ),
                            "util_min": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_min"
                            ),
                            "util_median": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_median"
                            ),
                            "util_std": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_std"
                            ),
                            "util_time_above_90pct": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_util_time_above_{self.util_threshold_pct_display:.0f}pct"
                            ),
                            "mem_util_mean": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_mem_util_mean"
                            ),
                            "mem_used_gb_mean": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_mem_used_gb_mean"
                            ),
                            "mem_total_gb": interval_stats.get(
                                f"gpu_{self.gpu_id_to_display}_total_memory_gb"
                            ),
                        }
                    )
                    # CPU Data
                    resource_info_for_display["cpu"] = {
                        "util_mean": interval_stats.get("system_cpu_util_mean"),
                        "util_max": interval_stats.get("system_cpu_util_max"),
                        "util_min": interval_stats.get("system_cpu_util_min"),
                        "util_median": interval_stats.get("system_cpu_util_median"),
                        "util_std": interval_stats.get("system_cpu_util_std"),
                        "util_time_above_90pct": interval_stats.get(
                            f"system_cpu_util_time_above_{self.util_threshold_pct_display:.0f}pct"
                        ),
                        "mem_util_mean": interval_stats.get("system_mem_util_mean"),
                        "mem_used_gb_mean": interval_stats.get("system_mem_used_gb_mean"),
                        "mem_total_gb": interval_stats.get("system_total_memory_gb"),
                    }
                else:  # No samples in interval or error from get_utilization_stats_for_interval
                    interval_status_msg = f"No samples in interval [{t_start:.2f}-{t_end:.2f}]"
                    logger.debug(
                        f"ProgressObserver: {interval_status_msg}. Error in stats: {interval_stats.get('error')}"
                    )

            else:  # t_start or t_end is None, or t_end <= t_start
                interval_status_msg = "Invalid/missing interval times from phaser"
                logger.debug(
                    f"ProgressObserver: {interval_status_msg} t_start={t_start}, t_end={t_end}"
                )

            # Fallback if resource_info_for_display is still None
            if resource_info_for_display is None:
                logger.debug(
                    f"ProgressObserver: Falling back to get_current_utilization. Reason: {interval_status_msg}"
                )
                current_util_data = self.system_monitor.get_current_utilization()
                resource_info_for_display = {
                    "gpus": [],
                    "cpu": {},
                    "interval_info": {
                        "status_msg": ""
                    },
                }
                g0_current = current_util_data.get("gpu", {})
                resource_info_for_display["gpus"].append(
                    {
                        "id": self.gpu_id_to_display,
                        "util_mean": g0_current.get(
                            f"gpu_{self.gpu_id_to_display}_utilization"
                        ),  # Store as 'mean' for display consistency
                        "mem_used_gb_mean": g0_current.get(
                            f"gpu_{self.gpu_id_to_display}_memory_used_gb"
                        ),
                        "mem_total_gb": g0_current.get(
                            f"gpu_{self.gpu_id_to_display}_total_memory_gb"
                        ),
                    }
                )
                cpu_current = current_util_data.get("system", {})
                resource_info_for_display["cpu"] = {
                    "util_mean": cpu_current.get("cpu_utilization"),
                    "mem_used_gb_mean": cpu_current.get("system_memory_used_gb"),  # From current
                    "mem_total_gb": cpu_current.get("system_total_memory_gb"),
                }
        else:  # System monitor not active
            resource_info_for_display = {
                "interval_info": {"status_msg": "System monitor not active or not provided"}
            }
            logger.debug("ProgressObserver: System monitor not active or not present.")

        # logger.debug(
        #     f"ProgressObserver: Final resource_info_for_display to be used: {resource_info_for_display}"
        # )
        self.display_fn(
            self.initial_num_hyperparams,
            current_steps,
            active_status,
            self.target_total_steps,
            self.all_possible_milestones,
            milestone_indices,
            segment_info_for_display_from_trainer,
            current_returns,
            learning_indicators=learning_indicators,
            resource_info=resource_info_for_display,
            display_completed_hps_flag=self.display_completed_hps_flag,
        )
