import dataclasses
import logging
import math
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hyperlax.base_types import (
    AlgorithmGlobalSetupArgs,
    AnakinTrainOutput,
    EvaluationMetrics,
    GenericState,
    HPRuntimeState,
)
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.evaluator.setup import setup_distributed_evaluator
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.layout.base_layout import create_eval_strategy, create_train_strategy
from hyperlax.layout.summary import summarize_layout
from hyperlax.logger.decorator import log_gpu_memory
from hyperlax.logger.hp_progress_bar import TrainingObserver
from hyperlax.logger.metrics import (
    calculate_learning_trend_indicators,
    log_and_save_aggregated_metrics,  # noqa: F401
    reduce_metrics_over_batching_axes,
)
from hyperlax.logger.return_tracker import (
    HyperparamReturns,
    initialize_hyperparam_returns,
    update_hyperparam_returns,
)
from hyperlax.trainer.utils import (
    _calculate_milestones,
    build_active_learner_state,
    extract_current_avg_returns_to_display,
    get_env_step_counter,
    merge_active_state_into_full,
    sum_total_env_steps_per_hyperparam,
)
from hyperlax.utils.algo_setup import (
    AlgorithmNetworkSetup,
    select_train_style_fn,
    setup_generic_learner,
)

logger = logging.getLogger(__name__)


class HPStaticConfig(NamedTuple):
    steps_per_update: int
    original_index: int


@dataclasses.dataclass(frozen=True)
class AlgoStateAndJitFns:
    learner_state: Any
    train_one_unit_fn: Callable[[GenericState], AnakinTrainOutput]
    evaluate_fn: Callable[[GenericState], EvaluationMetrics]
    eval_keys: Any
    is_sliced: bool
    original_indices_map: list[int]
    train_strategy: DistributionStrategy
    eval_strategy: DistributionStrategy


class TrainingLoopState(NamedTuple):
    hp_runtime_states: list[HPRuntimeState]
    current_algo_state_and_jit_fns: AlgoStateAndJitFns
    master_full_learner_state: Any
    return_trackers: list["HyperparamReturns"]
    segment_count: int


SetupInitialFn = Callable[[Any, AlgorithmGlobalSetupArgs, int], AlgoStateAndJitFns]
ReSetupForActiveFn = Callable[
    [Any, list[int], Any, Any, AlgorithmGlobalSetupArgs, int], AlgoStateAndJitFns
]
GetEnvStepsFn = Callable[[Any, int], int]
MergeActiveStateIntoFullFn = Callable[[Any, Any, list[int]], Any]


@dataclasses.dataclass(frozen=True)
class AlgoSetupFns:
    setup_initial: SetupInitialFn
    re_setup_for_active: ReSetupForActiveFn


@dataclasses.dataclass(frozen=True)
class EvaluationSegmentInfo:
    indices_to_eval: list[int]
    original_indices: list[int]
    milestone_values: list[int]


@dataclasses.dataclass(frozen=True)
class ExecutionHistory:
    segments_executed: int
    scan_lengths_per_segment: list[int]
    evaluations_performed_info: list[dict[str, Any]]  # e.g. {"segment": X, "evaluations": [...]}
    completed_hps_original_indices: list[int]


@dataclasses.dataclass(frozen=True)
class PhaseTrainingResult:
    final_env_steps: list[int]
    final_active_status: list[bool]
    hp_final_milestone_target_indices: list[int]
    return_trackers: list["HyperparamReturns"]
    execution_history: ExecutionHistory
    total_jit_time: float = 0.0
    total_execution_time: float = 0.0
    final_master_learner_state: Any | None = None

    def get_summary_metrics(self) -> dict[str, float]:
        completed_hps = sum(1 for active in self.final_active_status if not active)
        total_hps = len(self.final_active_status)
        avg_final_steps = (
            sum(self.final_env_steps) / len(self.final_env_steps) if self.final_env_steps else 0
        )
        return {
            "completion_rate": float(completed_hps / total_hps if total_hps > 0 else 0.0),
            "avg_final_env_steps": float(avg_final_steps),
            "total_segments_executed": float(self.execution_history.segments_executed),
            "total_jit_time": self.total_jit_time,
            "total_execution_time": self.total_execution_time,
        }


@dataclasses.dataclass
class ETACalculator:
    total_train_updates_executed_for_avg: int = 0
    total_train_exec_time_for_avg: float = 0.0
    total_evals_executed_for_avg: int = 0
    total_eval_exec_time_for_avg: float = 0.0
    total_jit_time: float = 0.0
    total_pure_execution_time: float = 0.0
    jit_train_fn_duration: float | None = None
    jit_eval_fn_duration: float | None = None
    timed_function_ids: set[Any] = dataclasses.field(default_factory=set)  # To track jitted func.

    def _log_time(self, name: str, duration: float, is_jit: bool):
        if is_jit:
            logger.info(f"JIT & First Exec of {name}: {duration:.3f}s")
        else:
            logger.debug(f"Exec of {name}: {duration:.3f}s")

    def time_execution(
        self,
        func_to_time: Callable,
        args_tuple: tuple,
        func_id_for_jit_tracking: Any,
        log_name: str,
        enable_timing_logs: bool,
    ) -> tuple[Any, float, float | None, float, float]:
        is_first_run = func_id_for_jit_tracking not in self.timed_function_ids
        jit_duration_this_call: float | None = None

        t_start_exec = time.time()
        result = func_to_time(*args_tuple)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple) and any(
            hasattr(el, "block_until_ready") for el in result if el is not None
        ):
            for el in result:
                if hasattr(el, "block_until_ready") and el is not None:
                    el.block_until_ready()
        t_end_exec = time.time()
        wall_time = t_end_exec - t_start_exec

        if is_first_run:
            self.timed_function_ids.add(func_id_for_jit_tracking)
            jit_duration_this_call = wall_time
            if enable_timing_logs:
                self._log_time(log_name, wall_time, is_jit=True)
        else:
            if enable_timing_logs:
                self._log_time(log_name, wall_time, is_jit=False)

        return result, wall_time, jit_duration_this_call, t_start_exec, t_end_exec

    def record_train_segment_time(
        self, exec_time: float, num_updates_in_segment: int, jit_time: float | None
    ):
        if jit_time is not None:
            self.jit_train_fn_duration = jit_time
            self.total_jit_time += jit_time
        else:
            if num_updates_in_segment > 0:
                self.total_train_updates_executed_for_avg += num_updates_in_segment
                self.total_train_exec_time_for_avg += exec_time
                self.total_pure_execution_time += exec_time

    def record_eval_time(self, exec_time: float, jit_time: float | None):
        if jit_time is not None:
            self.jit_eval_fn_duration = jit_time
            self.total_jit_time += jit_time
        else:  # Pure execution time
            self.total_evals_executed_for_avg += 1
            self.total_eval_exec_time_for_avg += exec_time
            self.total_pure_execution_time += exec_time

    def get_avg_train_update_exec_time(self) -> float | None:
        if self.total_train_updates_executed_for_avg > 0:
            return self.total_train_exec_time_for_avg / self.total_train_updates_executed_for_avg
        return None

    def get_avg_eval_exec_time(self) -> float | None:
        if self.total_evals_executed_for_avg > 0:
            return self.total_eval_exec_time_for_avg / self.total_evals_executed_for_avg
        return None

    def reset_jit_flags(self):
        self.jit_train_fn_duration = None
        self.jit_eval_fn_duration = None
        self.timed_function_ids.clear()

    def reset_averages(self):
        self.total_train_updates_executed_for_avg = 0
        self.total_train_exec_time_for_avg = 0.0
        self.total_evals_executed_for_avg = 0
        self.total_eval_exec_time_for_avg = 0.0

    def calculate_eta_info(
        self,
        active_hp_rts_view: list[HPRuntimeState],
        hp_static_configs: list[HPStaticConfig],
        all_possible_milestone_values: list[int],
        target_total_steps: int,
    ) -> dict[str, str]:
        avg_train_time_per_update = self.get_avg_train_update_exec_time()
        avg_eval_time = self.get_avg_eval_exec_time()

        eta_next_eval_str = "N/A (waiting)"
        if avg_train_time_per_update is not None:
            min_time_to_next_eval_seconds = float("inf")
            eval_will_happen = False
            for hp_state in active_hp_rts_view:
                hp_config = next(
                    (c for c in hp_static_configs if c.original_index == hp_state.original_index),
                    None,
                )
                if not hp_config:
                    continue
                current_target_idx = hp_state.current_milestone_target_idx
                if current_target_idx < len(all_possible_milestone_values):
                    target_milestone_value = all_possible_milestone_values[current_target_idx]
                    steps_to_reach_milestone = target_milestone_value - hp_state.env_steps
                    if steps_to_reach_milestone <= 0:
                        min_time_to_next_eval_seconds = 0
                        eval_will_happen = True
                        break
                    if hp_config.steps_per_update > 0:
                        updates_needed = math.ceil(
                            steps_to_reach_milestone / hp_config.steps_per_update
                        )
                        estimated_train_time = updates_needed * avg_train_time_per_update
                        min_time_to_next_eval_seconds = min(
                            min_time_to_next_eval_seconds, estimated_train_time
                        )
                        eval_will_happen = True
            if eval_will_happen and min_time_to_next_eval_seconds != float("inf"):
                total_eta_seconds = min_time_to_next_eval_seconds
                if avg_eval_time is not None:
                    if min_time_to_next_eval_seconds == 0:
                        total_eta_seconds = avg_eval_time
                    else:
                        total_eta_seconds += avg_eval_time
                eta_next_eval_str = str(timedelta(seconds=total_eta_seconds)).split(".")[0]
            elif not eval_will_happen:
                eta_next_eval_str = "N/A (no upcoming eval)"

        eta_next_rephase_str = "N/A (waiting)"
        if avg_train_time_per_update is not None:
            min_time_to_rephase_seconds = float("inf")
            rephase_will_happen = False
            for hp_state in active_hp_rts_view:
                hp_config = next(
                    (c for c in hp_static_configs if c.original_index == hp_state.original_index),
                    None,
                )
                if not hp_config:
                    continue
                steps_to_completion = target_total_steps - hp_state.env_steps
                if steps_to_completion <= 0:
                    min_time_to_rephase_seconds = 0
                    rephase_will_happen = True
                    break
                if hp_config.steps_per_update > 0:
                    updates_needed_for_completion = math.ceil(
                        steps_to_completion / hp_config.steps_per_update
                    )
                    estimated_train_time_to_completion = (
                        updates_needed_for_completion * avg_train_time_per_update
                    )
                    min_time_to_rephase_seconds = min(
                        min_time_to_rephase_seconds, estimated_train_time_to_completion
                    )
                    rephase_will_happen = True
            if rephase_will_happen and min_time_to_rephase_seconds != float("inf"):
                eta_next_rephase_str = str(timedelta(seconds=min_time_to_rephase_seconds)).split(
                    "."
                )[0]
            elif not rephase_will_happen:
                eta_next_rephase_str = "N/A (all complete or no progress)"

        return {
            "eta_next_eval": eta_next_eval_str,
            "eta_next_rephase": eta_next_rephase_str,
        }


def _validate_training_parameters(
    target_total_steps: int,
    num_evaluation_milestones: int,
    initial_num_hyperparams: int,
    hp_steps_per_update: list[int],
) -> None:
    if target_total_steps < 0:
        raise ValueError("target_total_steps must be non-negative")
    if num_evaluation_milestones < 0:
        raise ValueError("num_evaluation_milestones must be non-negative")
    if initial_num_hyperparams < 0:
        raise ValueError("initial_num_hyperparams must be non-negative")
    if len(hp_steps_per_update) != initial_num_hyperparams:
        raise ValueError("Length mismatch: hp_steps_per_update vs initial_num_hyperparams")
    if initial_num_hyperparams > 0 and not all(s > 0 for s in hp_steps_per_update):
        raise ValueError("All hp_steps_per_update must be positive for active HPs.")


def _initialize_execution_history() -> ExecutionHistory:
    return ExecutionHistory(0, [], [], [])


def _initialize_hp_runtime_states(
    initial_num_hyperparams: int,
    master_full_learner_state: Any,
    train_strategy: DistributionStrategy,
) -> list[HPRuntimeState]:
    runtime_states = []
    for i in range(initial_num_hyperparams):
        env_steps = get_env_step_counter(
            master_full_learner_state,
            i,
            train_strategy,
        )
        hp_runtime_state = HPRuntimeState(
            original_index=i,
            env_steps=env_steps,
            is_active=True,
            current_milestone_target_idx=0,
        )
        runtime_states.append(hp_runtime_state)
    return runtime_states


def _get_active_hp_info(
    hp_runtime_states: list[HPRuntimeState],
) -> tuple[list[HPRuntimeState], list[int]]:
    active_rts = [hp for hp in hp_runtime_states if hp.is_active]
    active_indices = [hp.original_index for hp in active_rts]
    return active_rts, active_indices


def _calculate_fixed_scan_length_for_phase(
    active_hp_runtime_states: list[HPRuntimeState],
    hp_static_configs: list[HPStaticConfig],
    all_possible_milestone_values: list[int],
) -> int:
    if not active_hp_runtime_states:
        return 1

    scan_lengths_needed = []
    for hp_state in active_hp_runtime_states:
        current_target_idx = hp_state.current_milestone_target_idx
        hp_config = next(
            sc for sc in hp_static_configs if sc.original_index == hp_state.original_index
        )

        if current_target_idx < len(all_possible_milestone_values):
            target_milestone_value = all_possible_milestone_values[current_target_idx]
            steps_to_reach = target_milestone_value - hp_state.env_steps
            if steps_to_reach > 0 and hp_config.steps_per_update > 0:
                updates_needed = math.ceil(steps_to_reach / hp_config.steps_per_update)
                scan_lengths_needed.append(updates_needed)

    if scan_lengths_needed:
        fixed_scan_length = int(jnp.min(jnp.array(scan_lengths_needed)))
        fixed_scan_length = max(1, fixed_scan_length)
        logger.debug(
            f"Calculated fixed scan_length for phase: {fixed_scan_length} (from individual needs: {scan_lengths_needed})"
        )
        return fixed_scan_length
    else:
        return 1


def _determine_dynamic_scan_length(
    active_hp_rts: list[HPRuntimeState],
    hp_static_configs: list[HPStaticConfig],
    milestones: list[int],
) -> int:
    if not active_hp_rts:
        return 0
    min_updates = float("inf")
    needs_positive_steps = False
    at_milestone_needs_eval = False

    for hp_state in active_hp_rts:
        hp_config = next(
            c for c in hp_static_configs if c.original_index == hp_state.original_index
        )
        if hp_state.current_milestone_target_idx < len(milestones):
            target_steps = milestones[hp_state.current_milestone_target_idx]
            steps_to_go = target_steps - hp_state.env_steps

            if steps_to_go <= 0:
                at_milestone_needs_eval = True
            else:
                updates_needed = (
                    math.ceil(steps_to_go / hp_config.steps_per_update)
                    if hp_config.steps_per_update > 0
                    else float("inf")
                )
                needs_positive_steps = True
                min_updates = min(min_updates, updates_needed)

    if at_milestone_needs_eval:
        return 1
    if needs_positive_steps:
        return int(max(1, min_updates))
    return 0


def _determine_segment_scan_length(
    active_hp_rts_view: list[HPRuntimeState],
    hp_static_configs: list[HPStaticConfig],
    all_possible_milestone_values: list[int],
    trainer_style: str,
    current_phase_scan_length: int | None,
) -> int:
    if trainer_style == "phased_scan_step":
        if current_phase_scan_length is None:
            raise ValueError(
                "current_phase_scan_length cannot be None for 'phased_scan_step' during main loop."
            )
        return current_phase_scan_length
    elif trainer_style == "phased":
        return _determine_dynamic_scan_length(
            active_hp_rts_view, hp_static_configs, all_possible_milestone_values
        )
    else:
        raise ValueError(f"Unknown trainer_style: {trainer_style}")


def _update_hp_env_steps_after_train_segment(
    hp_rts: list[HPRuntimeState],
    active_indices: list[int],
    master_full_learner_state: Any,
    train_strategy: DistributionStrategy,
) -> list[HPRuntimeState]:
    updated_rts = list(hp_rts)
    for i, rt_state in enumerate(updated_rts):
        if rt_state.original_index in active_indices:
            new_steps = get_env_step_counter(
                master_full_learner_state,
                rt_state.original_index,
                train_strategy,
            )
            updated_rts[i] = rt_state._replace(env_steps=new_steps)
    return updated_rts


def _identify_hps_for_evaluation(
    hp_rts: list[HPRuntimeState], active_orig_indices: list[int], milestones: list[int]
) -> EvaluationSegmentInfo:
    indices_to_eval, orig_indices_eval, milestone_vals_hit = [], [], []
    for rt_state in hp_rts:
        if not rt_state.is_active:
            continue
        m_idx = rt_state.current_milestone_target_idx
        if m_idx < len(milestones) and rt_state.env_steps >= milestones[m_idx]:
            orig_indices_eval.append(rt_state.original_index)
            milestone_vals_hit.append(milestones[m_idx])
            try:
                indices_to_eval.append(active_orig_indices.index(rt_state.original_index))
            except ValueError:
                raise RuntimeError(f"HP {rt_state.original_index} for eval not in active set.")
    return EvaluationSegmentInfo(indices_to_eval, orig_indices_eval, milestone_vals_hit)


def _update_milestone_progress(
    hp_rts: list[HPRuntimeState],
    eval_info: EvaluationSegmentInfo,
    milestones: list[int],
) -> tuple[list[HPRuntimeState], list[dict[str, Any]]]:
    updated_rts, history = list(hp_rts), []
    for i, rt_state in enumerate(updated_rts):
        if rt_state.original_index in eval_info.original_indices:
            m_idx = rt_state.current_milestone_target_idx
            if m_idx < len(milestones):
                history.append(
                    {
                        "hp_original_idx": rt_state.original_index,
                        "milestone_value_hit": milestones[m_idx],
                        "steps_at_eval": rt_state.env_steps,
                    }
                )
                updated_rts[i] = rt_state._replace(current_milestone_target_idx=m_idx + 1)
                logger.debug(f"HP {rt_state.original_index} advanced to milestone {m_idx + 1}")
    return updated_rts, history


def _check_and_mark_completed_hps(
    hp_rts: list[HPRuntimeState], target_steps: int
) -> tuple[list[HPRuntimeState], list[int]]:
    updated_rts, completed_indices = list(hp_rts), []
    for i, rt_state in enumerate(updated_rts):
        if rt_state.is_active and rt_state.env_steps >= target_steps:
            updated_rts[i] = rt_state._replace(is_active=False)
            completed_indices.append(rt_state.original_index)
            logger.info(f"HP {rt_state.original_index} COMPLETED at {rt_state.env_steps} steps")
    return updated_rts, completed_indices


def _execute_training_segment_w_diagnostics(
    algo_state_and_jit_fns: AlgoStateAndJitFns,
    segment_scan_length: int,
    trainer_style: str,
    enable_timing_logs: bool,
    enable_gpu_mem_log: bool,
    eta_calculator: ETACalculator,
    segment_num: int,
) -> tuple[AnakinTrainOutput, float, float, float]:
    current_learner_state = algo_state_and_jit_fns.learner_state
    train_fn_core = algo_state_and_jit_fns.train_one_unit_fn
    is_sliced = algo_state_and_jit_fns.is_sliced

    output_from_segment: AnakinTrainOutput | None = None
    segment_total_wall_time: float = 0.0
    overall_t_start_exec: float | None = None
    overall_t_end_exec: float | None = None

    if segment_scan_length == 0:
        logger.warning(f"Segment {segment_num}: No training units to execute (scan_length=0).")
        current_ts = time.time()
        return (
            AnakinTrainOutput(
                learner_state=current_learner_state,
                episode_metrics={},
                train_metrics={},
            ),
            0.0,
            current_ts,
            current_ts,
        )

    func_id_for_jit = id(train_fn_core)
    log_name_base = f"TrainFn_Seg{segment_num}{'_S' if is_sliced else '_I'}"

    if trainer_style == "phased_scan_step":
        log_name = f"{log_name_base}_Scanned"

        @log_gpu_memory(name=f"Seg{segment_num}_TrainScanned_GPUMem")
        def _call_scanned_gpu_logged():
            return train_fn_core(current_learner_state)

        call_fn = (
            _call_scanned_gpu_logged
            if enable_gpu_mem_log
            else lambda: train_fn_core(current_learner_state)
        )

        output_from_segment, wall_time, jit_time, t_s, t_e = eta_calculator.time_execution(
            call_fn, (), func_id_for_jit, log_name, enable_timing_logs
        )
        segment_total_wall_time = wall_time
        overall_t_start_exec, overall_t_end_exec = t_s, t_e
        eta_calculator.record_train_segment_time(wall_time, segment_scan_length, jit_time)

    elif trainer_style == "phased":
        temp_learner_state = current_learner_state
        for i_iter in range(segment_scan_length):
            log_name_iter = f"{log_name_base}_Iter{i_iter}"

            @log_gpu_memory(name=f"Seg{segment_num}_TrainIter{i_iter}_GPUMem")
            def _call_single_step_gpu_logged(state_for_step):
                return train_fn_core(state_for_step)

            current_call_fn = lambda state: (
                _call_single_step_gpu_logged(state) if enable_gpu_mem_log else train_fn_core(state)
            )

            output_step, wall_time_step, jit_time_step, t_s_step, t_e_step = (
                eta_calculator.time_execution(
                    current_call_fn,
                    (temp_learner_state,),
                    func_id_for_jit,
                    log_name_iter,
                    enable_timing_logs,
                )
            )

            if overall_t_start_exec is None:
                overall_t_start_exec = t_s_step
            overall_t_end_exec = t_e_step

            eta_calculator.record_train_segment_time(wall_time_step, 1, jit_time_step)

            output_from_segment = output_step
            temp_learner_state = output_from_segment.learner_state
            segment_total_wall_time += wall_time_step
    else:
        raise ValueError(f"Unknown trainer_style: {trainer_style}")

    if output_from_segment is None:
        logger.error(
            "Training segment executed but output_from_segment is None. This should not happen."
        )
        current_ts = time.time()
        return (
            AnakinTrainOutput(
                learner_state=current_learner_state,
                episode_metrics={},
                train_metrics={},
            ),
            0.0,
            current_ts,
            current_ts,
        )

    if overall_t_start_exec is None or overall_t_end_exec is None:
        logger.warning(
            f"Segment {segment_num}: Timestamps for training were not set despite non-zero scan length. Using current time."
        )
        current_ts = time.time()
        overall_t_start_exec = overall_t_start_exec or current_ts
        overall_t_end_exec = overall_t_end_exec or current_ts

    return (
        output_from_segment,
        segment_total_wall_time,
        overall_t_start_exec,
        overall_t_end_exec,
    )


def _execute_evaluation_segment_w_diagnostics(
    algo_state_and_jit_fns: AlgoStateAndJitFns,
    eval_info: EvaluationSegmentInfo,
    enable_timing_logs: bool,
    enable_gpu_mem_log: bool,
    eta_calculator: ETACalculator,
    segment_num: int,
) -> tuple[EvaluationMetrics, float, float, float]:
    if not eval_info.indices_to_eval:
        current_ts = time.time()
        return (
            EvaluationMetrics(episode_metrics={}, other_metrics={}),
            0.0,
            current_ts,
            current_ts,
        )

    logger.info(
        f"Segment {segment_num}: Evaluating {len(eval_info.indices_to_eval)} HPs for milestones: {eval_info.milestone_values}"
    )

    eval_fn_core = algo_state_and_jit_fns.evaluate_fn
    learner_state_for_eval = algo_state_and_jit_fns.learner_state
    is_sliced = algo_state_and_jit_fns.is_sliced

    func_id_for_jit = id(eval_fn_core)
    log_name = f"EvalFn_Seg{segment_num}{'_S' if is_sliced else '_I'}"

    @log_gpu_memory(name=f"Seg{segment_num}_Evaluation_GPUMem")
    def _call_eval_gpu_logged():
        return eval_fn_core(learner_state_for_eval)

    call_fn = (
        _call_eval_gpu_logged
        if enable_gpu_mem_log
        else lambda: eval_fn_core(learner_state_for_eval)
    )

    eval_output_raw, wall_time, jit_time, t_s, t_e = eta_calculator.time_execution(
        call_fn, (), func_id_for_jit, log_name, enable_timing_logs
    )
    eta_calculator.record_eval_time(wall_time, jit_time)

    processed_eval_output: EvaluationMetrics
    if isinstance(eval_output_raw, EvaluationMetrics):
        processed_eval_output = eval_output_raw
    elif hasattr(eval_output_raw, "episode_metrics") and hasattr(eval_output_raw, "train_metrics"):
        processed_eval_output = EvaluationMetrics(
            episode_metrics=eval_output_raw.episode_metrics,
            other_metrics=eval_output_raw.train_metrics,
        )
    else:
        logger.error(f"Unexpected return type from evaluation function: {type(eval_output_raw)}")
        processed_eval_output = EvaluationMetrics(episode_metrics={}, other_metrics={})

    return processed_eval_output, wall_time, t_s, t_e


def _apply_gpu_log_to_setup(
    setup_fn_to_wrap: Callable, enable_gpu_log: bool, name: str
) -> Callable:
    if enable_gpu_log:

        @log_gpu_memory(name=name)
        def _wrapped_setup_fn(*args, **kwargs):
            return setup_fn_to_wrap(*args, **kwargs)

        return _wrapped_setup_fn
    return setup_fn_to_wrap


def _execute_algo_setup(
    algo_setup_fns: AlgoSetupFns,
    initial_hp_configs: Any,
    non_vec_hyperparams,
    global_args: AlgorithmGlobalSetupArgs,
    initial_scan_len: int,
    enable_gpu_mem_log: bool,
) -> tuple[AlgoStateAndJitFns, Any, DistributionStrategy, DistributionStrategy]:
    actual_setup_initial_fn = _apply_gpu_log_to_setup(
        algo_setup_fns.setup_initial, enable_gpu_mem_log, "AlgorithmInitialSetup"
    )
    initial_algo_s_f = actual_setup_initial_fn(
        initial_hp_configs, non_vec_hyperparams, global_args, initial_scan_len
    )
    return (
        initial_algo_s_f,
        initial_algo_s_f.learner_state,
        initial_algo_s_f.train_strategy,
        initial_algo_s_f.eval_strategy,
    )


def _execute_re_setup_after_phasing(
    algo_setup_fns: AlgoSetupFns,
    master_state: Any,
    active_indices: list[int],
    active_configs_slice: Any,
    initial_hp_configs: Any,
    non_vec_hyperparams: Any,
    global_args: AlgorithmGlobalSetupArgs,
    seg_num_log: int,
    new_phase_scan_len: int,
    enable_gpu_mem_log: bool,
) -> AlgoStateAndJitFns:
    actual_re_setup_fn = _apply_gpu_log_to_setup(
        algo_setup_fns.re_setup_for_active,
        enable_gpu_mem_log,
        f"Segment{seg_num_log}_ReSetupActive",
    )
    return actual_re_setup_fn(
        master_state,
        active_indices,
        active_configs_slice,
        initial_hp_configs,
        non_vec_hyperparams,
        global_args,
        new_phase_scan_len,
    )


def _execute_merge_active_state_into_full(
    master_state: Any,
    active_slice_state: Any,
    active_indices: list[int],
    seg_num_log: int,
    train_strategy,
    enable_gpu_mem_log: bool,
) -> Any:
    actual_merge_fn = _apply_gpu_log_to_setup(
        merge_active_state_into_full,
        enable_gpu_mem_log,
        f"Segment{seg_num_log}_MergeActiveState",
    )
    return actual_merge_fn(master_state, active_slice_state, active_indices, train_strategy)


def _handle_metrics_and_trackers(
    config: BaseExperimentConfig,
    train_output_final: AnakinTrainOutput | None,
    eval_results: EvaluationMetrics,
    active_hp_orig_indices: list[int],
    initial_num_hyperparams: int,
    last_full_eval_metrics_tree: Any | None,
    current_segment_count: int,
    current_return_trackers: list[HyperparamReturns],
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
    master_full_learner_state: Any | None,
) -> tuple[list[HyperparamReturns], Any | None]:
    metrics_handler_start_time = time.time()

    active_train_metrics = {}
    if train_output_final:
        active_train_metrics.update(train_output_final.episode_metrics or {})
        active_train_metrics.update(train_output_final.train_metrics or {})

    active_eval_metrics = {}
    active_eval_metrics.update(eval_results.episode_metrics or {})
    active_eval_metrics.update(eval_results.other_metrics or {})

    updated_last_full_eval_tree, eval_agg_dict = log_and_save_aggregated_metrics(
        base_exp_path=config.logger.base_exp_path,
        aggregate_metrics_flag=config.logger.aggregate_metrics,
        active_train_metrics=active_train_metrics,
        active_eval_metrics=active_eval_metrics,
        active_hyp_indices=jnp.asarray(active_hp_orig_indices),
        num_hyperparams=initial_num_hyperparams,
        last_full_eval_metrics=last_full_eval_metrics_tree,
        eval_step=current_segment_count,
        train_strategy=train_strategy,
        eval_strategy=eval_strategy,
    )

    active_eval_metrics_reduced_for_tracking = reduce_metrics_over_batching_axes(
        active_eval_metrics, eval_strategy, "eval"
    )

    raw_episode_returns_for_tracking = active_eval_metrics_reduced_for_tracking.get(
        "episode_return"
    )
    updated_trackers = current_return_trackers

    if raw_episode_returns_for_tracking is not None and len(raw_episode_returns_for_tracking) > 0:
        timesteps_for_active_hps_list = None
        if master_full_learner_state is not None and hasattr(
            master_full_learner_state, "total_env_steps_counter"
        ):
            total_steps_for_all_hps = sum_total_env_steps_per_hyperparam(
                total_env_steps_counter=master_full_learner_state.total_env_steps_counter,
                strategy=train_strategy,
            )
            timesteps_for_active_hps_jnp = total_steps_for_all_hps[
                jnp.array(active_hp_orig_indices)
            ]
            timesteps_for_active_hps_list = timesteps_for_active_hps_jnp.tolist()
        else:
            logger.warning(
                "master_full_learner_state is None or missing total_env_steps_counter for metric tracking. Using eval_step as dummy timesteps."
            )
            timesteps_for_active_hps_list = [current_segment_count] * len(active_hp_orig_indices)

        updated_trackers = update_hyperparam_returns(
            return_trackers=current_return_trackers,
            active_indices=jnp.asarray(active_hp_orig_indices),
            timesteps=timesteps_for_active_hps_list,
            raw_episode_returns_reduced=raw_episode_returns_for_tracking,
            aggregated_eval_stats=eval_agg_dict,
        )
    else:
        logger.info(
            "No episode returns generated from evaluation. Skipping return tracker update."
        )

    metrics_handler_duration = time.time() - metrics_handler_start_time
    logger.info(
        f"Metrics calculating for segment {current_segment_count} took: {metrics_handler_duration:.4f}s"
    )

    return updated_trackers, updated_last_full_eval_tree


# ---- Main Phaser ----
def run_training_w_phaser(
    target_total_steps: int,
    num_evaluation_milestones: int,
    initial_num_hyperparams: int,
    hp_steps_per_update: list[int],
    algo_setup_fns: AlgoSetupFns,
    initial_hyperparam_configs_for_algo: Any,
    non_vec_hyperparams: Any,
    global_args: AlgorithmGlobalSetupArgs,
    observers: list[TrainingObserver] | None = None,
    include_final_master_state: bool = False,
) -> PhaseTrainingResult:
    _validate_training_parameters(
        target_total_steps,
        num_evaluation_milestones,
        initial_num_hyperparams,
        hp_steps_per_update,
    )
    if observers is None:
        observers = []
    config = global_args.config
    trainer_style = config.training.trainer_style
    enable_jit_timing_flag = config.logger.enable_timing_logs
    enable_gpu_memory_logging_flag = config.logger.enable_gpu_memory_logging
    logger.info(
        f"Trainer style: {trainer_style}, Target Steps: {target_total_steps}, Eval Milestones: {num_evaluation_milestones}"
    )
    if enable_jit_timing_flag:
        logger.debug("JIT timing is enabled.")
    if enable_gpu_memory_logging_flag:
        logger.info("GPU Memory Logging is enabled for JAX setup/exec functions.")

    all_possible_milestone_values = _calculate_milestones(
        target_total_steps, num_evaluation_milestones
    )
    logger.debug(f"Calculated evaluation milestones: {all_possible_milestone_values}")

    hp_static_configs_list = [
        HPStaticConfig(steps, i) for i, steps in enumerate(hp_steps_per_update)
    ]
    eta_calculator = ETACalculator()

    _hp_rts_pre_setup = [HPRuntimeState(i, 0, True, 0) for i in range(initial_num_hyperparams)]
    _initial_scan_length_for_setup = 1
    if trainer_style == "phased_scan_step":
        _initial_scan_length_for_setup = _calculate_fixed_scan_length_for_phase(
            [hp for hp in _hp_rts_pre_setup if hp.is_active],
            hp_static_configs_list,
            all_possible_milestone_values,
        )
    logger.info(f"Initial scan_length for algorithm setup: {_initial_scan_length_for_setup}")

    (
        initial_algo_s_f,
        master_full_learner_state_from_setup,
        train_strategy,
        eval_strategy,
    ) = _execute_algo_setup(
        algo_setup_fns,
        initial_hyperparam_configs_for_algo,
        non_vec_hyperparams,
        global_args,
        _initial_scan_length_for_setup,
        enable_gpu_memory_logging_flag,
    )

    initial_return_trackers: list[HyperparamReturns] = initialize_hyperparam_returns(
        global_args.config
    )
    logger.debug(f"initial_return_trackers: {initial_return_trackers}")

    hp_rts_current = _initialize_hp_runtime_states(
        initial_num_hyperparams,
        initial_algo_s_f.learner_state,
        train_strategy,
    )

    training_state = TrainingLoopState(
        hp_rts_current,
        initial_algo_s_f,
        master_full_learner_state_from_setup,
        initial_return_trackers,
        0,
    )
    execution_history = _initialize_execution_history()
    current_phase_scan_length = (
        _initial_scan_length_for_setup if trainer_style == "phased_scan_step" else None
    )
    _last_known_learning_indicators_phaser_state: dict[int, str] = {}
    last_full_eval_metrics_tree = None

    avg_returns_display_init = extract_current_avg_returns_to_display(
        training_state.return_trackers, training_state.hp_runtime_states
    )
    learning_indicators_init = calculate_learning_trend_indicators(
        training_state.hp_runtime_states,
        training_state.return_trackers,
        _last_known_learning_indicators_phaser_state,
    )
    segment_info_for_display_init = {
        "segment_num": 0,
        "scan_length": 0,
        "eta_next_eval": "N/A",
        "eta_next_rephase": "N/A",
        "jit_train_s": "N/A",
        "jit_eval_s": "N/A",
        "last_segment_duration_s": 0.0,
        "segment_work_t_start": time.time(),
        "segment_work_t_end": time.time(),
    }
    for obs in observers:
        obs.on_segment_start(0, 0, segment_info_for_display_init)
        obs.on_training_progress(
            training_state.hp_runtime_states,
            segment_info_for_display_init,
            avg_returns_display_init,
            learning_indicators_init,
        )

    while any(hp.is_active for hp in training_state.hp_runtime_states):
        current_segment_count = training_state.segment_count + 1
        logger.info(f"--- Starting SEGMENT {current_segment_count} ---")

        active_hp_rts_view, active_hp_orig_indices = _get_active_hp_info(
            training_state.hp_runtime_states
        )
        if not active_hp_orig_indices:
            logger.info("No active HPs remaining. Exiting training loop.")
            break

        segment_scan_length = _determine_segment_scan_length(
            active_hp_rts_view,
            hp_static_configs_list,
            all_possible_milestone_values,
            trainer_style,
            current_phase_scan_length,
        )
        logger.info(
            f"Segment {current_segment_count}: Determined scan_length = {segment_scan_length}"
        )

        eta_info = eta_calculator.calculate_eta_info(
            active_hp_rts_view,
            hp_static_configs_list,
            all_possible_milestone_values,
            target_total_steps,
        )
        jit_train_str = (
            f"{eta_calculator.jit_train_fn_duration:.2f}s"
            if eta_calculator.jit_train_fn_duration is not None
            else "N/A"
        )
        jit_eval_str = (
            f"{eta_calculator.jit_eval_fn_duration:.2f}s"
            if eta_calculator.jit_eval_fn_duration is not None
            else "N/A"
        )

        segment_info_for_observer_start = {
            "segment_num": current_segment_count,
            "scan_length": segment_scan_length,
            "eta_next_eval": eta_info["eta_next_eval"],
            "eta_next_rephase": eta_info["eta_next_rephase"],
            "jit_train_s": jit_train_str,
            "jit_eval_s": jit_eval_str,
            "last_segment_duration_s": segment_info_for_display_init.get(
                "last_segment_duration_s", 0.0
            ),
        }
        for obs in observers:
            obs.on_segment_start(
                current_segment_count,
                segment_scan_length,
                segment_info_for_observer_start,
            )

        execution_history.scan_lengths_per_segment.append(segment_scan_length)

        train_segment_actual_t_start: float | None = None
        train_segment_actual_t_end: float | None = None
        eval_segment_actual_t_start: float | None = None
        eval_segment_actual_t_end: float | None = None
        wall_time_train_segment: float = 0.0
        wall_time_eval_segment: float = 0.0
        anakin_output_from_train_segment: AnakinTrainOutput | None = None

        # --- Training Phase ---
        if segment_scan_length > 0:
            logger.info(
                f"Segment {current_segment_count}: Executing training for {segment_scan_length} units."
            )
            (
                anakin_output_from_train_segment,
                wall_time_train_segment,
                train_segment_actual_t_start,
                train_segment_actual_t_end,
            ) = _execute_training_segment_w_diagnostics(
                training_state.current_algo_state_and_jit_fns,
                segment_scan_length,
                trainer_style,
                enable_jit_timing_flag,
                enable_gpu_memory_logging_flag,
                eta_calculator,
                current_segment_count,
            )
            logger.info(
                f"Segment {current_segment_count}: Training exec. took {wall_time_train_segment:.3f}s."
            )

            training_state = training_state._replace(
                current_algo_state_and_jit_fns=dataclasses.replace(
                    training_state.current_algo_state_and_jit_fns,
                    learner_state=anakin_output_from_train_segment.learner_state,
                )
            )
            master_full_learner_state_updated = _execute_merge_active_state_into_full(
                training_state.master_full_learner_state,
                training_state.current_algo_state_and_jit_fns.learner_state,
                active_hp_orig_indices,
                current_segment_count,
                train_strategy,
                enable_gpu_memory_logging_flag,
            )
            training_state = training_state._replace(
                master_full_learner_state=master_full_learner_state_updated
            )

            hp_rts_after_train = _update_hp_env_steps_after_train_segment(
                training_state.hp_runtime_states,
                active_hp_orig_indices,
                training_state.master_full_learner_state,
                train_strategy,
            )
            training_state = training_state._replace(hp_runtime_states=hp_rts_after_train)
        else:
            logger.warning(
                f"Segment {current_segment_count}: No training units executed (scan_length = 0)."
            )
            current_ts = time.time()
            train_segment_actual_t_start, train_segment_actual_t_end = (
                current_ts,
                current_ts,
            )
            anakin_output_from_train_segment = AnakinTrainOutput(
                learner_state=training_state.current_algo_state_and_jit_fns.learner_state,
                episode_metrics={},
                train_metrics={},
            )

        # --- Evaluation Phase ---
        eval_segment_info = _identify_hps_for_evaluation(
            training_state.hp_runtime_states,
            active_hp_orig_indices,
            all_possible_milestone_values,
        )
        eval_results = EvaluationMetrics({}, {})

        if eval_segment_info.indices_to_eval and config.training.num_eval_episodes > 0:
            (
                eval_results,
                wall_time_eval_segment,
                eval_segment_actual_t_start,
                eval_segment_actual_t_end,
            ) = _execute_evaluation_segment_w_diagnostics(
                training_state.current_algo_state_and_jit_fns,
                eval_segment_info,
                enable_jit_timing_flag,
                enable_gpu_memory_logging_flag,
                eta_calculator,
                current_segment_count,
            )
            logger.info(
                f"Segment {current_segment_count}: Evaluation execution took {wall_time_eval_segment:.3f}s."
            )
            for obs in observers:
                obs.on_evaluation_complete(
                    current_segment_count, len(eval_segment_info.original_indices)
                )
        else:
            if eval_segment_info.indices_to_eval:
                logger.info(
                    f"Segment {current_segment_count}: Skipping evaluation as num_eval_episodes is 0."
                )
            current_ts = time.time()
            eval_segment_actual_t_start, eval_segment_actual_t_end = (
                current_ts,
                current_ts,
            )
            wall_time_eval_segment = 0.0

        display_interval_t_start = train_segment_actual_t_start
        display_interval_t_end = train_segment_actual_t_end
        if segment_scan_length == 0 and eval_segment_info.indices_to_eval:
            display_interval_t_start = eval_segment_actual_t_start
            display_interval_t_end = eval_segment_actual_t_end
        current_ts_fallback = time.time()
        display_interval_t_start = (
            display_interval_t_start
            if display_interval_t_start is not None
            else current_ts_fallback
        )
        display_interval_t_end = (
            display_interval_t_end if display_interval_t_end is not None else current_ts_fallback
        )
        total_jax_work_duration_s = wall_time_train_segment + wall_time_eval_segment
        segment_info_for_progress_update = segment_info_for_observer_start.copy()
        segment_info_for_progress_update["last_segment_duration_s"] = total_jax_work_duration_s
        segment_info_for_progress_update["segment_work_t_start"] = display_interval_t_start
        segment_info_for_progress_update["segment_work_t_end"] = display_interval_t_end
        segment_info_for_display_init["last_segment_duration_s"] = total_jax_work_duration_s

        # --- Update HP States Post-Evaluation Milestone Advancement ---
        current_return_trackers, last_full_eval_metrics_tree = _handle_metrics_and_trackers(
            config,
            anakin_output_from_train_segment,
            eval_results,
            active_hp_orig_indices,
            initial_num_hyperparams,
            last_full_eval_metrics_tree,
            current_segment_count,
            training_state.return_trackers,
            train_strategy=train_strategy,
            eval_strategy=eval_strategy,
            master_full_learner_state=training_state.master_full_learner_state,
        )
        training_state = training_state._replace(return_trackers=current_return_trackers)

        # --- Update HP States Post-Evaluation Milestone Advancement ---
        hp_rts_after_milestone_update = list(training_state.hp_runtime_states)
        if eval_results and eval_segment_info.indices_to_eval:
            hp_rts_after_milestone_update, eval_history_details = _update_milestone_progress(
                hp_rts_after_milestone_update,
                eval_segment_info,
                all_possible_milestone_values,
            )
            if eval_history_details:
                execution_history.evaluations_performed_info.append(
                    {
                        "segment": current_segment_count,
                        "evaluations": eval_history_details,
                    }
                )
        training_state = training_state._replace(hp_runtime_states=hp_rts_after_milestone_update)

        # --- Update display ---
        learning_indicators_list_for_display = calculate_learning_trend_indicators(
            training_state.hp_runtime_states,
            training_state.return_trackers,
            _last_known_learning_indicators_phaser_state,
        )
        avg_returns_for_display = extract_current_avg_returns_to_display(
            training_state.return_trackers, training_state.hp_runtime_states
        )
        for obs in observers:
            obs.on_training_progress(
                training_state.hp_runtime_states,
                segment_info_for_progress_update,
                avg_returns_for_display,
                learning_indicators_list_for_display,
            )

        # --- Completion Check ---
        hp_rts_after_completion_check, newly_completed_hps_indices = _check_and_mark_completed_hps(
            training_state.hp_runtime_states,
            target_total_steps,
        )
        if newly_completed_hps_indices:
            execution_history.completed_hps_original_indices.extend(newly_completed_hps_indices)
            logger.info(
                f"Segment {current_segment_count}: HPs {newly_completed_hps_indices} completed."
            )
        training_state = training_state._replace(hp_runtime_states=hp_rts_after_completion_check)

        # --- Re-Phasing if any HP completed ---
        if newly_completed_hps_indices:
            active_hp_rts_after_completion, active_indices_for_next_phase = _get_active_hp_info(
                training_state.hp_runtime_states
            )
            if active_indices_for_next_phase:
                logger.info(
                    f"Re-phasing for segment {current_segment_count + 1}. Active HPs: {active_indices_for_next_phase}"
                )
                eta_calculator.reset_jit_flags()

                new_phase_scan_length_for_setup = 1
                if trainer_style == "phased_scan_step":
                    new_phase_scan_length_for_setup = _calculate_fixed_scan_length_for_phase(
                        active_hp_rts_after_completion,
                        hp_static_configs_list,
                        all_possible_milestone_values,
                    )
                    current_phase_scan_length = new_phase_scan_length_for_setup

                logger.info(
                    f"Re-phasing: New scan_length for setup of next phase: {new_phase_scan_length_for_setup}"
                )

                active_configs_slice = {
                    key: batch.get_slice(jnp.array(active_indices_for_next_phase))
                    for key, batch in initial_hyperparam_configs_for_algo.items()
                    if hasattr(batch, "get_slice")
                }

                new_algo_s_and_f = _execute_re_setup_after_phasing(
                    algo_setup_fns,
                    training_state.master_full_learner_state,
                    active_indices_for_next_phase,
                    active_configs_slice,
                    initial_hyperparam_configs_for_algo,
                    non_vec_hyperparams,
                    global_args,
                    current_segment_count + 1,
                    new_phase_scan_length_for_setup,
                    enable_gpu_memory_logging_flag,
                )
                training_state = training_state._replace(
                    current_algo_state_and_jit_fns=new_algo_s_and_f
                )
                train_strategy = new_algo_s_and_f.train_strategy
                eval_strategy = new_algo_s_and_f.eval_strategy
            else:
                logger.info("All HPs have completed training after re-phasing check.")
                break

        training_state = training_state._replace(segment_count=current_segment_count)
        # logger.info(f"--- Finished SEGMENT {current_segment_count} ---")

    logger.info(
        f"--- Phaser Training Finished. Total Segments Executed: {training_state.segment_count}. ---"
    )
    execution_history = dataclasses.replace(
        execution_history, segments_executed=training_state.segment_count
    )
    final_master_state_optional = (
        training_state.master_full_learner_state if include_final_master_state else None
    )

    return PhaseTrainingResult(
        final_env_steps=[hp.env_steps for hp in training_state.hp_runtime_states],
        final_active_status=[hp.is_active for hp in training_state.hp_runtime_states],
        hp_final_milestone_target_indices=[
            hp.current_milestone_target_idx for hp in training_state.hp_runtime_states
        ],
        return_trackers=training_state.return_trackers,
        execution_history=execution_history,
        total_jit_time=eta_calculator.total_jit_time,
        total_execution_time=eta_calculator.total_pure_execution_time,
        final_master_learner_state=final_master_state_optional,
    )


def build_generic_phaser_setup_fns(
    # Callbacks for setting up the learner
    build_network_setup_fn: Callable[[], AlgorithmNetworkSetup],
    build_network_fn: Callable,
    build_optimizer_fn: Callable,
    build_update_step_fn: Callable,
    build_distributed_layout_fn: Callable,
    build_warmup_rollout_fn: Callable,
    # Callbacks for setting up the evaluator
    get_eval_act_fn_callback: Callable,
    extract_params_for_eval_fn: Callable,
    extract_norm_params_for_eval_fn: Callable,
) -> AlgoSetupFns:
    phaser_factory_logger = logging.getLogger(f"{logger.name}.GENERIC_PHASER_FACTORY")

    def _setup_initial(
        initial_hyperparam_configs: dict[str, Any],
        hyperparam_non_vectorizeds: Any,  # TODO type
        global_args: AlgorithmGlobalSetupArgs,
        num_updates_per_scan: int,
    ) -> AlgoStateAndJitFns:
        algo_hp_batch = next(iter(initial_hyperparam_configs.values()))
        phaser_factory_logger.info(
            f"Initial setup for {algo_hp_batch.shape[0]} HPs, num_scan={num_updates_per_scan}"
        )

        train_strategy = create_train_strategy(
            num_update_batch=global_args.config.training.update_batch_size,
            num_devices=len(jax.devices()),
            num_seeds=global_args.config.training.num_agents_slash_seeds,
            num_hyperparams=algo_hp_batch.shape[0],
            jit_enabled=global_args.config.training.jit_enabled,
        )
        eval_strategy = create_eval_strategy(
            num_devices=len(jax.devices()),
            num_seeds=global_args.config.training.num_agents_slash_seeds,
            num_hyperparams=algo_hp_batch.shape[0],
            jit_enabled=global_args.config.training.jit_enabled,
        )

        single_fn, scanned_fn, networks, initial_state = setup_generic_learner(
            global_args,
            initial_hyperparam_configs,
            hyperparam_non_vectorizeds,
            num_updates_per_scan,
            train_strategy,
            build_network_setup_fn,
            build_network_fn,
            build_optimizer_fn,
            build_update_step_fn,
            build_distributed_layout_fn,
            with_params=True,
            build_warmup_rollout_fn=build_warmup_rollout_fn,
        )

        train_fn_selected = select_train_style_fn(
            global_args.config.training.trainer_style, single_fn, scanned_fn
        )

        if global_args.config.logger.enable_summarize_layout:
            summarize_layout(
                {
                    "train_strategy": train_strategy,
                    "eval_strategy": eval_strategy,
                    "initial_state": initial_state,
                }
            )

        # Assumes first network in dict is the primary one for eval (actor or q-network)
        primary_eval_net_key = next(iter(networks.keys()))
        eval_net_struct = networks[primary_eval_net_key]

        eval_fn_distributed = setup_distributed_evaluator(
            eval_env=global_args.eval_env,
            eval_key_base=global_args.algo_specific_keys[1],
            get_eval_act_fn_callback=get_eval_act_fn_callback,
            model_apply_fn_for_eval=eval_net_struct.apply,
            normalizer_fns=global_args.normalizer_fns,
            config=global_args.config,
            eval_strategy=eval_strategy,
            train_strategy=train_strategy,
            extract_params_for_eval_fn=extract_params_for_eval_fn,
            extract_norm_params_for_eval_fn=extract_norm_params_for_eval_fn,
        )

        return AlgoStateAndJitFns(
            learner_state=initial_state,
            train_one_unit_fn=train_fn_selected,
            evaluate_fn=eval_fn_distributed,
            eval_keys=None,
            is_sliced=False,
            original_indices_map=list(range(algo_hp_batch.shape[0])),
            train_strategy=train_strategy,
            eval_strategy=eval_strategy,
        )

    def _re_setup_for_active(
        master_full_learner_state: Any,
        active_hp_original_indices: list[int],
        active_hp_configs_slice: dict[str, Any],
        full_hp_configs_initial: dict[str, Any],
        hyperparam_non_vectorizeds: Any,
        global_args: AlgorithmGlobalSetupArgs,
        num_updates_per_scan: int,
    ) -> AlgoStateAndJitFns:
        phaser_factory_logger.info(
            f"Re-setup for {len(active_hp_original_indices)} HPs, num_scan={num_updates_per_scan}"
        )

        train_strategy_active = create_train_strategy(
            num_update_batch=global_args.config.training.update_batch_size,
            num_devices=len(jax.devices()),
            num_seeds=global_args.config.training.num_agents_slash_seeds,
            num_hyperparams=len(active_hp_original_indices),
            jit_enabled=global_args.config.training.jit_enabled,
        )
        eval_strategy_active = create_eval_strategy(
            num_devices=len(jax.devices()),
            num_seeds=global_args.config.training.num_agents_slash_seeds,
            num_hyperparams=len(active_hp_original_indices),
            jit_enabled=global_args.config.training.jit_enabled,
        )

        active_learner_state_slice = build_active_learner_state(
            master_full_learner_state,
            jnp.array(active_hp_original_indices),
            full_hp_configs_initial["algo"].shape[0],
            train_strategy_active,
        )

        # print_pytree_shapes(active_learner_state_slice)

        single_fn_active, scanned_fn_active, networks, _ = setup_generic_learner(
            global_args,
            full_hp_configs_initial,
            hyperparam_non_vectorizeds,
            num_updates_per_scan,
            train_strategy_active,
            build_network_setup_fn,
            build_network_fn,
            build_optimizer_fn,
            build_update_step_fn,
            build_distributed_layout_fn,
            with_params=False,
            build_warmup_rollout_fn=None,
        )
        train_fn_active_selected = select_train_style_fn(
            global_args.config.training.trainer_style,
            single_fn_active,
            scanned_fn_active,
        )

        primary_eval_net_key = next(iter(networks.keys()))
        eval_net_struct = networks[primary_eval_net_key]

        eval_fn_distributed_active = setup_distributed_evaluator(
            eval_env=global_args.eval_env,
            eval_key_base=global_args.algo_specific_keys[1],
            get_eval_act_fn_callback=get_eval_act_fn_callback,
            model_apply_fn_for_eval=eval_net_struct.apply,
            normalizer_fns=global_args.normalizer_fns,
            config=global_args.config,
            eval_strategy=eval_strategy_active,
            train_strategy=train_strategy_active,
            extract_params_for_eval_fn=extract_params_for_eval_fn,
            extract_norm_params_for_eval_fn=extract_norm_params_for_eval_fn,
        )

        return AlgoStateAndJitFns(
            learner_state=active_learner_state_slice,
            train_one_unit_fn=train_fn_active_selected,
            evaluate_fn=eval_fn_distributed_active,
            eval_keys=None,
            is_sliced=True,
            original_indices_map=list(active_hp_original_indices),
            train_strategy=train_strategy_active,
            eval_strategy=eval_strategy_active,
        )

    return AlgoSetupFns(setup_initial=_setup_initial, re_setup_for_active=_re_setup_for_active)
