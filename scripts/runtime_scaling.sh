#!/bin/bash
#
# runtime_scaling.sh
#
# Benchmark the vectorization performance of HyperLaX for PPO + <MODEL> under
# different conditions of hyperparameter heterogeneity.
#
# Supports multiple models (e.g., ppo_mlp, ppo_drpqc). If a model is missing some
# configs, the script will SKIP those runs gracefully without failing.
#
# Usage examples:
#   bash runtime_scaling.sh
#   bash runtime_scaling.sh --grouping                      # also run grouped variant
#   bash runtime_scaling.sh --no-grouping                   # (default) only ungrouped
#   bash runtime_scaling.sh --models "ppo_mlp ppo_drpqc" --batch-sizes "4 8 16"
#   bash runtime_scaling.sh --variant lower_num_envs        # also try *_lower_num_envs
#   bash runtime_scaling.sh --variants "lower_num_envs alt" # multiple variants
#   bash runtime_scaling.sh --variants lower_num_envs --no-base  # only variants
#
# Notes:
# - Assumes config IDs follow the pattern with the model name embedded, e.g.:
#     runtime_scaling_1_<MODEL>_homo_best_case
#     runtime_scaling_2_<MODEL>_hetero_algo_hps
#     runtime_scaling_3_<MODEL>_hetero_network_hps
#     runtime_scaling_4_<MODEL>_hetero_algo_and_network_hps
# - You can add variant suffixes via flags, e.g.:
#     runtime_scaling_1_ppo_mlp_homo_best_case_lower_num_envs
#   Missing configs (per-model) will be skipped automatically.

set -e # Exit immediately if a command exits with a non-zero status.
shopt -s extglob  # safer parsing & pattern features

# --- Default Configuration ---
RUN_MODIFIER="quick"
NUM_SAMPLES=2                  # Small default for quick tests
BATCH_SIZES="1 2"              # Small default for quick tests
OUTPUT_DIR="./results_runtime_scaling_benchmark_$(date +%Y%m%d_%H%M%S)"
MODELS=("ppo_mlp")             # Default model set

# Grouping control (ungrouped always runs; grouped added when enabled)
GROUPING_ENABLED=false

# Variant handling: include base configs plus zero or more suffix variants
VARIANT_SUFFIXES=()            # user-provided variants; base included by default
INCLUDE_BASE_VARIANT=true

# --- Helpers ---
to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

parse_grouping_value() {
  local v
  v="$(to_lower "$1")"
  case "$v" in
    1|true|yes|on|enable|enabled|grouping) GROUPING_ENABLED=true ;;
    0|false|no|off|disable|disabled|no-grouping) GROUPING_ENABLED=false ;;
    *) echo "Invalid value for --grouping: $1 (use true/false, enabled/disabled, yes/no, on/off)"; exit 1 ;;
  esac
}

# Normalize variant suffix: ensure it starts with a single underscore
norm_variant() {
  local v="$1"
  [[ -z "$v" ]] && { echo ""; return; }
  v="${v#_}"           # strip leading underscores
  echo "_$v"
}

# Expand a list of base config ids with variant suffixes (portable; no namerefs).
# Usage: expand_with_variants IN_ARRAY_NAME OUT_ARRAY_NAME
expand_with_variants() {
  local in_name="$1"
  local out_name="$2"

  # Read caller's input array safely
  local -a __in=()
  eval "__in=(\"\${${in_name}[@]}\")"

  # Build variant list (base + normalized suffixes), de-duped
  local -a __vars=()
  if [[ "$INCLUDE_BASE_VARIANT" == true ]]; then
    __vars+=("")
  fi
  local v
  for v in "${VARIANT_SUFFIXES[@]}"; do
    [[ -n "$v" ]] && __vars+=("$(norm_variant "$v")")
  done

  # De-duplicate while preserving order
  local -a __uniq=()
  for v in "${__vars[@]}"; do
    local found=""
    local _u
    for _u in "${__uniq[@]}"; do
      if [[ "$_u" == "$v" ]]; then found=1; break; fi
    done
    [[ -z "$found" ]] && __uniq+=("$v")
  done

  # Cross product: each base id × each variant
  local -a __out=()
  local base
  for base in "${__in[@]}"; do
    for v in "${__uniq[@]}"; do
      __out+=("${base}${v}")
    done
  done

  # Write result back to caller's array
  eval "$out_name=(\"\${__out[@]}\")"
}

# --- CLI parsing (robust to stripped quotes, supports = and multi-args) ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-modifier)
      RUN_MODIFIER="$2"; shift 2 ;;

    --num-samples)
      NUM_SAMPLES="$2"; shift 2 ;;

    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;

    --model)
      MODELS=("$2"); shift 2 ;;

    --models)
      MODELS=()
      shift
      # collect tokens until next --flag or end
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        IFS=',' read -r -a _tmp <<< "$1"
        for m in "${_tmp[@]}"; do
          [[ -n "$m" ]] && MODELS+=("$m")
        done
        shift
      done
      ;;

    --batch-sizes)
      BATCH_SIZES=""
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        IFS=',' read -r -a _sizes <<< "$1"
        for s in "${_sizes[@]}"; do
          [[ -n "$s" ]] && BATCH_SIZES+="${BATCH_SIZES:+ }$s"
        done
        shift
      done
      ;;

    # equals-style variants
    --models=*)
      MODELS=()
      val="${1#*=}"; val="${val//,/ }"
      read -r -a MODELS <<< "$val"
      shift ;;

    --batch-sizes=*)
      val="${1#*=}"; val="${val//,/ }"
      BATCH_SIZES="$val"
      shift ;;

    --run-modifier=* )
      RUN_MODIFIER="${1#*=}" ; shift ;;

    --num-samples=* )
      NUM_SAMPLES="${1#*=}" ; shift ;;

    --output-dir=* )
      OUTPUT_DIR="${1#*=}" ; shift ;;

    # --- Grouping toggles ---
    --grouping)
      GROUPING_ENABLED=true; shift ;;

    --no-grouping)
      GROUPING_ENABLED=false; shift ;;

    --grouping=*)
      parse_grouping_value "${1#*=}"; shift ;;

    # --- Variants: accept comma/space list; values like "lower_num_envs" or "_foo" are OK
    --variant)
      # Guard: ensure a value follows and isn't another flag
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Missing value for --variant" >&2
        exit 1
      fi
      VARIANT_SUFFIXES+=("$2"); shift 2 ;;

    --variants)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        IFS=',' read -r -a _v <<< "$1"
        for v in "${_v[@]}"; do [[ -n "$v" ]] && VARIANT_SUFFIXES+=("$v"); done
        shift
      done
      ;;

    --variant=*)
      val="${1#*=}"; val="${val//,/ }"
      read -r -a _v <<< "$val"
      for v in "${_v[@]}"; do [[ -n "$v" ]] && VARIANT_SUFFIXES+=("$v"); done
      shift ;;

    --no-base)
      INCLUDE_BASE_VARIANT=false; shift ;;

    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

echo "================================================="
echo "        HyperLaX Speedup Benchmark Script        "
echo "================================================="
echo "Run Modifier:      $RUN_MODIFIER"
echo "Number of Samples: $NUM_SAMPLES"
echo "Batch Sizes:       $BATCH_SIZES"
echo "Models:            ${MODELS[*]}"
echo "Grouping:          $([[ "$GROUPING_ENABLED" == true ]] && echo enabled || echo disabled)"
if [[ ${#VARIANT_SUFFIXES[@]} -gt 0 || "$INCLUDE_BASE_VARIANT" != true ]]; then
  # Show normalized variants for clarity
  _disp_variants=()
  if [[ "$INCLUDE_BASE_VARIANT" == true ]]; then _disp_variants+=("(base)"); fi
  for v in "${VARIANT_SUFFIXES[@]}"; do _disp_variants+=("$(norm_variant "$v")"); done
  echo "Variants:          ${_disp_variants[*]}"
else
  echo "Variants:          (base only)"
fi
echo "Output Directory:  $OUTPUT_DIR"
echo "================================================="

# --- Base command shared by all runs ---
BASE_CMD=(
    python -m hyperlax.cli sweep-hp-samples
    --env-config "gymnax.pendulum"
    --run-length-modifier "$RUN_MODIFIER"
    --num-samples "$NUM_SAMPLES"
    --log-level "INFO"
)

# --- Helper: build config names for a given model (base IDs only) ---
get_configs_for_model() {
    local MODEL="$1"

    local HOMO_TEMPLATE="runtime_scaling_1_%MODEL%_homo_best_case"
    local HETERO_TEMPLATES=(
        "runtime_scaling_2_%MODEL%_hetero_algo_hps"
        "runtime_scaling_3_%MODEL%_hetero_network_hps"
        "runtime_scaling_4_%MODEL%_hetero_algo_and_network_hps"
    )

    HOMO_CONFIGS_BASE=("${HOMO_TEMPLATE//%MODEL%/$MODEL}")

    HETERO_CONFIGS_BASE=()
    local t
    for t in "${HETERO_TEMPLATES[@]}"; do
        HETERO_CONFIGS_BASE+=("${t//%MODEL%/$MODEL}")
    done

    # Expand with variants
    HOMO_CONFIGS=()
    expand_with_variants HOMO_CONFIGS_BASE HOMO_CONFIGS

    HETERO_CONFIGS=()
    expand_with_variants HETERO_CONFIGS_BASE HETERO_CONFIGS

    # Defensive fallback: if expansion fails, use base so we don't no-op silently
    if [[ ${#HOMO_CONFIGS[@]} -eq 0 ]]; then HOMO_CONFIGS=("${HOMO_CONFIGS_BASE[@]}"); fi
    if [[ ${#HETERO_CONFIGS[@]} -eq 0 ]]; then HETERO_CONFIGS=("${HETERO_CONFIGS_BASE[@]}"); fi
}

# --- Helper: resolve the python module path for a config id ---
config_module_path() {
    local CONFIG_ID="$1"
    # Matches the import path HyperLaX tries to import
    echo "hyperlax.configs.algo.$CONFIG_ID"
}

# --- Helper: check if a config module is importable in Python ---
config_available() {
    local CONFIG_ID="$1"
    local MOD_PATH
    MOD_PATH="$(config_module_path "$CONFIG_ID")"
    # Use Python to check availability without producing noisy stack traces
    python - "$MOD_PATH" <<'PY' >/dev/null 2>&1
import sys, importlib.util
mod = sys.argv[1]
spec = importlib.util.find_spec(mod)
sys.exit(0 if spec is not None else 1)
PY
}

# --- Helper: run and skip gracefully on failure (even with set -e) ---
SKIPPED_RUNS=()
SKIPPED_CONFIGS=()

run_or_skip() {
    local label="$1"; shift
    echo -e "\n[RUNNING] $label"
    set +e
    "$@"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "[SKIP] $label (exit code $rc)."
        SKIPPED_RUNS+=("$label")
        return 1
    fi
    return 0
}

# ==============================================================================
# MAIN LOOP: iterate over requested models
# ==============================================================================
for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n#############################################"
    echo "Model: $MODEL"
    echo "#############################################"

    # Build config names for this model (including variants)
    get_configs_for_model "$MODEL"

    # ------------------------------------------------------------------------------
    # EXPERIMENT 1: HOMOGENEOUS THROUGHPUT SCALING (BEST CASE)
    #   Grouping is a no-op here; we still run ungrouped vectorized baseline.
    # ------------------------------------------------------------------------------
    echo -e "\n\n--- EXPERIMENT 1: Homogeneous Throughput Scaling ---"
    for HOMO_CONFIG in "${HOMO_CONFIGS[@]}"; do
      echo "--- CONFIG: $HOMO_CONFIG ---"

      if ! config_available "$HOMO_CONFIG"; then
          msg="Model=$MODEL | $HOMO_CONFIG | (all modes) — config module not found, skipping entire config"
          echo "[SKIP] $msg"
          SKIPPED_CONFIGS+=("$msg")
          continue
      fi

      # 1a. Sequential Baseline
      OUT_PATH="$OUTPUT_DIR/$MODEL/$HOMO_CONFIG/sequential"
      run_or_skip "Model=$MODEL | $HOMO_CONFIG | sequential" \
      "${BASE_CMD[@]}" \
          --algo-and-network-config "$HOMO_CONFIG" \
          --output-root "$OUT_PATH" \
          --sequential True

      # 1b. Vectorized runs (ungrouped; grouping would be a no-op in homogeneous case)
      for bs in $BATCH_SIZES; do
          OUT_PATH="$OUTPUT_DIR/$MODEL/$HOMO_CONFIG/vectorized_B$bs"
          run_or_skip "Model=$MODEL | $HOMO_CONFIG | vectorized B=$bs" \
          "${BASE_CMD[@]}" \
              --algo-and-network-config "$HOMO_CONFIG" \
              --output-root "$OUT_PATH" \
              --hparam-batch-size "$bs"
      done
    done

    # ------------------------------------------------------------------------------
    # EXPERIMENTS 2, 3, 4: HETEROGENEOUS COST ANALYSIS
    #   Always run UNGROUPED vectorized; optionally ADD GROUPED when enabled.
    # ------------------------------------------------------------------------------
    for ALGO_CONFIG in "${HETERO_CONFIGS[@]}"; do
        echo -e "\n\n--- HETEROGENEOUS EXPERIMENT ---"
        echo "--- CONFIG: $ALGO_CONFIG ---"

        if ! config_available "$ALGO_CONFIG"; then
            msg="Model=$MODEL | $ALGO_CONFIG | (all modes) — config module not found, skipping entire config"
            echo "[SKIP] $msg"
            SKIPPED_CONFIGS+=("$msg")
            continue
        fi

        # a. Sequential Baseline
        OUT_PATH="$OUTPUT_DIR/$MODEL/$ALGO_CONFIG/sequential"
        run_or_skip "Model=$MODEL | $ALGO_CONFIG | sequential" \
        "${BASE_CMD[@]}" \
            --algo-and-network-config "$ALGO_CONFIG" \
            --output-root "$OUT_PATH" \
            --sequential True

        # b. Ungrouped Vectorized (always)
        for bs in $BATCH_SIZES; do
            OUT_PATH="$OUTPUT_DIR/$MODEL/$ALGO_CONFIG/ungrouped_vectorized_B$bs"
            run_or_skip "Model=$MODEL | $ALGO_CONFIG | ungrouped vectorized B=$bs" \
            "${BASE_CMD[@]}" \
                --algo-and-network-config "$ALGO_CONFIG" \
                --output-root "$OUT_PATH" \
                --hparam-batch-size "$bs"
        done

        # c. Grouped Vectorized (optional)
        if [[ "$GROUPING_ENABLED" == true ]]; then
          for bs in $BATCH_SIZES; do
              OUT_PATH="$OUTPUT_DIR/$MODEL/$ALGO_CONFIG/grouped_vectorized_B$bs"
              run_or_skip "Model=$MODEL | $ALGO_CONFIG | grouped vectorized B=$bs" \
              "${BASE_CMD[@]}" \
                  --algo-and-network-config "$ALGO_CONFIG" \
                  --output-root "$OUT_PATH" \
                  --hparam-batch-size "$bs" \
                  --group-by-structural-hparams True
          done
        fi
    done
done

# ------------------------------------------------------------------------------
# Summary of skipped configs & runs
# ------------------------------------------------------------------------------
echo -e "\n\n------------------ SKIP SUMMARY ------------------"
if ((${#SKIPPED_CONFIGS[@]})); then
  echo "Configs skipped (missing module):"
  for s in "${SKIPPED_CONFIGS[@]}"; do
    echo "  - $s"
  done
else
  echo "No configs were skipped for missing modules."
fi

if ((${#SKIPPED_RUNS[@]})); then
  echo -e "\nIndividual runs skipped (runtime errors):"
  for s in "${SKIPPED_RUNS[@]}"; do
    echo "  - $s"
  done
else
  echo -e "\nNo individual runs were skipped for runtime errors."
fi
echo "--------------------------------------------------"

echo -e "\n\n================================================="
echo "           Benchmark Script Finished!            "
echo "         Results are in: $OUTPUT_DIR"
echo "================================================="
