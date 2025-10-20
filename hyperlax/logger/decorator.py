import functools
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Try to import nvitop, but make it optional
try:
    from nvitop import Device as NvitopDevice

    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    NvitopDevice = None  # Placeholder


def log_timing(name: str = ""):
    """Simple timing decorator"""

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(f"{func_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"{func_name} failed after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def log_gpu_memory(name: str = ""):
    """Logs both nvitop and JAX internal GPU memory stats (if available) before and after the function call."""

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- NVITOP ---
            nvitop_mem_before = {}
            nvitop_mem_after = {}
            nvitop_ok = False
            nvitop_error = None
            if NVITOP_AVAILABLE:
                try:
                    for device in NvitopDevice.all():
                        nvitop_mem_before[device.index] = device.memory_used()
                    nvitop_ok = True
                except Exception as e:
                    nvitop_error = e
                    logger.debug(
                        f"MEMLOG ({func_name}) (nvitop): Error getting initial memory: {e}"
                    )

            # # --- JAX Internal ---
            # jax_mem_before = {}
            # jax_mem_after = {}
            # jax_ok = False
            # jax_error = None
            # try:
            #     gpu_devices = [d for d in jax.devices() if d.platform.upper() == 'GPU']
            #     for device in gpu_devices:
            #         try:
            #             stats = device.memory_stats()
            #             jax_mem_before[device.id] = stats.get('bytes_in_use', 0)
            #         except Exception as e_stats:
            #             logger.debug(f"MEMLOG ({func_name}) (JAX) GPU{device.id}: Error getting initial memory_stats: {e_stats}")
            #             jax_mem_before[device.id] = 0
            #     jax_ok = True
            # except Exception as e:
            #     jax_error = e
            #     logger.debug(f"MEMLOG ({func_name}) (JAX): Error getting device list: {e}")

            # --- Run the function ---
            result = func(*args, **kwargs)

            # --- NVITOP after ---
            if NVITOP_AVAILABLE and nvitop_ok:
                try:
                    for device in NvitopDevice.all():
                        nvitop_mem_after[device.index] = device.memory_used()
                except Exception as e:
                    logger.debug(f"MEMLOG ({func_name}) (nvitop): Error getting final memory: {e}")
                    nvitop_mem_after = nvitop_mem_before.copy()  # fallback

                for dev_idx in nvitop_mem_before:
                    before = nvitop_mem_before[dev_idx]
                    after = nvitop_mem_after.get(dev_idx, before)
                    delta = after - before
                    delta_mb = delta / (1024**2)
                    after_mb = after / (1024**2)
                    logger.info(
                        f"MEMLOG ({func_name}) GPU{dev_idx} (nvitop): {delta_mb:+.2f}MB. Total after: {after_mb:.2f}MB"
                    )
            elif NVITOP_AVAILABLE and nvitop_error:
                logger.debug(
                    f"MEMLOG ({func_name}) (nvitop): Skipped final logging due to earlier error: {nvitop_error}"
                )

            ## NOTE disabling since only works properly when platform env var is not set but we do use it often
            ## so not useful atm
            # # --- JAX Internal after ---
            # if jax_ok:
            #     try:
            #         gpu_devices = [d for d in jax.devices() if d.platform.upper() == 'GPU']
            #         for device in gpu_devices:
            #             try:
            #                 stats = device.memory_stats()
            #                 jax_mem_after[device.id] = stats.get('bytes_in_use', 0)
            #             except Exception as e_stats:
            #                 logger.debug(f"MEMLOG ({func_name}) (JAX) GPU{device.id}: Error getting final memory_stats: {e_stats}")
            #                 jax_mem_after[device.id] = jax_mem_before.get(device.id, 0)
            #     except Exception as e:
            #         logger.debug(f"MEMLOG ({func_name}) (JAX): Error getting device list after: {e}")
            #         jax_mem_after = jax_mem_before.copy()  # fallback

            #     for dev_id in jax_mem_before:
            #         before = jax_mem_before[dev_id]
            #         after = jax_mem_after.get(dev_id, before)
            #         delta = after - before
            #         delta_mb = delta / (1024 ** 2)
            #         after_mb = after / (1024 ** 2)
            #         logger.info(f"MEMLOG ({func_name}) JAX GPU Device {dev_id}: {delta_mb:+.2f}MB. Total after: {after_mb:.2f}MB")
            # elif jax_error:
            #     logger.debug(f"MEMLOG ({func_name}) (JAX): Skipped final logging due to earlier error: {jax_error}")

            return result

        return wrapper

    return decorator


def log_array_shapes(name: str = ""):
    """Log JAX array shapes for debugging"""

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log input shapes
            def log_shapes(obj, prefix=""):
                if hasattr(obj, "shape"):
                    logger.debug(f"{func_name} {prefix}shape: {obj.shape}")
                elif isinstance(obj, (tuple, list)):
                    for i, item in enumerate(obj):
                        log_shapes(item, f"{prefix}[{i}].")
                elif isinstance(obj, dict):
                    for key, item in obj.items():
                        log_shapes(item, f"{prefix}{key}.")

            if args:
                log_shapes(args, "input_")

            result = func(*args, **kwargs)
            log_shapes(result, "output_")
            return result

        return wrapper

    return decorator
