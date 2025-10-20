from enum import Enum  # ADD THIS
from typing import Optional

import jax


class JAXDebugMode(Enum):  # ADD THIS
    ORIGINAL = 0
    FILTERING = 1
    NOOP = 2


class JAXDebugFilter:
    """Simple JAX debug filter"""

    _instance: Optional["JAXDebugFilter"] = None

    def __init__(self):
        self.enabled_contexts: set[str] = set()
        self._true_original_jax_debug_print = jax.debug.print  # REPLACE original_debug_print
        self._current_mode = JAXDebugMode.ORIGINAL
        jax.debug.print = self._true_original_jax_debug_print

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _noop_debug_print_impl(self, *args, **kwargs):  # ADD THIS
        pass

    def _filtered_debug_print_impl(self, *args, **kwargs):  # ADD THIS
        if args and isinstance(args[0], str):
            msg = str(args[0])
            if any(ctx in msg.upper() for ctx in self.enabled_contexts):
                return self._true_original_jax_debug_print(*args, **kwargs)
        # Otherwise, suppress

    def activate_filtering(self):  # ADD THIS
        if self._current_mode != JAXDebugMode.FILTERING:
            jax.debug.print = self._filtered_debug_print_impl
            self._current_mode = JAXDebugMode.FILTERING

    def deactivate_to_original(self):  # ADD THIS
        if self._current_mode != JAXDebugMode.ORIGINAL:
            jax.debug.print = self._true_original_jax_debug_print
            self._current_mode = JAXDebugMode.ORIGINAL

    def deactivate_to_noop(self):  # ADD THIS
        if self._current_mode != JAXDebugMode.NOOP:
            jax.debug.print = self._noop_debug_print_impl
            self._current_mode = JAXDebugMode.NOOP

    def enable_context(self, context: str):
        """Enable debug prints for specific context"""
        self.enabled_contexts.add(context.upper())
        if self._current_mode != JAXDebugMode.FILTERING:
            self.activate_filtering()

    def disable_context(self, context: str):
        """Disable debug prints for specific context"""
        self.enabled_contexts.discard(context.upper())

    def restore_original(self):  # UPDATE
        self.deactivate_to_original()

    def _patch_jax_debug(self):  # UPDATE
        self.activate_filtering()

    @property
    def is_patched(self):  # ADD THIS
        return self._current_mode == JAXDebugMode.FILTERING

    @property
    def is_noop(self):  # ADD THIS
        return self._current_mode == JAXDebugMode.NOOP


def enable_jax_debug_for_context(context: str):
    """Simple function to enable JAX debug for a context"""
    JAXDebugFilter.get_instance().enable_context(context)
