import asyncio
import functools
import inspect
import time
from typing import Callable, Any, Literal

from upsonic.utils.package.exception import GuardrailValidationError
from upsonic.exceptions import ExecutionTimeoutError

# A type hint for our specific retry modes, can be imported by other modules.
RetryMode = Literal["raise", "return_false"]

def _get_retries_from_call(
    func: Callable[..., Any],
    retries_from_param: str,
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fallback: int,
) -> int:
    if retries_from_param in kwargs:
        val = kwargs[retries_from_param]
        return int(val) if val is not None else fallback
    try:
        sig = inspect.signature(func)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        val = bound.arguments.get(retries_from_param)
        return int(val) if val is not None else fallback
    except (ValueError, TypeError, KeyError):
        return fallback


def retryable(
    retries: int | None = None,
    mode: RetryMode | None = None,
    delay: float = 1.0,
    backoff: float = 2.0,
    retries_from_param: str | None = None,
) -> Callable:
    """
    Decorator that wraps sync and async functions and handles retrying logic.

    When this decorates a method of a class instance, it dynamically resolves its
    retry configuration with the following priority:
    1. Arguments passed directly to the decorator (e.g., @retryable(retries=5)).
    2. When retries_from_param is set: the instance attribute (e.g., self.retry) first, then the method parameter from the call.
    3. Otherwise: attributes on the instance (e.g., self.retry).
    4. The hardcoded default values in this function (e.g., 3).

    Args:
        retries (int | None): The maximum number of attempts. Overrides instance config.
        mode (RetryMode | None): 'raise' or 'return_false'. Overrides instance config.
        delay (float): The initial delay between retries in seconds.
        backoff (float): The factor by which the delay increases after each failure.
        retries_from_param (str | None): If set, use the instance attribute with this name (e.g. self.retry) as retry count, with priority over the method parameter; if the instance has no such attribute, use the method parameter (e.g. do_async(..., retry=2)).

    Returns:
        A decorator that can be applied to a function or method.
    """

    def decorator(func: Callable) -> Callable:
        """The actual decorator that wraps the function."""

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if retries_from_param:
                instance_retry = getattr(self, "retry", None)
                if instance_retry is not None:
                    final_retries = int(instance_retry)
                else:
                    final_retries = _get_retries_from_call(
                        func, retries_from_param, self, args, kwargs,
                        fallback=0,
                    )
            else:
                final_retries = retries if retries is not None else getattr(self, "retry", 0)
            final_mode = mode if mode is not None else getattr(self, "mode", "raise")

            if final_retries < 1:
                raise ValueError("Number of retries must be at least 1.")

            last_known_exception = None
            current_delay = delay

            for attempt in range(1, final_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if isinstance(e, (GuardrailValidationError, ExecutionTimeoutError)):
                        raise e
                    last_known_exception = e
                    if attempt < final_retries:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed (Attempt {attempt}/{final_retries}). Retrying in {current_delay:.2f}s... Error: {e}", "RetryHandler")
                        time.sleep(current_delay)
                        current_delay *= backoff

            # All retries exhausted. Invoke optional sync hook on the
            # wrapped instance to let it flush per-attempt state.
            hook = getattr(self, '_on_retries_exhausted', None)
            if callable(hook):
                try:
                    hook_result = hook()
                    # Sync wrapper: ignore the result if it happens to be a coroutine
                    if inspect.iscoroutine(hook_result):
                        hook_result.close()
                except Exception:
                    pass

            from upsonic.utils.printing import error_log
            error_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed after {final_retries} attempts.", "RetryHandler")
            if final_mode == "raise":
                raise last_known_exception
            elif final_mode == "return_false":
                return False

        @functools.wraps(func)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if retries_from_param:
                instance_retry = getattr(self, "retry", None)
                if instance_retry is not None:
                    final_retries = int(instance_retry)
                else:
                    final_retries = _get_retries_from_call(
                        func, retries_from_param, self, args, kwargs,
                        fallback=3,
                    )
            else:
                final_retries = retries if retries is not None else getattr(self, "retry", 3)
            final_mode = mode if mode is not None else getattr(self, "mode", "raise")

            if final_retries < 1:
                raise ValueError("Number of retries must be at least 1.")

            last_known_exception = None
            current_delay = delay

            for attempt in range(1, final_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if isinstance(e, (GuardrailValidationError, ExecutionTimeoutError)):
                        raise e
                    last_known_exception = e
                    if attempt < final_retries:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed (Attempt {attempt}/{final_retries}). Retrying in {current_delay:.2f}s... Error: {e}", "RetryHandler")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            # All retries exhausted. Invoke optional hook so the wrapped
            # instance can flush per-attempt state (e.g., agent-level usage
            # accumulation that the final failed attempt's finally block
            # skipped because the pipeline marked the task as paused).
            hook = getattr(self, '_on_retries_exhausted', None)
            if callable(hook):
                try:
                    hook_result = hook()
                    if inspect.iscoroutine(hook_result):
                        await hook_result
                except Exception:
                    pass  # Don't let hook errors mask the original exception

            from upsonic.utils.printing import error_log
            error_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed after {final_retries} attempts.", "RetryHandler")
            if final_mode == "raise":
                raise last_known_exception
            else:
                return False
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    if callable(retries):
        func_to_decorate = retries
        retries = None
        return decorator(func_to_decorate)
    
    return decorator