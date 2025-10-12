import asyncio
import functools
import inspect
import time
from typing import Callable, Any, Literal

from upsonic.utils.package.exception import GuardrailValidationError

# A type hint for our specific retry modes, can be imported by other modules.
RetryMode = Literal["raise", "return_false"]

def retryable(
    retries: int | None = None,
    mode: RetryMode | None = None,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """
    Decorator that wraps sync and async functions and handles retrying logic.
    
    When this decorates a method of a class instance, it dynamically resolves its
    retry configuration with the following priority:
    1. Arguments passed directly to the decorator (e.g., @retryable(retries=5)).
    2. Attributes found on the instance the method belongs to (e.g., self.retry).
    3. The hardcoded default values in this function (e.g., 3).

    Args:
        retries (int | None): The maximum number of attempts. Overrides instance config.
        mode (RetryMode | None): 'raise' or 'return_false'. Overrides instance config.
        delay (float): The initial delay between retries in seconds.
        backoff (float): The factor by which the delay increases after each failure.

    Returns:
        A decorator that can be applied to a function or method.
    """
    
    def decorator(func: Callable) -> Callable:
        """The actual decorator that wraps the function."""

        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            final_retries = retries if retries is not None else getattr(self, 'retry', 0)
            final_mode = mode if mode is not None else getattr(self, 'mode', 'raise')

            if final_retries < 1:
                raise ValueError("Number of retries must be at least 1.")

            last_known_exception = None
            current_delay = delay

            for attempt in range(1, final_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if isinstance(e, GuardrailValidationError):
                        raise e
                    last_known_exception = e
                    if attempt < final_retries:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed (Attempt {attempt}/{final_retries}). Retrying in {current_delay:.2f}s... Error: {e}", "RetryHandler")
                        time.sleep(current_delay)
                        current_delay *= backoff

            from upsonic.utils.printing import error_log
            error_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed after {final_retries} attempts.", "RetryHandler")
            if final_mode == "raise":
                raise last_known_exception
            elif final_mode == "return_false":
                return False

        @functools.wraps(func)
        async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any:
            final_retries = retries if retries is not None else getattr(self, 'retry', 3)
            final_mode = mode if mode is not None else getattr(self, 'mode', 'raise')

            if final_retries < 1:
                raise ValueError("Number of retries must be at least 1.")

            last_known_exception = None
            current_delay = delay

            for attempt in range(1, final_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if isinstance(e, GuardrailValidationError):
                        raise e
                    last_known_exception = e
                    if attempt < final_retries:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Call to '{self.__class__.__name__}.{func.__name__}' failed (Attempt {attempt}/{final_retries}). Retrying in {current_delay:.2f}s... Error: {e}", "RetryHandler")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

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