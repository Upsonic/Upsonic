"""
Task decorator system for durable execution with side effects.

This module provides the @task decorator for wrapping operations with side effects,
enabling idempotent execution, caching, and retry policies.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import random
from contextvars import ContextVar
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union
from dataclasses import dataclass

from upsonic.graphv2.cache import BaseCache, CachePolicy


# Context variable to store the current cache instance
_current_cache: ContextVar[Optional[BaseCache]] = ContextVar('current_cache', default=None)


# Type variables
P = TypeVar('P')
T = TypeVar('T')


@dataclass
class RetryPolicy:
    """Configuration for retrying operations.
    
    Attributes:
        initial_interval: Amount of time before first retry (seconds)
        backoff_factor: Multiplier for interval after each retry
        max_interval: Maximum time between retries (seconds)
        max_attempts: Maximum number of attempts including the first
        jitter: Whether to add random jitter to intervals
        retry_on: Exception types to retry on, or callable to determine if should retry
    """
    
    initial_interval: float = 0.5
    """Amount of time that must elapse before the first retry occurs. In seconds."""
    
    backoff_factor: float = 2.0
    """Multiplier by which the interval increases after each retry."""
    
    max_interval: float = 128.0
    """Maximum amount of time that may elapse between retries. In seconds."""
    
    max_attempts: int = 3
    """Maximum number of attempts to make before giving up, including the first."""
    
    jitter: bool = True
    """Whether to add random jitter to the interval between retries."""
    
    retry_on: Union[type[Exception], Sequence[type[Exception]], Callable[[Exception], bool]] = Exception
    """List of exception classes that should trigger a retry, or a callable that returns True for exceptions that should trigger a retry."""


def default_retry_on(exception: Exception) -> bool:
    """Default retry logic - retries on most exceptions except common programming errors.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if should retry, False otherwise
    """
    # Don't retry on these exception types
    non_retryable = (
        ValueError,
        TypeError,
        ArithmeticError,
        ImportError,
        LookupError,
        NameError,
        SyntaxError,
        RuntimeError,
        ReferenceError,
        StopIteration,
        StopAsyncIteration,
        OSError,
    )
    
    # Check if it's a non-retryable exception
    if isinstance(exception, non_retryable):
        return False
    
    # Check for HTTP exceptions (requests, httpx)
    # Only retry on 5xx errors
    if hasattr(exception, 'response'):
        response = getattr(exception, 'response')
        if hasattr(response, 'status_code'):
            status_code = response.status_code
            # Don't retry 4xx errors (client errors)
            if 400 <= status_code < 500:
                return False
    
    return True


def should_retry(exception: Exception, retry_policy: RetryPolicy) -> bool:
    """Determine if an exception should trigger a retry.
    
    Args:
        exception: The exception that occurred
        retry_policy: The retry policy to check against
        
    Returns:
        True if should retry, False otherwise
    """
    retry_on = retry_policy.retry_on
    
    # If it's a sequence of exception types
    if isinstance(retry_on, (list, tuple)):
        return isinstance(exception, tuple(retry_on))
    
    # If it's a single exception type (check using inspect.isclass to avoid calling it)
    import inspect
    if inspect.isclass(retry_on) and issubclass(retry_on, BaseException):
        return isinstance(exception, retry_on)
    
    # If it's a callable function (not a class), use it
    if callable(retry_on):
        return retry_on(exception)
    
    # Default: don't retry
    return False


def calculate_retry_delay(
    attempt: int,
    retry_policy: RetryPolicy
) -> float:
    """Calculate the delay before the next retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        retry_policy: The retry policy to use
        
    Returns:
        Delay in seconds
    """
    # Calculate base delay with exponential backoff
    delay = retry_policy.initial_interval * (retry_policy.backoff_factor ** attempt)
    
    # Cap at max_interval
    delay = min(delay, retry_policy.max_interval)
    
    # Add jitter if enabled
    if retry_policy.jitter:
        delay = delay * (0.5 + random.random())
    
    return delay


class TaskResult(Generic[T]):
    """Result of a task execution.
    
    This acts as a future-like object that can be used to get the result.
    """
    
    def __init__(self, value: T):
        """Initialize with the result value."""
        self._value = value
        self._is_async = isinstance(value, asyncio.Future) or inspect.iscoroutine(value)
    
    def result(self) -> T:
        """Get the task result (blocks if async).
        
        Returns:
            The task result
        """
        if self._is_async:
            # If it's a coroutine, we need to run it
            if inspect.iscoroutine(self._value):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, return the coroutine
                    return self._value
                else:
                    return loop.run_until_complete(self._value)
            return self._value
        return self._value
    
    async def aresult(self) -> T:
        """Get the task result asynchronously.
        
        Returns:
            The task result
        """
        if inspect.iscoroutine(self._value):
            return await self._value
        elif isinstance(self._value, asyncio.Future):
            return await self._value
        return self._value


class TaskFunction(Generic[T]):
    """A wrapped task function with retry and cache support."""
    
    def __init__(
        self,
        func: Callable[..., T],
        name: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None,
    ):
        """Initialize the task function.
        
        Args:
            func: The function to wrap
            name: Optional name for the task
            retry_policy: Optional retry policy
            cache_policy: Optional cache policy
        """
        self.func = func
        self.name = name or func.__name__
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self._is_async = inspect.iscoroutinefunction(func)
        
        # Copy function metadata
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs) -> TaskResult[T]:
        """Execute the task.
        
        Returns:
            TaskResult wrapping the result
        """
        if self._is_async:
            result = self._execute_async(*args, **kwargs)
        else:
            result = self._execute_sync(*args, **kwargs)
        
        return TaskResult(result)
    
    def _execute_sync(self, *args, **kwargs) -> T:
        """Execute synchronous task with retry logic.
        
        Returns:
            Task result
        """
        # Check cache if policy is set
        if self.cache_policy:
            cache_key = self._get_cache_key(args, kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute with retry logic
        last_exception = None
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1
        
        for attempt in range(max_attempts):
            try:
                result = self.func(*args, **kwargs)
                
                # Store in cache if policy is set
                if self.cache_policy:
                    self._put_in_cache(cache_key, result)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if self.retry_policy and attempt < max_attempts - 1:
                    if should_retry(e, self.retry_policy):
                        delay = calculate_retry_delay(attempt, self.retry_policy)
                        time.sleep(delay)
                        continue
                
                # No retry or max attempts reached
                raise
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
    
    async def _execute_async(self, *args, **kwargs) -> T:
        """Execute asynchronous task with retry logic.
        
        Returns:
            Task result
        """
        # Check cache if policy is set
        if self.cache_policy:
            cache_key = self._get_cache_key(args, kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute with retry logic
        last_exception = None
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else 1
        
        for attempt in range(max_attempts):
            try:
                result = await self.func(*args, **kwargs)
                
                # Store in cache if policy is set
                if self.cache_policy:
                    self._put_in_cache(cache_key, result)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if self.retry_policy and attempt < max_attempts - 1:
                    if should_retry(e, self.retry_policy):
                        delay = calculate_retry_delay(attempt, self.retry_policy)
                        await asyncio.sleep(delay)
                        continue
                
                # No retry or max attempts reached
                raise
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _get_cache_key(self, args, kwargs) -> str:
        """Generate cache key from arguments.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Combine args and kwargs into a single dict-like structure
        cache_input = {
            'args': args,
            'kwargs': kwargs,
        }
        return self.cache_policy.key_func(cache_input)
    
    def _get_from_cache(self, key: str) -> Optional[T]:
        """Get result from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None
        """
        # Get cache from context variable
        cache = _current_cache.get()
        if cache is None:
            return None
        
        # Use task name as namespace
        namespace = (f"task:{self.name}",)
        return cache.get(namespace, key)
    
    def _put_in_cache(self, key: str, value: T) -> None:
        """Store result in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Get cache from context variable
        cache = _current_cache.get()
        if cache is None:
            return
        
        # Use task name as namespace
        namespace = (f"task:{self.name}",)
        ttl = self.cache_policy.ttl if self.cache_policy else None
        cache.put(namespace, key, value, ttl=ttl)


def task(
    func: Optional[Callable[..., T]] = None,
    *,
    name: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    cache_policy: Optional[CachePolicy] = None,
) -> Union[TaskFunction[T], Callable[[Callable[..., T]], TaskFunction[T]]]:
    """Decorator to create a durable task.
    
    Tasks are units of work that Upsonic tracks individually. They support:
    - Automatic retry on failures
    - Result caching to avoid re-execution
    - Idempotent execution for durability
    
    Args:
        func: The function to decorate (when used without parentheses)
        name: Optional name for the task
        retry_policy: Optional retry policy for failures
        cache_policy: Optional cache policy for results
        
    Returns:
        Decorated task function
        
    Example:
        ```python
        @task(retry_policy=RetryPolicy(max_attempts=3))
        def call_api(user_id: str) -> dict:
            response = requests.post("https://api.example.com/process", 
                                    json={"user": user_id})
            return response.json()
        
        @task(cache_policy=CachePolicy(ttl=120))
        def expensive_computation(data: list) -> float:
            return sum(data) / len(data)
        ```
    """
    def decorator(f: Callable[..., T]) -> TaskFunction[T]:
        return TaskFunction(
            func=f,
            name=name,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
        )
    
    if func is not None:
        # Called without parentheses: @task
        return decorator(func)
    
    # Called with parentheses: @task(...)
    return decorator


def set_cache_context(cache: Optional[BaseCache]):
    """Set the cache instance for task functions.
    
    This is used internally by the graph execution engine to make the cache
    available to @task decorated functions. You typically don't need to call
    this directly unless you're implementing custom execution logic.
    
    Args:
        cache: The cache instance to use, or None to clear
        
    Returns:
        A token that can be used to reset the context
    """
    return _current_cache.set(cache)


def get_cache_context() -> Optional[BaseCache]:
    """Get the current cache instance from context.
    
    Returns:
        The current cache instance, or None if not set
    """
    return _current_cache.get()

