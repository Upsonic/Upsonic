import functools



def wrapper(func):
    """A decorator that logs the start and end of the function call."""
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        from .gpt_computer_assistant import the_main_window

        print("GOOGLE-searching")
        function_name = "Tool: " + func.__name__
        the_main_window.active_border_animation(function_name)
        result = func(*args, **kwargs)
        the_main_window.deactive_border_animation(function_name)
        print("GOOGLE SEARCHİNG COMPLEATES")
        
        return result
    return wrapped_func
