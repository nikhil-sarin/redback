import warnings


def cond_jit(func=None, **kwargs):
    """A conditional function wrapper for number jit functions.

    Parameters
    ----------
    func : callable, optional
        The function to wrap. This is automatically passed when
        the decorator has explicit arguments (e.g. no () afterward).
    **kwargs : keyword arguments
        Additional arguments to pass to the JIT compiler.

    Returns
    -------
    function
        The wrapped function or method.
    """
    try:
        # Try to use numba's jit function wrapper directly.
        from numba import jit

        decorator = jit(**kwargs)
    except ImportError:
        warnings.warn("Numba is not installed. Using the non-compiled function.")

        # If numba is not available, fall back to a no-op decorator.
        def no_op_decorator(func):
            return func

        decorator = no_op_decorator

    # We check if the user passed a function to wrap. This is needed
    # because func may be None if the decorator is used without parentheses
    # otherwise the decorator will be applied to the function itself.
    if func is not None:
        return decorator(func)
    return decorator
