import multiprocessing
import functools

def timeout(seconds):
    """
    Sample Usage
        @timeout(3 * 60 * 60)  # 3 hours
        def my_long_function():
            ...
    
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = multiprocessing.Queue()

            def target():
                try:
                    result_queue.put(("ok", func(*args, **kwargs)))
                except Exception as e:
                    result_queue.put(("error", e))

            p = multiprocessing.Process(target=target)
            p.start()
            p.join(timeout=seconds)

            if p.is_alive():
                p.terminate()
                p.join()
                raise TimeoutError(f"Timed out after {seconds}s: {func.__name__}.")

            status, value = result_queue.get()
            if status == "error":
                raise value
            return value
        return wrapper
    return decorator


