import multiprocessing
import queue
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

            # Wait on the queue, not on the process. Joining first deadlocks when the
            # result is larger than the OS pipe buffer: the child blocks writing it,
            # so it stays alive and the wait is reported as a timeout even though the
            # work finished.
            try:
                status, value = result_queue.get(timeout=seconds)
            except queue.Empty:
                p.terminate()
                p.join()
                raise TimeoutError(f"Timed out after {seconds}s: {func.__name__}.")

            p.join()   # the child has handed over its result, so this returns promptly
            if status == "error":
                raise value
            return value
        return wrapper
    return decorator


