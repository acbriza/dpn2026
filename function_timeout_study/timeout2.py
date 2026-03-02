import time
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue

def run_division_process(queue, numerator, denominator, sleep_sec):
    try:        
        time.sleep(sleep_sec)
        result = numerator/denominator        
        queue.put(result)
    except Exception as e:
        queue.put(e)

def division_with_timeout(numerator, denominator, timeout_sec):
    q = Queue()
    sleep_sec = random.randint(1, timeout_sec*2)
    print(f"denominator: {denominator}, timeout_sec: {timeout_sec}, sleep_sec:{sleep_sec}" )
    p = Process(
        target=run_division_process,
        args=(q, numerator, denominator, sleep_sec)
    )
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        print("!!!! Timed out !!!!")
        return None  # Timeout occurred

    result = q.get() if not q.empty() else None
    # print("division_with_timeout result", result)
    if isinstance(result, Exception):
        raise result
    return result

def check_division_with_timeout(numerator, denominators, timeout_sec=30*60):
    results = {}
    for d in denominators:
        results[d] = {"division": "fail"}
        try:
            result = division_with_timeout(numerator, d, timeout_sec)
            if result is None:
                print(f"⏰ Timeout (>{timeout_sec} sec) for denominator '{d}' — skipping to next.")
                results[d]["division"] = "timeout"
                continue

            results[d]["division"] = result

        except Exception as e:
            print(f'Error calculating quotient for {d}')
            print(f'{e}')
            results[d]["division"] = "Error"
            
        print('division: ', results[d]["division"])

    print(results)

    return results


if __name__ == '__main__':
    numerator = 20
    denominators = [1,2,3,4,6,7,8,9,0]
    check_division_with_timeout(numerator,denominators,5)
