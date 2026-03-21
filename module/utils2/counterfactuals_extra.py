"""
    Contains functions for running counterfactuals with timeout
"""

import time
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue

def run_dice_in_process(queue, dice_exp, instance, total_CFs, desired_class,
                        features_to_vary, permitted_range, maxiterations):
    """Run DiCE in a separate process so it can be timed out."""
    try:
        cf_suf = dice_exp.generate_counterfactuals(
                        instance, total_CFs=total_CFs, desired_class=desired_class, features_to_vary=features_to_vary, 
                        permitted_range=permitted_range, maxiterations=maxiterations,
                    )
        queue.put(cf_suf)
    except Exception as e:
        queue.put(e)

def generate_cfs_with_timeout(timeout_sec, dice_exp, instance, total_CFs, desired_class,
                          features_to_vary, permitted_range, maxiterations):
    """Run generate_counterfactuals() with timeout control."""
    q = Queue()
    p = Process(
        target=run_dice_in_process,
        args=(q, dice_exp, instance, total_CFs, desired_class,
              features_to_vary, permitted_range, maxiterations)
    )
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # Timeout occurred

    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        raise result
    return result

def check_sufficiency_with_timeout(dice_exp, instance, all_features, permitted_range, 
                      desired_class="opposite", 
                      maxiterations=500, timeout_sec=30*60):
    """Determine sufficiency of each feature for one instance."""

    results = {}
    for f in tqdm(all_features):
        results[f] = {"sufficient": "False"}
        print(f)

        try:
            cf_suf = generate_cfs_with_timeout(
                timeout_sec, dice_exp, instance, total_CFs=1, desired_class=desired_class, features_to_vary=[f], 
                permitted_range=permitted_range, maxiterations=maxiterations,
            )

            # timeout
            if cf_suf is None:
                print(f"⏰ Timeout (>{timeout_sec//60} min) for feature '{f}' — skipping to next.")
                results[f]["sufficient"] = "Timeout"
                continue

            if len(cf_suf.cf_examples_list[0].final_cfs_df) > 0:
                results[f]["sufficient"] = "True"

        except Exception as e:
            print(f'Error calculating sufficiency for {f}')
            print(f'{e}')
            results[f]["sufficient"] = "Error"
            
        print('sufficient: ', results[f]["sufficient"])

    return pd.DataFrame(results).T.reset_index(names="feature")

