import numpy as np
import random
# Importing the model from your local file
from model_2026_04_26 import ProblemSolvingModel

def run_single_simulation(params):
    """
    Worker function to run one simulation instance.
    params: (kb_fraction, seed, steps_to_run, local_obs_fraction, type_network, connect_prob, min_deg, clause_interval, run_mode)
    """
    # Handling both 2-tuple and 3-tuple for backward compatibility
    kb_f, seed = params[0], params[1]
    steps_to_run = params[2] if len(params) > 2 else 1000
    local_obs_f = params[3] if len(params) > 3 else 1.0
    type_network = params[4] if len(params) > 4 else "Random"
    connect_prob = params[5] if len(params) > 5 else 0.03
    min_deg = params[6] if len(params) > 6 else 3
    clause_interval = params[7] if len(params) > 7 else 10
    run_mode = params[8] if len(params) > 8 else "basic"


    try:
        model = ProblemSolvingModel(
            N=100,
            K=50,
            alpha=2.0,
            obs_prob=0.01,
            clause_interval=clause_interval, # Keep landscape static for this sweep
            R=steps_to_run,
            setup_source="generate",
            type_network=type_network,
            connect_prob=connect_prob,
            min_deg=min_deg,
            run_mode=run_mode,
            seed=seed,
            kb_fraction=kb_f,
            local_obs_fraction=local_obs_f
        )
        
        results = []
        for _ in range(steps_to_run):
            model.step()
            results.append({
                'step': model.steps,
                'run_mode': run_mode,
                'kb_fraction': kb_f,
                'network': type_network,
                'local_obs_fraction': local_obs_f,
                'seed': seed,
                'avg_violations': float(model.avg_true_V),
                'homogeneity': float(model.homogeneity)
            })
        return results
    except Exception as e:
        print(f"Error in simulation (KB={kb_f}, Seed={seed}): {e}")
        return []
