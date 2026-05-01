"""
Quick test script to verify model.py functionality.
Generates density-matched Random and Scale-Free networks and runs a short simulation.
"""

import numpy as np
from model_2026_04_21 import ProblemSolvingModel

def run_test():
    # Configuration
    N = 60
    K = 30
    ALPHA = 2
    STEPS = 10
    MIN_DEG = 2
    
    # Match edge density:
    # Scale-Free edges ≈ 2 * (N - min_deg)
    # Random edges = p * N * (N - 1)
    # So p ≈ (2 * (N - min_deg)) / (N * (N - 1))
    connect_prob = round(2 * (N - MIN_DEG) / (N * (N - 1)), 4)

    configs = [
        ("Random", {"connect_prob": connect_prob}),
        ("Scale Free", {"min_deg": MIN_DEG})
    ]

    print(f"=== Starting Quick Test (N={N}, K={K}, Alpha={ALPHA}) ===")
    print(f"Density Matching: Random p={connect_prob} matches SF min_deg={MIN_DEG}\n")

    for label, net_kwargs in configs:
        print(f"--- Testing {label} Network ---")
        
        # Initialize model
        model = ProblemSolvingModel(
            N=N,
            K=K,
            alpha=ALPHA,
            setup_source="generate",
            type_network=label,
            seed=42,
            **net_kwargs
        )

        print(f"Initial Performance: Avg Violations = {model.avg_true_V:.2f}, Homogeneity = {model.homogeneity:.3f}")

        for i in range(1, STEPS + 1):
            model.step()
            if i % 2 == 0:
                print(f"Step {i:2}: Avg Violations = {model.avg_true_V:.2f}, Homogeneity = {model.homogeneity:.3f}")
        print(f"✅ {label} test complete.\n")

if __name__ == "__main__":
    run_test()