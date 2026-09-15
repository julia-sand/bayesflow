import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from bayesflow.simulators.benchmark_simulators.sir import SIR
from bayesflow.simulators.cost import CostAwareSimulator, CostInterpModel
from bayesflow.approximators import RatioApproximator, ContinuousApproximator
from bayesflow.adapters import Adapter

def main():
    # 1. Initialize the base SIR simulator
    # This simulator samples beta (contact rate) and gamma (recovery rate)
    sir_sim = SIR()
    
    # 2. Generate synthetic cost data
    # In a real scenario, 'cost' would be the actual wall-clock time of the simulation.
    # For this example, we simulate that cost increases with the contact rate (beta).
    cost_data_dir = "cost_data_sir"
    os.makedirs(cost_data_dir, exist_ok=True)
    
    print("Generating synthetic cost data...")
    num_cost_samples = 20
    for i in range(num_cost_samples):
        # Use the simulator's own prior to get realistic parameter values
        theta = sir_sim.prior() 
        
        # Simulate the cost (e.g., higher beta -> more complex dynamics -> slightly more time)
        # We add some noise to make it a realistic regression problem for the GP
        actual_cost = 0.1 + 2.0 * theta[0] + np.random.normal(0, 0.05)
        
        with open(f"{cost_data_dir}/sample_{i}.pkl", "wb") as f:
            pickle.dump({"theta": theta, "cost": actual_cost}, f)
    
    # 3. Fit the Cost Interpolation Model
    # This model uses a GP to predict cost given theta
    print("Fitting cost interpolation model...")
    cost_model = CostInterpModel(root=cost_data_dir).fit(length_scale=1.0)
    
    # 4. Initialize the CostAwareSimulator
    # It wraps the SIR simulator and uses the fitted cost model
    cost_aware_sim = CostAwareSimulator(simulator=sir_sim, cost_model=cost_model)
        
    # 5. Demonstrate the effect of cost-awareness
    print("\nTesting cost-aware sampling...")
    
    # Use rejection_sample from the base Simulator class via CostAwareSimulator
    num_samples = 20
    
    # We create a lambda to pass the cost_model to the predicate
    #cost_predicate = lambda samples: cost_aware_sim.predicate(samples)
    
    # This will keep sampling until we have exactly num_samples accepted
    accepted_samples = cost_aware_sim.rejection_sample(
        batch_shape=(num_samples,),
        predicate=cost_aware_sim.predicate#cost_predicate
    )
    
    # The base SIR simulator returns "parameters" and "observables"
    accepted_theta = accepted_samples.get("theta")
    if accepted_theta is None:
        accepted_theta = accepted_samples.get("parameters")

    print(f"Successfully sampled {len(accepted_theta)} cost-efficient parameters.")

    # To compute metrics, we need to see how many total candidates were generated.
    # Since rejection_sample abstracts this, for the sake of the example's metrics,
    # we'll perform a manual batch check to demonstrate compute_metrics.
    num_candidates = 100
    candidates_theta = np.array([sir_sim.prior() for _ in range(num_candidates)])
    candidates_dict = {"theta": candidates_theta}
    accepted_mask = cost_aware_sim.predicate(candidates_dict, cost_model)
    num_accepted = np.sum(accepted_mask)
    
    metrics = cost_aware_sim.compute_metrics(candidates_dict, accepted_mask, cost_model)
    print("\nPerformance Metrics (based on a test batch of 100):")
    print(f"  ESS: {metrics['ess']:.2f}")
    print(f"  CG:  {metrics['cg']:.2f}")

    # Only run the expensive simulator on accepted parameters
    results = [sir_sim.observation_model(t) for t in accepted_theta]
    print(f"\nSuccessfully simulated {len(results)} samples using the expensive simulator.")

    # 6. Example: Train a model (Approximator) on the cost-efficient samples
    print("\nTraining an example model on cost-efficient samples...")
    if num_accepted > 0:
        # We need a dataset for training. Let's create a simple one.
        # In a real scenario, you'd use a BayesFlow Dataset object.
        # For this example, we'll just simulate a few more to have a training set.
        
        # Generate training data using the cost-aware simulator's logic
        train_num_candidates = 500
        train_candidates_theta = np.array([sir_sim.prior() for _ in range(train_num_candidates)])
        train_candidates_dict = {"theta": train_candidates_theta}
        
        train_accepted_mask = cost_aware_sim.predicate(train_candidates_dict, cost_model)
        train_accepted_theta = train_candidates_theta[train_accepted_mask]
        
        # Simulate observations for accepted theta
        train_observations = np.array([sir_sim.observation_model(t) for t in train_accepted_theta])
        
        # Prepare data for ContinuousApproximator
        # It expects a dictionary with "inference_variables" and "inference_conditions"
        
        # Compute importance weights for the training set to correct sampling bias
        train_predicted_cost, _ = cost_model.predict(train_candidates_theta)
        train_g_val = cost_aware_sim.regularise_cost(train_predicted_cost)
        train_weights = cost_aware_sim.compute_weights(train_g_val, train_accepted_mask)
        
        # We use an adapter to prepare the data. 
        # We include 'weights' in the data so the adapter can pass them through
        # or a custom transform can handle them if needed.
        adapter = Adapter().create_default(inference_variables=["theta"])
        
        # The training data needs to be in a dictionary format for the adapter
        train_data = {
            "theta": train_accepted_theta,
            "observations": train_observations,
            "weights": train_weights,
        }
        
        # Apply adapter
        transformed_data = adapter(train_data)
        
        # Define a simple MLP for the inference network
        inference_net = nn.Sequential(
            nn.Linear(train_observations.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Initialize ContinuousApproximator
        # Pass the adapter to the approximator
        approximator = ContinuousApproximator(inference_network=inference_net, adapter=adapter)
        
        # We need to build the approximator with the correct shapes
        data_shapes = {
            "inference_variables": train_accepted_theta.shape[1:],
            "inference_conditions": train_observations.shape[1:],
        }
        approximator.build(data_shapes)
        
        # In a real workflow, you would use:
        # approximator.fit(dataset=train_dataset, sample_weights=train_weights)
        print(f"Model built successfully. Training on {len(train_accepted_theta)} samples with importance weights.")
        print("The workflow is now complete: Cost-aware sampling -> Model training.")
    else:
        print("Skipping model training as no samples were accepted.")

if __name__ == "__main__":
    main()
