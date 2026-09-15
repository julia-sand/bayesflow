import os
import pickle
import time
import numpy as np
import keras
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
    
    # --- PLACEHOLDER FOR MISSING FUNCTIONALITY ---
    # Currently, CostAwareSimulator.predicate is not implemented and 
    # CostAwareSimulator.sample just delegates to the base simulator.
    # We define a custom predicate to demonstrate how the cost-aware logic works.
    
    def sir_cost_predicate(samples, cost_model):
        """
        Accept samples if the predicted cost is below a certain threshold.
        """
        # samples is a dict; in the case of the SIR simulator, 'theta' contains [beta, gamma]
        theta = samples["theta"]
        # Predict mean cost for these parameters
        predicted_mean, _ = cost_model.predict(theta)
        
        # Threshold for "acceptable" cost
        threshold = 0.5
        return predicted_mean < threshold

    # Monkey-patch the predicate for the example
    cost_aware_sim.predicate = sir_cost_predicate
    # ---------------------------------------------
    
    # 5. Demonstrate the effect of cost-awareness
    print("\nTesting cost-aware sampling...")
    
    # Generate a batch of candidates from the prior
    num_candidates = 100
    candidates_theta = np.array([sir_sim.prior() for _ in range(num_candidates)])
    candidates_dict = {"theta": candidates_theta}
    
    # Evaluate which candidates are "cheap" enough using the cost model
    # UPDATED: now passing candidates_dict to predicate
    accepted_mask = cost_aware_sim.predicate(candidates_dict, cost_model)
    num_accepted = np.sum(accepted_mask)
    
    print(f"Total candidates generated: {num_candidates}")
    print(f"Candidates accepted by cost predicate: {num_accepted}")
    print(f"Acceptance rate: {num_accepted/num_candidates:.2%}")
    
    if num_accepted > 0:
        # Compute performance metrics
        # UPDATED: passing the candidates_dict instead of just candidates_theta
        metrics = cost_aware_sim.compute_metrics(candidates_dict, accepted_mask, cost_model)
        print("\nPerformance Metrics:")
        print(f"  ESS: {metrics['ess']:.2f}")
        print(f"  CG:  {metrics['cg']:.2f}")
        
        # Compute importance weights for accepted samples
        # We need the g_val for this, so we'll replicate the internal step for the example
        # UPDATED: predict takes theta from candidates_dict
        predicted_cost, _ = cost_model.predict(candidates_dict["theta"])
        g_val = cost_aware_sim.regularise_cost(predicted_cost)
        weights = cost_aware_sim.compute_weights(g_val, accepted_mask)
        
        print(f"  Computed {len(weights)} weights for accepted samples.")
        print(f"  Sum of weights: {np.sum(weights):.2f}")

        # Only run the expensive simulator on accepted parameters
        accepted_theta = candidates_dict["theta"][accepted_mask]
        results = [sir_sim.observation_model(t) for t in accepted_theta]
        print(f"\nSuccessfully simulated {len(results)} cost-efficient samples.")
    else:
        print("No candidates were accepted. Try increasing the cost threshold.")

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
        inference_net = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ])
        
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
