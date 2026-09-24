from collections.abc import Callable

import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator

class CostAwareSimulator(Simulator):
    """Implements a simulator based on rejection sampling of a cost-aware proxy prior.

    Simulation-based inference can be expensive when the cost of a single simulation
    depends on the parameter value. Cost-aware SBI reduces this cost by sampling the
    parameters from a proxy prior that is biased towards cheaper parameter values. The
    bias is introduced through rejection sampling whose acceptance probability is a
    function of a regularisation of the predicted cost, ``g(c(theta))``, where
    ``c(theta)`` is predicted by a cost interpolation model.
    """

    def __init__(self, prior: Callable[[], np.ndarray], cost_model, *, gmin: float = 0.2):
        """
        Initialize a cost-aware simulator that samples from a provided prior.

        Parameters
        ----------
        prior : Callable[[], np.ndarray]
            A function that returns a single draw from the parameter prior.
        cost_model : Model
            A fitted cost interpolation model used to predict the cost of a parameter
            value. It is use to evaluate the acceptance
            probability. 
        gmin : float, optional
            Minimum value for the regularised cost used in the acceptance probability
            calculation. Default is 1.0.
        """
        self.prior = prior
        self.cost_model = cost_model
        self.gmin = gmin #cost offset 

    @allow_batch_size
    def sample(self, batch_shape: Shape, k: float | np.ndarray | list = 1.0, cost_aware: bool = True, **kwargs) -> dict[str, np.ndarray]:
        """Sample the prior

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number
            of samples, but an int can also be passed.
        k : float, np.ndarray, list, optional
            Power factor for cost regularisation. If a vector is provided, the total
            batch is split equally among the provided k values. Default is 1.0.
        cost_aware : bool, optional
            Whether to use rejection sampling based on cost. Default is True.
        **kwargs
            Additional keyword arguments passed to the base simulator.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary of sampled outputs, including a 'k' array indicating which
            k value was used for each sample.
        """

        if not cost_aware:
            # Return samples from the prior directly
            total_samples = np.prod(batch_shape) if not isinstance(batch_shape, int) else batch_shape
            theta_samples = np.stack([self.prior() for _ in range(total_samples)])
            return {"parameters": theta_samples}

        k_vals = np.atleast_1d(k)
        n_k = len(k_vals)
        
        # Resolve total batch size
        if isinstance(batch_shape, int):
            total_samples = batch_shape
        else:
            total_samples = np.prod(batch_shape)
        
        samples_per_k = total_samples // n_k
        remainder = total_samples % n_k

        all_outputs = {}
        all_ks = []

        print(f"Cost Aware simulator does rejection sampling of the prior for {n_k} different k values")

        for i, k_val in enumerate(k_vals):
            # Distribute remainder to the first few k values
            current_batch_size = samples_per_k + (1 if i < remainder else 0)
            if current_batch_size == 0:
                continue
                
            # Rejection sampling loop for parameters only
            accepted_theta = []
            while len(np.concatenate(accepted_theta)) < current_batch_size if accepted_theta else 0 < current_batch_size:
                current_count = len(np.concatenate(accepted_theta)) if accepted_theta else 0
                needed = current_batch_size - current_count
                
                # Sample a batch of parameters from the prior
                # We use a reasonable batch size to avoid too many loops
                sample_size = max(needed, current_batch_size)
                theta_batch = np.stack([self.prior() for _ in range(sample_size)])
                
                # Evaluate predicate
                mask = self.predicate({"parameters": theta_batch}, k=k_val)
                accepted_theta.append(theta_batch[mask])
            
            # Concatenate and truncate to exact size
            theta_accepted = np.concatenate(accepted_theta)[:current_batch_size]
            res = {"parameters": theta_accepted}
            
            # Assume res is a dict of arrays
            for key, val in res.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(val)
            
            all_ks.append(np.full(current_batch_size, k_val))

        # Reconstruct results dictionary
        final_results = {key: np.concatenate(vals) for key, vals in all_outputs.items()}
        final_results["k"] = np.concatenate(all_ks)

        return final_results
            
    def regularise_cost(self, cost: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Regularise the predicted cost to get the acceptance probability.

        Parameters
        ----------
        cost : np.ndarray
            Predicted cost for each parameter value.
        k : float, optional
            Power factor for cost regularisation. Default is 1.0.

        Returns
        -------
        g_val : np.ndarray
            Regularised cost values.
        """
        return np.maximum(self.gmin, (cost+self.gmin)**k)

    def compute_weights_per_k(self, theta: np.ndarray, k: float) -> np.ndarray:
        """Compute importance weights for samples using a scalar k.

        Parameters
        ----------
        theta : np.ndarray
            The parameter values sampled.
        k : float
            Power factor for cost regularisation.

        Returns
        -------
        weights : np.ndarray
            Importance weights for the samples.
        """
        costs = self.cost_model(theta)

        g_accepted = self.regularise_cost(costs, k=k)

        return g_accepted / np.sum(g_accepted) if len(g_accepted) > 0 else np.array([])

    def compute_weights(self, accepted_samples: dict[str, np.ndarray]) -> np.ndarray:
        """Compute importance weights for the accepted samples.

        Parameters
        ----------
        accepted_samples : dict of str to np.ndarray
            A dictionary containing 'theta' (or 'parameters') and 'k' arrays.

        Returns
        -------
        weights : np.ndarray
            Importance weights for the accepted samples.
        """
        theta = accepted_samples.get("theta")
        if theta is None:
            theta = accepted_samples.get("parameters")
        if theta is None:
            raise KeyError("accepted_samples must contain 'theta' or 'parameters'.")
        
        k_vals = accepted_samples.get("k")
        if k_vals is None:
            raise KeyError("accepted_samples must contain 'k'.")

        distinct_ks = np.unique(k_vals)
        weights = np.zeros_like(k_vals, dtype=float)

        for k in distinct_ks:
            mask = (k_vals == k)
            theta_k = theta[mask]
            weights[mask] = self.compute_weights_per_k(theta_k, k)

        return weights / len(k_vals)

    def compute_metrics(self, theta: np.ndarray, kvec: float | np.ndarray | list = 1.0) -> dict[str, float]:
        """Compute performance metrics for the cost-aware sampling.

        Parameters
        ----------
        theta : np.ndarray
            The parameter values sampled by the cost-aware sampler.
        kvec : float, np.ndarray, list, optional
            Power factor for cost regularisation. If a vector is provided, only the
            first entry is used. Default is 1.0.
         
        Returns
        -------
        metrics : dict of str to float
            A dictionary containing metrics 'ess' (Effective Sample Size)
            and 'cg' (Computational Gain).
        """

        if isinstance(theta, dict):
            theta = theta["theta"]

        k_val = kvec[0] if isinstance(kvec, (list, np.ndarray)) else kvec

        res = self.cost_model(theta)
        predicted_cost = res
        g_val = self.regularise_cost(predicted_cost, k=k_val)
        
        # Effective Sample Size (ESS)
        # ESS = (sum w)^2 / n sum(w^2)
        ess = np.sum(g_val)**2 / (len(g_val)*np.sum(g_val**2)) if len(g_val) > 0 else 0.0

        # Computational Gain (CG)
        # CG = (Average cost of prior samples) / (Average cost of accepted samples)
        # sample new thetas from the original prior 
        
        theta_new = self.sample(batch_shape=g_val.shape, k=k_val, cost_aware=False)
        res_prior = self.cost_model(theta_new.get("parameters"))
        prior_cost = res_prior

        avg_cost_prior = np.mean(prior_cost)

        avg_cost_accepted = np.mean(predicted_cost) 

        cg = avg_cost_prior / avg_cost_accepted
        
        return {"ess": ess, "cg": cg}

    def predicate(self, samples: dict[str, np.ndarray], k: float = 1.0) -> np.ndarray:
        """The cost-aware acceptance predicate.

        Given a batch of samples, this returns a boolean array indicating which
        samples are accepted based on their predicted cost.

        Parameters
        ----------
        samples : dict of str to np.ndarray
            A batch of samples, as returned by :py:meth:`sample`.
        k : float, optional
            Power factor for cost regularisation. Default is 1.0.
        
        Returns
        -------
        accept : np.ndarray
            A boolean array of shape ``(batch_size,)``.
        """
        theta = samples.get("theta")
        if theta is None:
            theta = samples.get("parameters")
        if theta is None:
            raise KeyError("Samples dictionary must contain 'theta' or 'parameters'.")

        res = self.cost_model(theta)
        predicted_cost = res
        
        g_val = self.regularise_cost(predicted_cost, k=k)
        
        # Acceptance probability = gmin / g(cost(theta))
        prob_accept = self.gmin / g_val
        
        # Rejection sampling: accept if random draw U(0, 1) <= prob_accept
        return np.random.random(len(prob_accept)) <= prob_accept
