from collections.abc import Callable

import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator

class CostAwareProposal(Simulator):
    """Samples a prior distribution based on a cost function.

    Simulation-based inference (SBI) can be expensive in terms of computational time. Its cost can often depend on the parameter values used in the simulation.
    Cost-aware SBI reduces the cost of simulating data by sampling the
    parameters from a proxy prior biased towards parameter values that have a lower simulation cost. The
    bias is introduced through rejection sampling whose acceptance probability is a
    function of a regularization of the cost, ``g(c(theta))``, where
    ``c(theta)`` is predicted by a cost interpolation model.
    """

    def __init__(self, prior: Callable[[], np.ndarray], cost_model, *, gmin: float = 0.2, max_attempts: int = 1000):
        """
        Initialize a cost-aware simulator that samples from a proxy prior that favours lower cost parameters.

        Parameters
        ----------
        prior : Callable[[], np.ndarray]
            A function that returns a single draw from the parameter prior.
        cost_model : Model
            A fitted cost interpolation model used to predict the cost of a parameter
            value. It is used to evaluate the acceptance
            probability.
        gmin : float, optional
            Minimum value for the regularized cost used in the acceptance probability
            calculation. Default is 0.2.
        max_attempts : int, optional
            Maximum number of sampling batches to attempt before stopping and
            returning whatever has been collected. Default is 1000.
        """
        self.prior = prior
        self.cost_model = cost_model
        self.gmin = gmin #cost offset
        self.max_attempts = max_attempts

    def _extract_samples(self, samples: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Extract theta and k values from the samples dictionary.

        Parameters
        ----------
        samples : dict of str to np.ndarray
            The samples dictionary.

        Returns
        -------
        theta : np.ndarray
            The parameter values.
        k_vals : np.ndarray
            The k values.
        """
        try:
            theta = samples["parameters"]
        except KeyError:
            raise KeyError("Samples dictionary must contain 'parameters'.")

        try:
            k_vals = samples["k"]
        except KeyError:
            raise KeyError("Samples dictionary must contain 'k'.")

        return np.asarray(theta), np.asarray(k_vals)


    @allow_batch_size
    def sample(self, batch_shape: Shape, k: float | np.ndarray | list = 1.0, cost_aware: bool = True, **kwargs) -> dict[str, np.ndarray]:
        """Sample the prior

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number
            of samples, but an int can also be passed.
        k : float, np.ndarray, list, optional
            Power factor for cost regularization. If a vector is provided, the total
            batch is split equally among the provided k values. Default is 1.0.
        cost_aware : bool, optional
            Whether to use rejection sampling based on cost. Default is True.
        **kwargs
            Additional keyword arguments passed to the base simulator.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary of sampled outputs. This typically includes:
            - 'parameters': An array of shape ``(total_samples, *prior_shape)``.
            - 'k': An array of shape ``(total_samples,)`` indicating which
              k value was used for each sample.
        """

        if not cost_aware:
            # Return samples from the prior directly
            total_samples = np.prod(batch_shape) if not isinstance(batch_shape, int) else batch_shape
            theta_samples = np.stack([self.prior() for _ in range(total_samples)])
            
            # Ensure k is provided in the output for consistency with _extract_samples
            # Use a default k=1.0 if the provided k is not a scalar
            k_val = k if np.isscalar(k) else 1.0
            return {"parameters": theta_samples, "k": np.full(total_samples, k_val)}
        if isinstance(k, (list, np.ndarray)) and len(k) == 0:
            raise ValueError("The 'k' argument cannot be an empty list or array.")


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
            current_count = 0
            attempts = 0
            while current_count < current_batch_size and attempts < self.max_attempts:
                attempts += 1
                needed = current_batch_size - current_count

                # Sample a batch of parameters from the prior
                # We use a reasonable batch size to avoid too many loops
                sample_size = max(needed, current_batch_size)
                theta_batch = np.stack([self.prior() for _ in range(sample_size)])

                # Evaluate predicate
                mask = self.predicate({"parameters": theta_batch}, k=k_val)
                accepted_batch = theta_batch[mask]
                
                if len(accepted_batch) > 0:
                    accepted_theta.append(accepted_batch)
                    current_count += len(accepted_batch)

            if current_count < current_batch_size:
                print(f"Warning: Rejection sampling for k={k_val} reached max_attempts ({self.max_attempts}) "
                      f"without filling the batch. Collected {current_count} "
                      f"out of {current_batch_size} samples.")

            # Concatenate and truncate to exact size
            if accepted_theta:
                theta_accepted = np.concatenate(accepted_theta)[:current_batch_size]
            else:
                print(f"Warning: No samples were accepted. Try adjusting gmin or k.")
                theta_accepted = np.array([]).reshape(0, *self.prior().shape)

            res = {"parameters": theta_accepted}

            for key, val in res.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(val)

            # Use the actual number of accepted samples for k values to avoid length mismatch
            all_ks.append(np.full(len(theta_accepted), k_val))

        # Reconstruct results dictionary
        final_results = {key: np.concatenate(vals) for key, vals in all_outputs.items()}
        final_results["k"] = np.concatenate(all_ks)

        return final_results

    def regularize_cost(self, cost: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Regularize the predicted cost to get the acceptance probability.

        Parameters
        ----------
        cost : np.ndarray
            Predicted cost for each parameter value, with shape ``(batch_size,)``.
        k : float, optional
            Power factor for cost regularization. Default is 1.0.

        Returns
        -------
        g_val : np.ndarray
            Regularized cost values with shape ``(batch_size,)``.
        """
        return np.maximum(self.gmin, (cost+self.gmin)**k)

    def compute_weights_per_k(self, theta: np.ndarray, k: float) -> np.ndarray:
        """Compute importance weights for samples using a scalar k.

        Parameters
        ----------
        theta : np.ndarray
            The parameter values sampled, with shape ``(batch_size, *prior_shape)``.
        k : float
            Power factor for cost regularization.

        Returns
        -------
        weights : np.ndarray
            Importance weights for the samples, with shape ``(batch_size,)``.
        """
        costs = self.cost_model(theta)

        g_accepted = self.regularize_cost(costs, k=k)

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
        theta, k_vals = self._extract_samples(accepted_samples)

        distinct_ks = np.unique(k_vals)
        weights = np.zeros_like(k_vals, dtype=float)

        for k in distinct_ks:
            mask = (k_vals == k)
            theta_k = theta[mask]
            weights[mask] = self.compute_weights_per_k(theta_k, k)

        return weights / len(distinct_ks)
    
    def compute_ess(self, accepted_samples: dict[str, np.ndarray]) -> float:
        """Compute the Effective Sample Size (ESS).

        Parameters
        ----------
        accepted_samples : dict of str to np.ndarray
            A dictionary containing 'theta' (or 'parameters') and 'k' arrays.

        Returns
        -------
        ess : float
            The Effective Sample Size.
        """
        theta, k_vals = self._extract_samples(accepted_samples)

        n = len(theta)
        predicted_cost = self.cost_model(theta)
        g_val = np.empty(n, dtype=float)

        for k in np.unique(k_vals):
            mask = k_vals == k
            g_val[mask] = self.regularize_cost(predicted_cost[mask], k=k)

        return np.sum(g_val) ** 2 / (n * np.sum(g_val**2)) if n > 0 else 0.0

    def compute_cg(self, accepted_samples: dict[str, np.ndarray]) -> float:
        """Compute the Computational Gain (CG).

        Parameters
        ----------
        accepted_samples : dict of str to np.ndarray
            A dictionary containing 'parameters' and 'k' arrays.

        Returns
        -------
        cg : float
            The Computational Gain.
        """
        theta, _ = self._extract_samples(accepted_samples)
        predicted_cost = self.cost_model(theta)
        n = len(predicted_cost)

        # sample new thetas from the original prior
        theta_new = self.sample(batch_shape=(n,), cost_aware=False)
        prior_cost = self.cost_model(theta_new.get("parameters"))

        avg_cost_prior = np.mean(prior_cost)
        avg_cost_accepted = np.mean(predicted_cost)

        return avg_cost_prior / avg_cost_accepted

    def compute_metrics(self, accepted_samples: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute performance metrics for the cost-aware sampling.

        Parameters
        ----------
        accepted_samples : dict of str to np.ndarray
            The dict returned by :py:meth:`sample`, containing 'parameters'
            and 'k' -- the k value each sample was actually
            accepted under. Each sample's cost is regularized with its own k,
            so a mixture of k values (as MIS produces) is handled correctly
            instead of being collapsed onto a single k.

        Returns
        -------
        metrics : dict of str to float
            A dictionary containing metrics 'ess' (Effective Sample Size)
            and 'cg' (Computational Gain).
        """
        # Effective Sample Size (ESS)
        ess = self.compute_ess(accepted_samples)

        # Computational Gain (CG)
        cg = self.compute_cg(accepted_samples)

        return {"ess": ess, "cg": cg}

    def predicate(self, samples: dict[str, np.ndarray] | np.ndarray, k: float = 1.0) -> np.ndarray:
        """The cost-aware acceptance predicate.

        Given a batch of samples, this returns a boolean array indicating which
        samples are accepted based on their predicted cost.

        Parameters
        ----------
        samples : dict of str to np.ndarray or np.ndarray
            A batch of samples, as returned by :py:meth:`sample` or just the parameters.
        k : float, optional
            Power factor for cost regularization. Default is 1.0.

        Returns
        -------
        accept : np.ndarray
            A boolean array of shape ``(batch_size,)``.
        """
        if isinstance(samples, dict):
            theta = samples.get("parameters")
            if theta is None:
                raise KeyError("Samples dictionary must contain 'parameters'.")
        else:
            theta = samples

        predicted_cost = self.cost_model(theta)

        g_val = self.regularize_cost(predicted_cost, k=k)

        # Acceptance probability = gmin / g(cost(theta))
        prob_accept = self.gmin / g_val

        # Rejection sampling: accept if random draw U(0, 1) <= prob_accept
        return np.random.random(len(prob_accept)) <= prob_accept
