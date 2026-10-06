import pytest
import keras
import numpy as np
from bayesflow.simulators.cost_aware_proposal import CostAwareProposal




def test_two_moons(two_moons_simulator, batch_size):
    samples = two_moons_simulator.sample((batch_size,))

    assert isinstance(samples, dict)
    assert list(samples.keys()) == ["parameters", "observables"]
    assert all(isinstance(value, np.ndarray) for value in samples.values())

    assert samples["parameters"].shape == (batch_size, 2)
    assert samples["observables"].shape == (batch_size, 2)


def test_gaussian_linear(gaussian_linear_simulator, batch_size):
    samples = gaussian_linear_simulator.sample((batch_size,))

    # test n_obs respected if applicable
    if hasattr(gaussian_linear_simulator, "n_obs") and isinstance(gaussian_linear_simulator.n_obs, int):
        assert samples["observables"].shape[1] == gaussian_linear_simulator.n_obs


def test_sample(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape
        assert value.shape[0] == batch_size

        # test batch randomness
        assert not np.allclose(value, value[0])


def test_sample_batched(simulator, batch_size):
    sample_size = 2
    samples = simulator.sample_batched((batch_size,), sample_size=sample_size)

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape (sample_batched rounds up to complete batches)
        assert value.shape[0] == int(np.ceil(batch_size / sample_size)) * sample_size

        # test batch randomness
        assert not np.allclose(value, value[0])


def test_fixed_sample(composite_gaussian, batch_size, fixed_n, fixed_mu):
    samples = composite_gaussian.sample((batch_size,), n=fixed_n, mu=fixed_mu)

    assert samples["n"] == fixed_n
    assert samples["mu"].shape == (batch_size, 1)
    assert np.all(samples["mu"] == fixed_mu)
    assert samples["y"].shape == (batch_size, fixed_n)


def test_multimodel_sample(multimodel, batch_size):
    samples = multimodel.sample(batch_size)

    assert set(samples) == {"n", "mu", "y", "model_indices"}
    assert samples["mu"].shape == (batch_size, 1)
    assert samples["y"].shape == (batch_size, samples["n"])


def test_multimodel_key_conflicts_sample(multimodel_key_conflicts, batch_size):
    if multimodel_key_conflicts.key_conflicts == "drop":
        samples = multimodel_key_conflicts.sample(batch_size)
        assert set(samples) == {"x", "model_indices"}
    elif multimodel_key_conflicts.key_conflicts == "fill":
        samples = multimodel_key_conflicts.sample(batch_size)
        assert set(samples) == {"x", "model_indices", "c", "w"}
        assert np.sum(np.isnan(samples["c"])) + np.sum(np.isnan(samples["w"])) == batch_size
    elif multimodel_key_conflicts.key_conflicts == "error":
        with pytest.raises(ValueError):
            samples = multimodel_key_conflicts.sample(batch_size)


def test_cost_aware_proposal_sample(batch_size):
    # Setup: simple prior and cost model
    def prior():
        return np.random.uniform(0, 1, size=1)

    # Cost model: cost increases with theta. Lower theta = cheaper.
    def cost_model(theta):
        return theta.flatten()

    sim = CostAwareProposal(prior=prior, cost_model=cost_model, gmin=0.1)

    # Test cost_aware=False: should return samples directly from prior
    samples_no_cost = sim.sample(batch_size, cost_aware=False)
    assert "parameters" in samples_no_cost
    assert samples_no_cost["parameters"].shape == (batch_size, 1)

    # Test cost_aware=True: should return samples and k
    samples_cost = sim.sample(batch_size, k=1.0, cost_aware=True)
    assert "parameters" in samples_cost
    assert "k" in samples_cost
    assert samples_cost["parameters"].shape == (batch_size, 1)
    assert samples_cost["k"].shape == (batch_size,)
    np.testing.assert_array_equal(samples_cost["k"], np.ones(batch_size))

    metrics = sim.compute_metrics(samples_cost)
    assert "ess" in metrics
    assert "cg" in metrics
    assert metrics["ess"] >= 0
    assert metrics["cg"] >= 0

    weights = sim.compute_weights(samples_cost)
    assert weights.shape == (batch_size,)
    np.testing.assert_allclose(np.sum(weights), 1.0)

def test_cost_aware_proposal_multiple_k(batch_size):
    def prior():
        return np.random.uniform(0, 1, size=1)

    def cost_model(theta):
        return theta.flatten()

    sim = CostAwareProposal(prior=prior, cost_model=cost_model)
    k_vals = [0.5, 1.5]
    
    samples = sim.sample(batch_size, k=k_vals, cost_aware=True)
    
    assert "parameters" in samples
    assert "k" in samples
    assert samples["parameters"].shape == (batch_size, 1)
    assert samples["k"].shape == (batch_size,)
    
    # Check if both k values are present (roughly equal split)
    unique_ks, counts = np.unique(samples["k"], return_counts=True)
    assert set(unique_ks) == set(k_vals)
    assert np.abs(counts[0] - counts[1]) <= 1

