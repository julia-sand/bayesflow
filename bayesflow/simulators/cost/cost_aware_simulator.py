import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from ..simulator import Simulator
from .cost_interp_model import CostInterpModel


class CostAwareSimulator(Simulator):
    """Implements a simulator based on rejection sampling of a cost-aware proxy prior.

    Simulation-based inference can be expensive when the cost of a single simulation
    depends on the parameter value. Cost-aware SBI reduces this cost by sampling the
    parameters from a proxy prior that is biased towards cheaper parameter values. The
    bias is introduced through rejection sampling whose acceptance probability is a
    function of a regularisation of the predicted cost, ``g(c(theta))``, where
    ``c(theta)`` is predicted by a cost interpolation model.
    """

    def __init__(self, simulator: Simulator, *, cost_model: CostInterpModel = None):
        """
        Initialize a cost-aware simulator that wraps a base simulator.

        Parameters
        ----------
        simulator : Simulator
            The base simulator that samples the parameters and the data.
        cost_model : CostInterpModel, optional
            A fitted cost interpolation model used to predict the cost of a parameter
            value. It is passed to `predicate` to evaluate the acceptance
            probability. Default is None.
        """
        self.simulator = simulator
        self.cost_model = cost_model

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """Sample using the wrapped sampling function.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number
            of samples, but an int can also be passed.
        **kwargs
            Additional keyword arguments passed to the base simulator.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary of sampled outputs.
        """
        return self.simulator.sample(batch_shape, **kwargs)

    def predicate(self, samples: dict[str, np.ndarray], cost_model: CostInterpModel) -> np.ndarray:
        """Placeholder for the cost-aware acceptance predicate.

        Given a batch of samples, this should return a boolean array of shape
        ``(batch_size,)`` indicating which samples are accepted. The acceptance
        probability is a function of a regularisation of the predicted cost,
        ``g(c(theta))``, where ``c(theta)`` is predicted by ``cost_model``.

        Parameters
        ----------
        samples : dict of str to np.ndarray
            A batch of samples, as returned by :py:meth:`sample`.
        cost_model : CostInterpModel
            A fitted cost interpolation model used to predict the cost of each
            parameter value in the batch.

        Returns
        -------
        accept : np.ndarray
            A boolean array of shape ``(batch_size,)``.
        """
        raise NotImplementedError 

