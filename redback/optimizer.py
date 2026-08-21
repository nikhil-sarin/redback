"""Point-estimation support for Redback fits."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
from bilby.core.prior import Constraint, DeltaFunction, Prior, PriorDict
from scipy.optimize import differential_evolution, minimize

from redback.result import RedbackResult
from redback.utils import logger

try:
    from bilby.core.likelihood import _safe_likelihood_call
except ImportError:  # pragma: no cover - compatibility with older Bilby releases

    def _safe_likelihood_call(likelihood, parameters, use_ratio=False):
        likelihood.parameters.update(parameters)
        if use_ratio:
            return likelihood.log_likelihood_ratio()
        return likelihood.log_likelihood()


_FIT_METHODS = {"map", "mle"}
_OPTIMIZERS = {"auto", "differential_evolution", "powell"}


class _PointObjective:
    """Evaluate a Bilby likelihood from coordinates in the prior unit cube."""

    def __init__(self, likelihood, priors, fit_method):
        self.likelihood = likelihood
        self.priors = priors
        self.fit_method = fit_method
        self.search_parameter_keys = [
            key
            for key, prior in priors.items()
            if isinstance(prior, Prior) and not prior.is_fixed
        ]
        self.fixed_parameters = {
            key: prior.peak
            for key, prior in priors.items()
            if isinstance(prior, DeltaFunction)
        }
        self.constraint_parameter_keys = [
            key for key, prior in priors.items() if isinstance(prior, Constraint)
        ]
        self.num_likelihood_evaluations = 0

    def point_from_unit_cube(self, unit_point):
        values = self.priors.rescale(self.search_parameter_keys, unit_point)
        point = self.fixed_parameters.copy()
        point.update(dict(zip(self.search_parameter_keys, values)))
        return point

    def unit_point_from_parameters(self, parameters):
        missing = set(self.search_parameter_keys) - set(parameters)
        if missing:
            raise ValueError(
                "initial_parameters is missing free parameters: "
                f"{', '.join(sorted(missing))}"
            )
        if any(
            not np.isfinite(self.priors[key].ln_prob(parameters[key]))
            for key in self.search_parameter_keys
        ):
            raise ValueError("initial_parameters must lie within the prior bounds")
        unit_point = np.array(
            [
                self.priors[key].cdf(parameters[key])
                for key in self.search_parameter_keys
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(unit_point)) or np.any(
            (unit_point < 0) | (unit_point > 1)
        ):
            raise ValueError("initial_parameters must lie within the prior bounds")
        if not self.is_valid(unit_point):
            raise ValueError("initial_parameters do not satisfy the prior constraints")
        return unit_point

    def is_valid(self, unit_point):
        point = self.point_from_unit_cube(unit_point)
        constraints_valid = np.all(self.priors.evaluate_constraints(point.copy()))
        if not constraints_valid:
            return False
        if self.fit_method == "map":
            return np.isfinite(self.log_prior(point))
        return True

    def log_prior(self, point):
        return float(self.priors.ln_prob(point.copy(), normalized=False))

    def evaluate(self, unit_point):
        point = self.point_from_unit_cube(unit_point)
        if not np.all(self.priors.evaluate_constraints(point.copy())):
            return np.inf

        if self.fit_method == "map":
            log_prior = self.log_prior(point)
            if not np.isfinite(log_prior):
                return np.inf

        self.num_likelihood_evaluations += 1
        log_likelihood = float(
            _safe_likelihood_call(self.likelihood, parameters=point, use_ratio=False)
        )
        if not np.isfinite(log_likelihood):
            return np.inf

        log_target = log_likelihood
        if self.fit_method == "map":
            log_target += log_prior
        return -log_target


def _valid_initial_population(objective, population_size, rng, initial_unit_point=None):
    population = []
    if initial_unit_point is not None:
        population.append(np.asarray(initial_unit_point, dtype=float))

    max_attempts = max(1000, 1000 * population_size)
    attempts = 0
    while len(population) < population_size and attempts < max_attempts:
        candidate = rng.uniform(
            np.finfo(float).eps,
            1 - np.finfo(float).eps,
            len(objective.search_parameter_keys),
        )
        if objective.is_valid(candidate):
            population.append(candidate)
        attempts += 1

    if len(population) < population_size:
        raise ValueError(
            "Could not construct a valid optimizer population from the priors. "
            "Check the prior bounds and derived constraints."
        )
    return np.asarray(population)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_point_estimation(
    likelihood,
    priors,
    fit_method,
    label,
    outdir,
    meta_data=None,
    optimizer="auto",
    optimizer_kwargs=None,
    initial_parameters=None,
    save_format="json",
    gzip=False,
):
    """Maximize a likelihood or posterior and return a one-row result.

    The optimization is performed in the unit cube defined by the supplied
    Bilby priors. MLE fits retain prior bounds and constraints but do not use
    the prior density in the objective.
    """
    fit_method = str(fit_method).lower()
    optimizer = str(optimizer).lower()
    if fit_method not in _FIT_METHODS:
        raise ValueError(f"fit_method must be one of {sorted(_FIT_METHODS)}")
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}")

    priors = deepcopy(priors) if isinstance(priors, PriorDict) else PriorDict(priors)
    priors.convert_floats_to_delta_functions()
    objective = _PointObjective(
        likelihood=likelihood, priors=priors, fit_method=fit_method
    )
    options = dict(optimizer_kwargs or {})
    seed = options.pop("seed", None)
    local_options = options.pop("local_options", None)
    fail_on_nonconvergence = options.pop("fail_on_nonconvergence", True)
    workers = options.get("workers", 1)
    if workers != 1:
        raise ValueError(
            "Point estimation currently requires optimizer_kwargs['workers'] == 1"
        )

    start = perf_counter()
    initial_unit_point = None
    if initial_parameters is not None:
        initial_unit_point = objective.unit_point_from_parameters(initial_parameters)

    global_result = None
    local_result = None
    ndim = len(objective.search_parameter_keys)
    bounds = [(0.0, 1.0)] * ndim

    if ndim == 0:
        best_unit_point = np.empty(0)
        best_objective = objective.evaluate(best_unit_point)
        success = np.isfinite(best_objective)
        message = "All model parameters are fixed"
    else:
        if optimizer in {"auto", "differential_evolution"}:
            popsize = int(options.get("popsize", 15))
            population_size = max(5, popsize * ndim)
            population = _valid_initial_population(
                objective=objective,
                population_size=population_size,
                rng=np.random.default_rng(seed),
                initial_unit_point=initial_unit_point,
            )
            de_options = dict(options)
            de_options.setdefault("polish", False)
            de_options["init"] = population
            de_options["seed"] = seed
            # Surface model errors before scipy can rewrite ValueError as an
            # optimizer protocol error, and reject an entirely non-finite
            # initial population without running a futile global search.
            if not any(
                np.isfinite(objective.evaluate(candidate)) for candidate in population
            ):
                raise RuntimeError(
                    "Point estimation did not find a finite likelihood value in "
                    "the initial population"
                )
            global_result = differential_evolution(
                objective.evaluate, bounds=bounds, **de_options
            )
            best_unit_point = global_result.x
        else:
            best_unit_point = initial_unit_point
            if best_unit_point is None:
                population = _valid_initial_population(
                    objective=objective,
                    population_size=1,
                    rng=np.random.default_rng(seed),
                )
                best_unit_point = population[0]

        if optimizer in {"auto", "powell"}:
            global_objective = objective.evaluate(best_unit_point)
            if not np.isfinite(global_objective):
                raise RuntimeError(
                    "Point estimation did not find a finite likelihood value"
                )
            local_result = minimize(
                objective.evaluate,
                x0=best_unit_point,
                method="Powell",
                bounds=bounds,
                options=local_options,
            )
            local_objective = objective.evaluate(local_result.x)
            local_is_not_worse = local_objective <= global_objective or np.isclose(
                local_objective, global_objective
            )
            if np.isfinite(local_objective) and local_is_not_worse:
                best_unit_point = local_result.x
                success = bool(local_result.success)
                message = str(local_result.message)
            elif global_result is not None:
                success = bool(global_result.success)
                message = str(global_result.message)
            else:
                success = False
                message = "Powell did not improve the finite initial point"
        else:
            success = bool(global_result.success)
            message = str(global_result.message)

        best_objective = objective.evaluate(best_unit_point)
        success = success and np.isfinite(best_objective)

    if not np.isfinite(best_objective):
        raise RuntimeError("Point estimation did not find a finite likelihood value")
    if not success and fail_on_nonconvergence:
        raise RuntimeError(f"Point estimation did not converge: {message}")
    if not success:
        logger.warning("Point estimation did not converge: %s", message)

    point = objective.point_from_unit_cube(best_unit_point)
    objective.num_likelihood_evaluations += 1
    log_likelihood = float(
        _safe_likelihood_call(likelihood, parameters=point, use_ratio=False)
    )
    log_prior = objective.log_prior(point)
    converted_point = priors.conversion_function(point.copy())
    posterior = pd.DataFrame(
        [
            {
                **converted_point,
                "log_likelihood": log_likelihood,
                "log_prior": log_prior,
                "log_posterior": log_likelihood + log_prior,
            }
        ]
    )
    elapsed = perf_counter() - start

    optimization_metadata = {
        "optimizer": optimizer,
        "success": success,
        "message": message,
        "objective": float(best_objective),
        "num_likelihood_evaluations": objective.num_likelihood_evaluations,
        "sampling_time": elapsed,
        "seed": seed,
    }
    if global_result is not None:
        optimization_metadata["global_success"] = bool(global_result.success)
        optimization_metadata["global_message"] = str(global_result.message)
        optimization_metadata["global_iterations"] = int(global_result.nit)
    if local_result is not None:
        optimization_metadata["local_success"] = bool(local_result.success)
        optimization_metadata["local_message"] = str(local_result.message)
        optimization_metadata["local_iterations"] = int(local_result.nit)

    result_meta_data = dict(meta_data or {})
    result_meta_data.update(
        {
            "fit_method": fit_method,
            "point_estimate": True,
            "optimization": _json_safe(optimization_metadata),
        }
    )
    stored_options = dict(optimizer_kwargs or {})
    stored_options["optimizer"] = optimizer
    result = RedbackResult(
        label=label,
        outdir=outdir,
        sampler=fit_method,
        search_parameter_keys=objective.search_parameter_keys,
        fixed_parameter_keys=list(objective.fixed_parameters),
        constraint_parameter_keys=objective.constraint_parameter_keys,
        priors=priors,
        sampler_kwargs=_json_safe(stored_options),
        meta_data=result_meta_data,
        posterior=posterior,
        samples=np.asarray([[point[key] for key in objective.search_parameter_keys]]),
        log_likelihood_evaluations=np.asarray([log_likelihood]),
        log_prior_evaluations=np.asarray([log_prior]),
        sampling_time=elapsed,
        num_likelihood_evaluations=objective.num_likelihood_evaluations,
        use_ratio=False,
    )
    result.save_to_file(extension=save_format, gzip=gzip, overwrite=True)
    return result
