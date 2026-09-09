from unittest.mock import MagicMock, PropertyMock, patch

import bilby
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from redback.optimizer import _PointObjective, _json_safe, run_point_estimation
from redback.result import RedbackResult, read_in_result


class QuadraticLikelihood(bilby.Likelihood):
    def __init__(self, target=2.0, sigma=0.5):
        super().__init__()
        self.target = target
        self.sigma = sigma

    def log_likelihood(self, parameters=None):
        parameters = self.parameters if parameters is None else parameters
        return -0.5 * ((parameters["x"] - self.target) / self.sigma) ** 2


def _optimizer_kwargs(seed=1234):
    return {
        "seed": seed,
        "maxiter": 50,
        "popsize": 8,
        "tol": 1e-8,
        "local_options": {"xtol": 1e-10, "ftol": 1e-10},
    }


def test_mle_recovers_quadratic_optimum(tmp_path):
    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(-5, 5)})

    result = run_point_estimation(
        likelihood=QuadraticLikelihood(),
        priors=priors,
        fit_method="mle",
        label="quadratic_mle",
        outdir=tmp_path,
        optimizer_kwargs=_optimizer_kwargs(),
    )

    assert isinstance(result, RedbackResult)
    assert len(result.posterior) == 1
    assert result.posterior.iloc[0]["x"] == pytest.approx(2.0, abs=1e-5)
    assert result.meta_data["fit_method"] == "mle"
    assert result.meta_data["point_estimate"] is True
    assert result.is_point_estimate is True
    assert result.point_estimate["x"] == pytest.approx(2.0, abs=1e-5)
    assert (tmp_path / "quadratic_mle_result.json").exists()
    loaded = read_in_result(outdir=tmp_path, label="quadratic_mle", extension="json")
    assert loaded.fit_method == "mle"
    assert loaded.point_estimate["x"] == pytest.approx(2.0, abs=1e-5)
    with pytest.raises(ValueError, match="Corner plots require posterior samples"):
        loaded.plot_corner(save=False)
    transient = MagicMock()
    with patch.object(
        RedbackResult, "transient", new_callable=PropertyMock, return_value=transient
    ):
        loaded.plot_lightcurve(model=lambda time, **kwargs: time)
    assert transient.plot_lightcurve.call_args.kwargs["random_models"] == 1


def test_nonuniform_prior_shifts_map_from_mle(tmp_path):
    priors = bilby.prior.PriorDict(
        {
            "x": bilby.prior.PowerLaw(alpha=2, minimum=0.1, maximum=5.0),
        }
    )

    mle = run_point_estimation(
        likelihood=QuadraticLikelihood(),
        priors=priors,
        fit_method="mle",
        label="quadratic_mle",
        outdir=tmp_path,
        optimizer_kwargs=_optimizer_kwargs(),
    )
    map_result = run_point_estimation(
        likelihood=QuadraticLikelihood(),
        priors=priors,
        fit_method="map",
        label="quadratic_map",
        outdir=tmp_path,
        optimizer_kwargs=_optimizer_kwargs(),
    )

    assert mle.posterior.iloc[0]["x"] == pytest.approx(2.0, abs=1e-5)
    expected_map = 1 + np.sqrt(1.5)
    assert map_result.posterior.iloc[0]["x"] == pytest.approx(expected_map, abs=1e-5)


def test_fixed_parameter_is_available_to_derived_constraint(tmp_path):
    def conversion_function(parameters):
        parameters["derived"] = parameters["x"] + parameters["offset"]
        return parameters

    priors = bilby.prior.PriorDict(
        dictionary={
            "x": bilby.prior.Uniform(0, 2),
            "offset": 0.5,
            "derived": bilby.prior.Constraint(0, 1),
        },
        conversion_function=conversion_function,
    )

    class StrictLikelihood(QuadraticLikelihood):
        def log_likelihood(self, parameters=None):
            assert set(parameters) == {"x", "offset"}
            return super().log_likelihood(parameters)

    result = run_point_estimation(
        likelihood=StrictLikelihood(target=1.5),
        priors=priors,
        fit_method="mle",
        label="constrained_mle",
        outdir=tmp_path,
        optimizer_kwargs=_optimizer_kwargs(),
    )

    row = result.posterior.iloc[0]
    assert row["x"] == pytest.approx(0.5, abs=1e-4)
    assert row["offset"] == pytest.approx(0.5)
    assert row["derived"] <= 1.0


def test_invalid_initial_parameters_are_rejected(tmp_path):
    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)})

    with pytest.raises(ValueError, match="prior bounds"):
        run_point_estimation(
            likelihood=QuadraticLikelihood(),
            priors=priors,
            fit_method="mle",
            label="invalid",
            outdir=tmp_path,
            optimizer="powell",
            initial_parameters={"x": 2.0},
        )


def test_nonfinite_likelihood_domain_raises(tmp_path):
    class NonFiniteLikelihood(QuadraticLikelihood):
        def log_likelihood(self, parameters=None):
            return np.nan

    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)})

    with pytest.raises(RuntimeError, match="finite likelihood"):
        run_point_estimation(
            likelihood=NonFiniteLikelihood(),
            priors=priors,
            fit_method="mle",
            label="nonfinite",
            outdir=tmp_path,
            optimizer_kwargs={**_optimizer_kwargs(), "fail_on_nonconvergence": False},
        )


def test_powell_nonfinite_initial_point_raises_runtime_error(tmp_path):
    class NonFiniteLikelihood(QuadraticLikelihood):
        def log_likelihood(self, parameters=None):
            return np.nan

    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)})

    with pytest.raises(RuntimeError, match="finite likelihood"):
        run_point_estimation(
            likelihood=NonFiniteLikelihood(),
            priors=priors,
            fit_method="mle",
            label="nonfinite_powell",
            outdir=tmp_path,
            optimizer="powell",
            initial_parameters={"x": 0.5},
        )


@patch("redback.optimizer.minimize")
@patch("redback.optimizer.differential_evolution")
def test_auto_retains_successful_global_result_when_powell_is_worse(
    differential_evolution_mock, minimize_mock, tmp_path
):
    differential_evolution_mock.return_value = OptimizeResult(
        x=np.array([0.2]), success=True, message="global success", nit=1
    )
    minimize_mock.return_value = OptimizeResult(
        x=np.array([0.9]), success=False, message="local failure", nit=1
    )
    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)})

    result = run_point_estimation(
        likelihood=QuadraticLikelihood(target=0.2),
        priors=priors,
        fit_method="mle",
        label="global_result",
        outdir=tmp_path,
        optimizer_kwargs=_optimizer_kwargs(),
    )

    assert result.point_estimate["x"] == pytest.approx(0.2)
    assert result.optimization["success"] is True
    assert result.optimization["message"] == "global success"


def test_model_exceptions_are_not_silenced(tmp_path):
    class BrokenLikelihood(QuadraticLikelihood):
        def log_likelihood(self, parameters=None):
            raise ValueError("broken model configuration")

    priors = bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)})

    with pytest.raises(ValueError, match="broken model configuration"):
        run_point_estimation(
            likelihood=BrokenLikelihood(),
            priors=priors,
            fit_method="mle",
            label="broken",
            outdir=tmp_path,
            optimizer_kwargs=_optimizer_kwargs(),
        )


def test_all_invalid_constraints_raise_before_optimization(tmp_path):
    def conversion_function(parameters):
        parameters["derived"] = np.full_like(parameters["x"], 2.0)
        return parameters

    priors = bilby.prior.PriorDict(
        dictionary={
            "x": bilby.prior.Uniform(0, 1),
            "derived": bilby.prior.Constraint(0, 1),
        },
        conversion_function=conversion_function,
    )

    with pytest.raises(ValueError, match="valid optimizer population"):
        run_point_estimation(
            likelihood=QuadraticLikelihood(),
            priors=priors,
            fit_method="mle",
            label="invalid_constraints",
            outdir=tmp_path,
            optimizer_kwargs=_optimizer_kwargs(),
        )


@pytest.mark.parametrize(
    "keyword,value,message",
    [("fit_method", "posterior", "fit_method"),
     ("optimizer", "nelder-mead", "optimizer")],
)
def test_invalid_point_estimation_modes_raise(tmp_path, keyword, value, message):
    arguments = dict(
        likelihood=QuadraticLikelihood(),
        priors={"x": bilby.prior.Uniform(0, 1)},
        fit_method="mle", optimizer="auto", label="invalid", outdir=tmp_path)
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        run_point_estimation(**arguments)


def test_parallel_optimizer_workers_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="workers.*1"):
        run_point_estimation(
            likelihood=QuadraticLikelihood(),
            priors={"x": bilby.prior.Uniform(0, 1)},
            fit_method="mle", optimizer="auto", label="parallel", outdir=tmp_path,
            optimizer_kwargs={"workers": 2})


def test_initial_parameters_require_every_free_parameter_and_constraints():
    priors = bilby.prior.PriorDict({
        "x": bilby.prior.Uniform(0, 1),
        "y": bilby.prior.Uniform(0, 1),
    })
    objective = _PointObjective(QuadraticLikelihood(), priors, "mle")
    with pytest.raises(ValueError, match="missing free parameters: y"):
        objective.unit_point_from_parameters({"x": 0.5})

    def conversion(parameters):
        parameters["derived"] = parameters["x"]
        return parameters

    constrained = bilby.prior.PriorDict(
        {"x": bilby.prior.Uniform(0, 1), "derived": bilby.prior.Constraint(0, 0.4)},
        conversion_function=conversion)
    constrained_objective = _PointObjective(QuadraticLikelihood(), constrained, "mle")
    with pytest.raises(ValueError, match="do not satisfy"):
        constrained_objective.unit_point_from_parameters({"x": 0.8})


def test_objective_rejects_constraints_and_nonfinite_targets():
    def conversion(parameters):
        parameters["derived"] = parameters["x"]
        return parameters

    priors = bilby.prior.PriorDict(
        {"x": bilby.prior.Uniform(0, 1), "derived": bilby.prior.Constraint(0, 0.4)},
        conversion_function=conversion)
    objective = _PointObjective(QuadraticLikelihood(), priors, "mle")
    assert objective.is_valid(np.array([0.8])) is False
    assert objective.evaluate(np.array([0.8])) == np.inf

    class NonFiniteLikelihood(QuadraticLikelihood):
        def log_likelihood(self, parameters=None):
            return np.inf

    nonfinite = _PointObjective(
        NonFiniteLikelihood(), bilby.prior.PriorDict({"x": bilby.prior.Uniform(0, 1)}), "mle")
    assert nonfinite.evaluate(np.array([0.5])) == np.inf


def test_all_fixed_parameters_return_without_optimizer(tmp_path):
    result = run_point_estimation(
        likelihood=QuadraticLikelihood(target=0.5), priors={"x": 0.5},
        fit_method="mle", optimizer="auto", label="fixed", outdir=tmp_path)
    assert result.point_estimate["x"] == pytest.approx(0.5)
    assert result.optimization["message"] == "All model parameters are fixed"


@patch("redback.optimizer.minimize")
@patch("redback.optimizer.logger.warning")
def test_nonconverged_powell_can_return_with_warning(warning, minimize_mock, tmp_path):
    minimize_mock.return_value = OptimizeResult(
        x=np.array([0.5]), success=False, message="iteration limit", nit=2)
    result = run_point_estimation(
        likelihood=QuadraticLikelihood(target=0.5),
        priors={"x": bilby.prior.Uniform(0, 1)}, fit_method="mle",
        optimizer="powell", label="warning", outdir=tmp_path,
        initial_parameters={"x": 0.5},
        optimizer_kwargs={"fail_on_nonconvergence": False})
    assert result.optimization["success"] is False
    warning.assert_called_once_with(
        "Point estimation did not converge: %s", "iteration limit")


def test_json_safe_normalizes_nested_numpy_values():
    value = {"array": np.array([1, 2]), "items": (np.float64(3.0),)}
    assert _json_safe(value) == {"array": [1, 2], "items": [3.0]}
