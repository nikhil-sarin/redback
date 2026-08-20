import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

import bilby

from redback.optimizer import run_point_estimation
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
