import unittest
from copy import deepcopy

import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

import gwpopulation
from gwpopulation.models import mass

from . import TEST_BACKENDS

xp = np
N_TEST = 10


def double_power_prior():
    power_prior = PriorDict()
    power_prior["alpha_1"] = Uniform(minimum=-4, maximum=12)
    power_prior["alpha_2"] = Uniform(minimum=-4, maximum=12)
    power_prior["beta"] = Uniform(minimum=-4, maximum=12)
    power_prior["mmin"] = Uniform(minimum=3, maximum=10)
    power_prior["mmax"] = Uniform(minimum=40, maximum=100)
    power_prior["break_fraction"] = Uniform(minimum=0, maximum=1)
    return power_prior


def power_prior():
    power_prior = PriorDict()
    power_prior["alpha"] = Uniform(minimum=-4, maximum=12)
    power_prior["beta"] = Uniform(minimum=-4, maximum=12)
    power_prior["mmin"] = Uniform(minimum=3, maximum=10)
    power_prior["mmax"] = Uniform(minimum=40, maximum=100)
    return power_prior


def gauss_prior():
    gauss_prior = PriorDict()
    gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
    gauss_prior["mpp"] = Uniform(minimum=20, maximum=60)
    gauss_prior["sigpp"] = Uniform(minimum=0, maximum=10)
    return gauss_prior


def double_gauss_prior():
    double_gauss_prior = PriorDict()
    double_gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
    double_gauss_prior["lam_1"] = Uniform(minimum=0, maximum=1)
    double_gauss_prior["mpp_1"] = Uniform(minimum=20, maximum=60)
    double_gauss_prior["mpp_2"] = Uniform(minimum=20, maximum=100)
    double_gauss_prior["sigpp_1"] = Uniform(minimum=0, maximum=10)
    double_gauss_prior["sigpp_2"] = Uniform(minimum=0, maximum=10)
    return double_gauss_prior


class TestDoublePowerLaw(unittest.TestCase):
    def setUp(self):
        gwpopulation.set_backend("numpy")
        self.m1s, self.qs, self.dataset = get_primary_mass_ratio_data(np)
        self.power_prior = double_power_prior()
        self.n_test = N_TEST

    def test_double_power_law_zero_below_mmin(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            del parameters["beta"]
            p_m = mass.double_power_law_primary_mass(self.m1s, **parameters)
            self.assertEqual(np.max(p_m[self.m1s <= parameters["mmin"]]), 0.0)

    def test_power_law_primary_mass_ratio_zero_above_mmax(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            p_m = mass.double_power_law_primary_power_law_mass_ratio(
                self.dataset, **parameters
            )
            self.assertEqual(
                np.max(p_m[self.dataset["mass_1"] > parameters["mmax"]]), 0.0
            )


def get_primary_mass_ratio_data(xp):
    m1s = np.linspace(3, 100, 1000)
    qs = np.linspace(0.01, 1, 500)
    m1s_grid, qs_grid = xp.meshgrid(m1s, qs)
    dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
    return m1s, qs, dataset


def get_primary_secondary_data(xp):
    ms = np.linspace(3, 100, 1000)
    dm = ms[1] - ms[0]
    m1s_grid, m2s_grid = xp.meshgrid(ms, ms)
    dataset = dict(mass_1=m1s_grid, mass_2=m2s_grid)
    return ms, dm, dataset


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_power_law_primary_mass_ratio_zero_below_mmin(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    m1s, qs, dataset = get_primary_mass_ratio_data(xp)
    prior = power_prior()
    m2s = dataset["mass_1"] * dataset["mass_ratio"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.power_law_primary_mass_ratio(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[m2s < parameters["mmin"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_power_law_primary_mass_ratio_zero_above_mmax(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_mass_ratio_data(xp)
    prior = power_prior()
    m1s = dataset["mass_1"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.power_law_primary_mass_ratio(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[m1s > parameters["mmax"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_two_component_primary_mass_ratio_zero_below_mmin(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    m1s, qs, dataset = get_primary_mass_ratio_data(xp)
    prior = power_prior()
    prior.update(gauss_prior())
    m2s = dataset["mass_1"] * dataset["mass_ratio"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.two_component_primary_mass_ratio(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[m2s <= parameters["mmin"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_power_law_primary_secondary_zero_below_mmin(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_secondary_data(xp)
    prior = power_prior()
    m2s = dataset["mass_2"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.power_law_primary_secondary_independent(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[m2s <= parameters["mmin"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_power_law_primary_secondary_zero_above_mmax(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_secondary_data(xp)
    prior = power_prior()
    del prior["beta"]
    m1s = dataset["mass_1"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.power_law_primary_secondary_identical(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[m1s >= parameters["mmax"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_two_component_primary_secondary_zero_below_mmin(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_secondary_data(xp)
    prior = power_prior()
    prior.update(gauss_prior())
    del prior["beta"]
    m2s = dataset["mass_2"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.two_component_primary_secondary_identical(dataset, **parameters)
        assert np.max(p_m[m2s <= parameters["mmin"]]) == 0.0


class TestSmoothedMassDistribution(unittest.TestCase):
    def setUp(self):
        gwpopulation.set_backend("numpy")
        self.trapz = np.trapz
        self.m1s = np.linspace(2, 100, 1000)
        self.qs = np.linspace(0.01, 1, 500)
        m1s_grid, qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
        self.power_prior = power_prior()
        self.gauss_prior = gauss_prior()
        self.double_gauss_prior = double_gauss_prior()
        self.broken_power_prior = double_power_prior()
        self.broken_power_peak_prior = double_power_prior()
        self.broken_power_peak_prior.update(gauss_prior())
        self.smooth_prior = PriorDict()
        self.smooth_prior["delta_m"] = Uniform(minimum=0, maximum=10)
        self.n_test = N_TEST

    def test_single_peak_delta_m_zero_matches_two_component_primary_mass_ratio(self):
        max_diffs = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.gauss_prior.sample())
            p_m1 = mass.two_component_primary_mass_ratio(self.dataset, **parameters)
            parameters["delta_m"] = 0
            p_m2 = mass.SinglePeakSmoothedMassDistribution()(self.dataset, **parameters)
            max_diffs.append(_max_abs_difference(p_m1, p_m2))
        self.assertAlmostEqual(max(max_diffs), 0.0)

    def test_double_peak_delta_m_zero_matches_two_component_primary_mass_ratio(self):
        max_diffs = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.double_gauss_prior.sample())
            del parameters["beta"]
            p_m1 = mass.three_component_single(
                mass=self.dataset["mass_1"], **parameters
            )
            parameters["delta_m"] = 0
            p_m2 = mass.MultiPeakSmoothedMassDistribution().p_m1(
                self.dataset, **parameters
            )
            max_diffs.append(_max_abs_difference(p_m1, p_m2))
        self.assertAlmostEqual(max(max_diffs), 0.0)

    def test_single_peak_normalised(self):
        norms = list()
        model = mass.SinglePeakSmoothedMassDistribution()
        prior = deepcopy(self.power_prior)
        prior.update(self.gauss_prior)
        prior.update(self.smooth_prior)
        for ii in range(self.n_test):
            parameters = prior.sample()
            p_m = model(self.dataset, **parameters)
            norms.append(self.trapz(self.trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_double_peak_normalised(self):
        norms = list()
        model = mass.MultiPeakSmoothedMassDistribution()
        prior = deepcopy(self.power_prior)
        prior.update(self.double_gauss_prior)
        prior.update(self.smooth_prior)
        for ii in range(self.n_test):
            parameters = prior.sample()
            p_m = model(self.dataset, **parameters)
            norms.append(self.trapz(self.trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_broken_power_law_normalised(self):
        norms = list()
        model = mass.BrokenPowerLawSmoothedMassDistribution()
        prior = deepcopy(self.broken_power_prior)
        prior.update(self.smooth_prior)
        for ii in range(self.n_test):
            parameters = prior.sample()
            p_m = model(self.dataset, **parameters)
            norms.append(self.trapz(self.trapz(p_m, self.m1s), self.qs))
            print(norms, parameters, p_m)
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_broken_power_law_peak_normalised(self):
        norms = list()
        model = mass.BrokenPowerLawPeakSmoothedMassDistribution()
        prior = deepcopy(self.broken_power_peak_prior)
        prior.update(self.smooth_prior)
        for ii in range(self.n_test):
            parameters = prior.sample()
            p_m = model(self.dataset, **parameters)
            norms.append(self.trapz(self.trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_set_minimum_and_maximum(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mpp"] = 130
        parameters["sigpp"] = 1
        parameters["lam"] = 0.5
        parameters["mmin"] = 5
        self.assertEqual(
            model(
                dict(mass_1=8 * np.ones(5), mass_ratio=0.5 * np.ones(5)), **parameters
            )[0],
            0,
        )
        self.assertGreater(
            model(
                dict(mass_1=130 * np.ones(5), mass_ratio=0.9 * np.ones(5)), **parameters
            )[0],
            0,
        )

    def test_mmin_below_global_minimum_raises_error(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mmin"] = 2
        with self.assertRaises(ValueError):
            model(self.dataset, **parameters)

    def test_mmax_above_global_maximum_raises_error(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mmax"] = 200
        with self.assertRaises(ValueError):
            model(self.dataset, **parameters)


def _max_abs_difference(array, comparison, xp=np):
    return float(xp.max(xp.abs(comparison - xp.asarray(array))))


if __name__ == "__main__":
    unittest.main()
