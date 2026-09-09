import sys
from unittest import mock

import pytest

import redback.get_data as get_data
import redback.filters as filters
from redback.get_data import otter
from redback.transient_models.afterglow_models import base_models


def test_afterglowpy_is_loaded_only_when_requested():
    original = (base_models.afterglow, base_models.jettype_dict, base_models.spectype_dict)
    try:
        base_models.afterglow = None
        base_models.jettype_dict = None
        base_models.spectype_dict = None
        with mock.patch.object(base_models, "import_module", side_effect=ModuleNotFoundError):
            with pytest.raises(ImportError, match=r"redback\[afterglow\]"):
                base_models._load_afterglowpy()
    finally:
        base_models.afterglow, base_models.jettype_dict, base_models.spectype_dict = original


def test_svo_filters_report_the_data_extra_when_astroquery_is_missing():
    with mock.patch.dict(
            sys.modules, {"astroquery": None, "astroquery.svo_fps": None}):
        with pytest.raises(ImportError, match=r"redback\[data\]"):
            filters._get_svo_fps()


def test_otter_availability_reflects_backend_import():
    assert get_data.OTTER_AVAILABLE is otter.OTTER_INSTALLED


def test_otter_wrapper_reports_the_data_extra_when_backend_is_missing():
    with mock.patch.object(get_data, "OTTER_AVAILABLE", False):
        with pytest.raises(ImportError, match=r"redback\[data\]"):
            get_data.get_kilonova_data_from_otter("AT2017gfo")
