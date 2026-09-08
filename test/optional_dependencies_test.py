import sys
from unittest import mock

import pytest

import redback.filters as filters
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
