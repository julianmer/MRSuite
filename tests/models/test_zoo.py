####################################################################################################
#                                           test_zoo.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-30                                                                              #
#                                                                                                  #
# Purpose: The model zoo on the Hugging Face Hub: every MRSuite model there loads by its name      #
#          alone, recipe and all, and runs in a pipeline on data acquired as it was trained.       #
#          Each repo is skipped until it is on the Hub, and all of them when the Hub is out of     #
#          reach.                                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import httpx

import mrsuite.models
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import challenge2016

from tests.models.test_models import _spectra


#***************#
#   challenge   #
#***************#
def _challenge():
    """Two of the 2016 challenge's spectra, acquired as WAND was trained."""
    if not challenge2016.BASIS.exists():
        pytest.skip("challenge submodule not checked out")
    return challenge2016.load(2)[0]


# each zoo repo: the SHA-256 of the weights it must hold (the published checkpoints, as
# converted and checked against the original code) and data acquired as it was trained
ZOO = {"MRSuite/wand-ismrm2016":
           ("635e74f89a9b6a33ea4fc757005f84c3986284d86f2e00334a111fd125f41107", _challenge),
       "MRSuite/mlp-supervised-7t-slaser":
           ("45a62ce6c27b96d48c1cd425a1ddd88e4ce7232ecf867395fb9c754ff6b142c4",
            lambda: _spectra(3)),
       "MRSuite/mlp-selfsupervised-7t-slaser":
           ("1db5b330097b9ae7673557c6d4a2664a5fbc307e43a268af4d5af23beef8d3e9",
            lambda: _spectra(3))}


#*********#
#   zoo   #
#*********#
@pytest.mark.parametrize("name", ZOO)
def test_a_zoo_model_loads_by_name_and_runs_in_a_pipeline(name):
    try:
        on_hub = HfApi().repo_exists(name)
    except (httpx.HTTPError, OSError):
        pytest.skip("the Hub is out of reach")
    if not on_hub:
        pytest.skip(f"{name} is not on the Hub yet")
    weights, data = ZOO[name]
    data = data()

    model = mrsuite.models.load(name)
    results = Pipeline([model]).run(data)

    assert model.fingerprint() == weights       # the mixin loads weights leniently
    assert len(results) == len(data.list())
    assert model.recipe["training"] and model.source["name"] == name


#*************#
#   listing   #
#*************#
def test_the_zoo_lists_every_repo_of_the_organisation_with_its_class():
    try:
        listed = mrsuite.models.zoo()
    except (httpx.HTTPError, OSError):
        pytest.skip("the Hub is out of reach")
    if not set(listed) >= set(ZOO) and get_token() is None:
        pytest.skip("the zoo is not visible from here (private repos need a token)")
    assert set(listed) >= set(ZOO)
    assert listed["MRSuite/wand-ismrm2016"] == "mrsuite.models.wand.WAND"
    assert listed["MRSuite/mlp-supervised-7t-slaser"] == "mrsuite.models.mlp.MLP"
    assert list(listed) == sorted(listed)
