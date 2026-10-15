####################################################################################################
#                                           conftest.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 2026-10-15                                                                              #
#                                                                                                  #
# Purpose: The GUI tests do not ask the Hub which checkpoints MRSuite keeps there: the pinned      #
#          ones stand, so the page opens as fast offline as on.                                    #
#                                                                                                  #
####################################################################################################
import pytest


#************#
#   no hub   #
#************#
@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    import mrsuite.models
    monkeypatch.setattr(mrsuite.models, "zoo", lambda **kwargs: {})
