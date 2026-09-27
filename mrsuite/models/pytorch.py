####################################################################################################
#                                            pytorch.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The Model interface for PyTorch: a torch module whose forward is the network, saved     #
#          to and loaded from the Hub as safetensors.                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
from typing import Any, Dict

import torch

from huggingface_hub import PyTorchModelHubMixin

from mrsuite.models.base import HUB, Model


#**************************************************************************************************#
#                                         Class TorchModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# A Model in PyTorch: implement prepare, forward and interpret, as for any torch module.           #
#                                                                                                  #
#**************************************************************************************************#
class TorchModel(Model, torch.nn.Module, PyTorchModelHubMixin, **HUB):
    """A Model in PyTorch: implement prepare, forward and interpret, as for any torch module.

    Prediction runs ``forward`` in eval mode without gradients, on the model's device.
    Calling the model on a tensor is plain PyTorch, so training code is unchanged.
    """

    #*************#
    #   predict   #
    #*************#
    def predict(self, x: Any) -> Any:
        device = next(self.parameters(), torch.empty(0)).device
        mode = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self(torch.as_tensor(x).to(device))
        finally:
            self.train(mode)

    #*****************#
    #   fingerprint   #
    #*****************#
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {"torch": torch.__version__}
