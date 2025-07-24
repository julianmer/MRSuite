####################################################################################################
#                                           auxiliary.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 22/07/25                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions are defined here.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pandas as pd


#****************************************************#
#   load ISMRM2016 fitting challenge concentration   #
#****************************************************#
def load_ISMRM2016_challenge_concs(path2conc):
    """
    Load the ground truth concentrations of the ISMRM 2016 fitting challenge,
    specifically a list of concentrations from an EXCEL file

    Parameters:
        path2conc (str): Path to the EXCEL file containing the concentrations.

    Returns:
        dict: A dictionary containing metabolite names as keys and their concentrations as values.
    """
    truth = {'Ace': 0.0}  # initialize, Ace is only partially present

    file = pd.read_excel(path2conc, header=17)
    for i, met in enumerate(file['Metabolites']):
        if not isinstance(met, str): break
        truth[met] = file['concentration'][i]

    truth['Mac'] = truth.pop('MMBL')  # rename key MMBL to Mac
    return dict(sorted(truth.items()))