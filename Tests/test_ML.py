#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Tests du module de machine learning.
"""

import sys
import os
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

from Machine_learning.machine_learning import (
    modifie_dataframe, visualisation_apprentissage, affichage_configuration
)
import pandas as pd
import pickle


def test_modifie_dataframe():
    """Teste la fonction modifie_dataframe."""
    data_test = {"prix_euro" : [250., 29.99, 32.99]}
    data_initial = pd.DataFrame(data_test)
    data_final = {"prix_euro" : [0, 1, 2]}
    data_final = pd.DataFrame(data_final, dtype="int32")
    assert modifie_dataframe(data_initial).equals(data_final)


def test_visualisation_apprentissage():
    """Teste la fonction visualisation_apprentissage."""
    with open(path + "\Rendu_final\Resultats_apprentissage.pkl", "rb") as fichier_modele:
        resultats_apprentissage = pickle.load(fichier_modele)
    assert visualisation_apprentissage(resultats_apprentissage) == None

    
def test_affichage_configuration():
    """Teste la fonction affichage_configuration."""
    chemin = path + "\Rendu_final\config.yaml"
    assert affichage_configuration(chemin) == None