#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Exemple :
>>> from Machine_learning import (
... genere_configuration, chargement_configuration, affichage_configuration, 
... apprentissage_modele, modifie_dataframe, visualisation_apprentissage,
... teste_precision_modele, meilleur_modele, matrice_confusion_donnees_test, 
... renvoie_echantillons_test_train
... )
"""

from .machine_learning import (
    genere_configuration, chargement_configuration, affichage_configuration, 
    apprentissage_modele, modifie_dataframe, visualisation_apprentissage,
    teste_precision_modele, matrice_confusion_donnees_test, renvoie_echantillons_test_train
)

__all__ = [
    "genere_configuration", "chargement_configuration", "affichage_configuration", 
    "apprentissage_modele", "modifie_dataframe", "visualisation_apprentissage",
    "teste_precision_modele", "matrice_confusion_donnees_test", "renvoie_echantillons_test_train"
]