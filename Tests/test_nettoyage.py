#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Tests du module de nettoyage de données.
"""

import sys
import os
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

from Nettoyage.nettoyage import (
    _nettoie_marque, _nettoie_prix, _nettoie_serie, _nettoie_couleur, _nettoie_taille_ecran,
    _nettoie_capacite_stockage, _nettoie_modele, _nettoie_megapixel, _nettoie_resolution_ecran,
    _nettoie_date_de_sortie, _nettoie_memoire, _nettoie_vitesse_processeur,
    _nettoie_port_carte_SD, _nettoie_pliable, _nettoie_poids, _nettoie_hauteur,
    _nettoie_largeur, _nettoie_profondeur, _nettoie_lien, nettoie_donnees
)
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def Data():
    data = {
        "etat": ["État correct", "Très bon état", "Parfait état", "Parfait état"],
        "marque" : ["", "SAMSUNG", "croscall", "xiaiomi"],
        "prix" : ["Déjà vendu", "28,25\u20Ac", "50\xa0", "84.23\u202f4"],
        "serie" : ["", "GALAXY", "GaLAxY", "IPhone"],
        "couleur" : ["", "(product)RED", "bleu", "GriS"],
        "taille_ecran" : ["7,20", "6,25", "6.45", "6"],
        "capacite_stockage" : ["28", "126,2", "32 Go", "45"],
        "modele" : ["", "(GAlaxy s20)", "Galaxy s20 Plus", "Galaxy s20+"],
        "megapixel" : ["", "20MP", " 20+40+2MP", "16"],
        "systeme_exploitation": ["IOS", "Android", "IOS", "Android"],
        "resolution_ecran" : ["6.42 inches", "2800x1500pixels", " 800*1500", "full hd"],
        "reseau": ["5G", "5G", "4G", "2G"],
        "date_de_sortie" : ["2007", "25 Mars 2008", "Août 2010", "Décembre 2020"],
        "memoire" : ["", "beaucoup", "", "16"],
        "connecteur": ["", "Jack", "Jack", "Jack"],
        "double_sim": ["Oui", "Oui", "Non", "Non"],
        "vitesse_processeur" : ["", "", "200 GHz", "20,5"],
        "port_carte_SD" : ["", "", " Oui", "Non"],
        "pliable" : ["", "", " ", "Oui"],
        "poids" : ["200", "200 g", "2500", "180"],
        "hauteur" : ["", "20 cm", "18,6", "22"],
        "largeur" : ["", "20 cm", "18,6", "22"],
        "profondeur" : ["", "20 cm", "18,6", "22"],
        "lien" : ["https:", "http", "", "https"]
    }
    return pd.DataFrame(data)


def test_nettoie_marque(Data):
    """Teste la fonction _nettoie_marque."""
    data_test = Data["marque"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"marque" : ["", "samsung", "crosscall", "xiaomi"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_marque(data_initial).equals(data_final)


def test_nettoie_prix(Data):
    """Teste la fonction _nettoie_prix."""
    data_test = Data["prix"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"prix_euro" : [28.25, 50, 84.23]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_prix(data_initial).equals(data_final)


def test_nettoie_serie(Data):
    """Teste la fonction _nettoie_serie."""
    data_test = Data["serie"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"serie" : ["", "galaxy", "galaxy", "iphone"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_serie(data_initial).equals(data_final)


def test_nettoie_couleur(Data):
    """Teste la fonction _nettoie_couleur."""
    data_test = Data["couleur"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"couleur" : ["", "rouge", "bleu", "gris"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_couleur(data_initial).equals(data_final)


def test_nettoie_taille_ecran(Data):
    """Teste la fonction _nettoie_taille_ecran."""
    data_test = Data["taille_ecran"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"taille_ecran_pouce" : [7.20, 6.25, 6.45, 6.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_taille_ecran(data_initial).equals(data_final)


def test_nettoie_capacite_stockage(Data):
    """Teste la fonction _nettoie_capacite_stockage."""
    data_test = Data["capacite_stockage"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"capacite_stockage_Go" : [28., 126.2, 32., 45.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_capacite_stockage(data_initial).equals(data_final)


def test_nettoie_modele(Data):
    """Teste la fonction _nettoie_modele."""
    data_test = Data["modele"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"modele" : ["", "galaxy s20", "galaxy s20 plus", "galaxy s20 plus"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_modele(data_initial).equals(data_final)
    
    
def test_nettoie_megapixel(Data):
    """Teste la fonction _nettoie_megapixel."""
    data_test = Data["megapixel"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"megapixel" : ["", "20", "20/40/2", "16"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_megapixel(data_initial).equals(data_final)


def test_nettoie_resolution_ecran(Data):
    """Teste la fonction _nettoie_resolution_ecran."""
    data_test = Data["resolution_ecran"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"resolution_ecran" : ["", "1500x2800", "800x1500", "1080x1920"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_resolution_ecran(data_initial).equals(data_final)


def test_nettoie_date_de_sortie(Data):
    """Teste la fonction _nettoie_date_de_sortie."""
    data_test = Data["date_de_sortie"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"date_de_sortie" : ["2007", "2008", "2010", "2020"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_date_de_sortie(data_initial).equals(data_final)


def test_nettoie_memoire(Data):
    """Teste la fonction _nettoie_memoire."""
    data_test = Data["memoire"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"memoire" : [np.NaN, np.NaN, np.NaN, 16.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_memoire(data_initial).equals(data_final)


def test_nettoie_vitesse_processeur(Data):
    """Teste la fonction _nettoie_vitesse_processeur."""
    data_test = Data["vitesse_processeur"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"vitesse_processeur_GHz" : [np.NaN, np.NaN, 200., 20.5]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_vitesse_processeur(data_initial).equals(data_final)


def test_nettoie_port_carte_SD(Data):
    """Teste la fonction _nettoie_port_carte_SD."""
    data_test = Data["port_carte_SD"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"port_carte_SD" : ["Non", "Non", "Oui", "Non"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_port_carte_SD(data_initial).equals(data_final)


def test_nettoie_pliable(Data):
    """Teste la fonction _nettoie_pliable."""
    data_test = Data["pliable"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"pliable" : ["Non", "Non", "Non", "Oui"]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_pliable(data_initial).equals(data_final)


def test_nettoie_poids(Data):
    """Teste la fonction _nettoie_poids."""
    data_test = Data["poids"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"poids_g" : [200, 200, 169, 180]}
    data_final = pd.DataFrame(data_final, dtype="int32")
    assert _nettoie_poids(data_initial).equals(data_final)


def test_nettoie_hauteur(Data):
    """Teste la fonction _nettoie_hauteur."""
    data_test = Data["hauteur"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"hauteur_cm" : [np.NaN, 20., 18.6, 22.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_hauteur(data_initial).equals(data_final)


def test_nettoie_largeur(Data):
    """Teste la fonction _nettoie_largeur."""
    data_test = Data["largeur"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"largeur_cm" : [np.NaN, 20., 18.6, 22.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_largeur(data_initial).equals(data_final)


def test_nettoie_profondeur(Data):
    """Teste la fonction _nettoie_profondeur."""
    data_test = Data["profondeur"]
    data_initial = pd.DataFrame(data_test)
    data_final = {"profondeur_cm" : [np.NaN, 20., 18.6, 22.]}
    data_final = pd.DataFrame(data_final)
    assert _nettoie_profondeur(data_initial).equals(data_final)


def test_nettoie_lien(Data):
    """Teste la fonction _nettoie_lien."""
    data_test = Data["lien"]
    data_initial = pd.DataFrame(data_test)
    data_final = pd.DataFrame(columns=[], index=[0, 1, 2, 3])
    assert _nettoie_lien(data_initial).equals(data_final)


def test_nettoie_donnees(Data):
    """Teste la fonction nettoie_donnees."""
    data_initial = Data
    data_final = {
        "etat": ["Parfait état"],
        "marque" : ["xiaomi"],
        "prix_euro" : [84.23],
        "couleur" : ["gris"],
        "taille_ecran_pouce" : [6.0],
        "capacite_stockage_Go" : [45.0],
        "megapixel" : ["16"],
        "systeme_exploitation": ["Android"],
        "resolution_ecran" : ["1080x1920"],
        "reseau": ["2G"],
        "date_de_sortie" : ["2020"],
        "memoire" : [16.0],
        "connecteur": ["Jack"],
        "double_sim": ["Non"],
        "port_carte_SD" : ["Non"],
        "pliable" : ["Oui"],
        "poids_g" : [180]
    }
    data_final = pd.DataFrame(data_final)
    data_final["poids_g"] = data_final["poids_g"].astype("int32")
    assert nettoie_donnees(data_initial).equals(data_final)