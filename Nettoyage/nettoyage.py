#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Fonctions nettoyant la base de données.

Fonctions principales :
- nettoie_donnees
- recupere_bases_de_donnees_complete
- sauvegarde_csv

Fonctions secondaires :
- _nettoie_marque
- _nettoie_prix
- _nettoie_serie
- _nettoie_couleur
- _nettoie_taille_ecran
- _nettoie_capacite_stockage
- _nettoie_modele
- _nettoie_megapixel
- _nettoie_resolution_ecran
- _nettoie_date_de_sortie
- _nettoie_memoire
- _nettoie_vitesse_processeur
- _nettoie_port_carte_SD
- _nettoie_pliable
- _nettoie_poids
- _nettoie_hauteur
- _nettoie_largeur
- _nettoie_profondeur
- _nettoie_lien
"""

from serde import deserialize, serialize
from serde.json import from_json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
import os
import warnings


@serialize
@deserialize
@dataclass
class Annonce:
    """Dataclass permettant de récupérer les données sous une forme déjà propre."""
    etat : str
    marque : str
    prix : str
    serie : str
    couleur : str
    taille_ecran : str
    capacite_stockage : str
    modele : str
    megapixel : str
    systeme_exploitation : str
    resolution_ecran : str
    reseau : str
    date_de_sortie : str
    memoire : str
    vitesse_processeur : str
    nombre_coeur : str
    connecteur : str
    double_sim : str
    port_carte_SD : str
    pliable : str
    reseau_5G : str
    appareil_photo : str
    poids : str
    hauteur : str
    largeur : str
    profondeur : str
    lien : str


def recupere_bases_de_donnees_complete(chemin_donnees: str) -> pd.DataFrame:
    """
    Récupère et combine les 2 bases de données, en enlevant les données dupliquées.
    
    Exemple :
>>> from Nettoyage recupere_bases_de_donnees_complete
>>> import os
>>> path = os.path.abspath(os.getcwd())
>>> recupere_bases_de_donnees_complete(path + "\Donnees")
    etat	        marque	    prix	    serie	couleur
0	État correct		        Déjà vendu	
1	Très bon état	SAMSUNG	    28,25€	    GALAXY	(product)RED
2	Parfait état	croscall	50	        GaLAxY	bleu
3	Parfait état	xiaiomi	    84.234	    IPhone	GriS
    """
    with open(chemin_donnees + "\data_complete_1.json", "r") as fichier:
        data_frame_1 = fichier.read()
    with open(chemin_donnees + "\data_complete_2.json", "r") as fichier:
        data_frame_2 = fichier.read()

    annonces_1 = from_json(List[Annonce], data_frame_1)
    annonces_2 = from_json(List[Annonce], data_frame_2)
    data_1 = pd.DataFrame(annonces_1)
    data_2 = pd.DataFrame(annonces_2)
    data = pd.concat([data_1, data_2], ignore_index=True)
    for i in range(0, data.shape[1]-1):
        data.iloc[:,i] = data.iloc[:,i].str.replace("^ ", "", regex=True)
    return data.drop_duplicates(ignore_index=True)
      

def nettoie_donnees(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction nettoyant la base de données complète.
    
    Exemple :
>>> from Machine_learning import nettoie_donnees
>>> import pandas as pd
>>> data = {
...        "etat": ["État correct", "Très bon état", "Parfait état", "Parfait état"],
...        "marque" : ["", "SAMSUNG", "croscall", "xiaiomi"],
...        "prix" : ["Déjà vendu", "28,25\u20Ac", "50\xa0", "84.23\u202f4"],
...        "serie" : ["", "GALAXY", "GaLAxY", "IPhone"],
...        "couleur" : ["", "(product)RED", "bleu", "GriS"],
...        "taille_ecran" : ["7,20", "6,25", "6.45", "6"],
...        "capacite_stockage" : ["28", "126,2", "32 Go", "45"],
...        "modele" : ["", "(GAlaxy s20)", "Galaxy s20 Plus", "Galaxy s20+"],
...        "megapixel" : ["", "20MP", " 20+40+2MP", "16"],
...        "systeme_exploitation": ["IOS", "Android", "IOS", "Android"],
...        "resolution_ecran" : ["6.42 inches", "2800x1500pixels", " 800*1500", "full hd"],
...        "reseau": ["5G", "5G", "4G", "2G"],
...        "date_de_sortie" : ["2007", "25 Mars 2008", "Août 2010", "Décembre 2020"],
...        "memoire" : ["", "beaucoup", "", "16"],
...        "connecteur": ["", "Jack", "Jack", "Jack"],
...        "double_sim": ["Oui", "Oui", "Non", "Non"],
...        "vitesse_processeur" : ["", "", "200 GHz", "20,5"],
...        "port_carte_SD" : ["", "", " Oui", "Non"],
...        "pliable" : ["", "", " ", "Oui"],
...        "poids" : ["200", "200 g", "2500", "180"],
...        "hauteur" : ["", "20 cm", "18,6", "22"],
...        "largeur" : ["", "20 cm", "18,6", "22"],
...        "profondeur" : ["", "20 cm", "18,6", "22"],
...        "lien" : ["https:", "http", "", "https"]
...    }
>>> data = pd.DataFrame(data)
>>> nettoie_donnees(data)
    etat	        marque	prix_euro	couleur	taille_ecran_pouce	capacite_stockage_Go	megapixel	systeme_exploitation	resolution_ecran	reseau	date_de_sortie	memoire	connecteur	double_sim	port_carte_SD	pliable	  poids_g
0	Parfait état	xiaomi	84.23	    gris	 6.0	             45.0	                 16	                  Android	              1080x1920	          2G	  2020	          16.0	   Jack	       Non	       Non	           Oui	     180
    """
    data = _nettoie_marque(data)
    data = _nettoie_prix(data)
    data = _nettoie_serie(data)
    data = _nettoie_couleur(data)
    data = _nettoie_taille_ecran(data)
    data = _nettoie_capacite_stockage(data)
    data = _nettoie_modele(data)
    data = _nettoie_megapixel(data)
    data = _nettoie_resolution_ecran(data)
    data = _nettoie_date_de_sortie(data)
    data = _nettoie_memoire(data)
    data = _nettoie_vitesse_processeur(data)
    data = _nettoie_port_carte_SD(data)
    data = _nettoie_pliable(data)
    data = _nettoie_poids(data)
    data = _nettoie_hauteur(data)
    data = _nettoie_largeur(data)
    data = _nettoie_profondeur(data)
    data = _nettoie_lien(data)
    data = data.drop_duplicates(ignore_index=True)
    data = data.replace("", np.NaN)
    data_final = data.loc[:,
     [
         "etat", "marque", "prix_euro", "couleur", "taille_ecran_pouce", "capacite_stockage_Go", "modele", 
         "megapixel", "systeme_exploitation", "resolution_ecran", "reseau", "date_de_sortie", "memoire", 
         "connecteur", "double_sim", "port_carte_SD", "pliable", "poids_g"
     ]
    ]
    data_final = data_final.dropna().drop_duplicates(ignore_index=True)
    return data_final.drop(axis=1, columns="modele")


def sauvegarde_csv(chemin_donnees: str, data_final: pd.DataFrame) -> str:
    """
    Sauvegarde la base de données dans le dossier Donnees.
    
    Exemple :
>>> from Nettoyage recupere_bases_de_donnees_complete, nettoie_donnees, sauvegarde_csv
>>> import os
>>> path = os.path.abspath(os.getcwd())
>>> data = recupere_bases_de_donnees_complete(path + "\Donnees")
>>> data_final = nettoie_donnees(data)
>>> sauvegarde_csv(path+"\Donnees", data_final)
    """
    data_final.to_csv(chemin_donnees + "\complete_database.csv")
    return "Base de données sauvegardée."
        

def _nettoie_marque(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable marque.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_marque
>>> import pandas as pd
>>> data = pd.DataFrame(
...     {'marque' : ['', 'SAMSUNG', 'croscall', 'xiaiomi']}
... )
>>> _nettoie_marque(data)
	marque
0	
1	samsung
2	crosscall
3	xiaomi
    """
    data["marque"] = data["marque"].str.lower()
    data["marque"] = data["marque"].str.replace("croscall", "crosscall").replace("xiaiomi", "xiaomi")
    return data


def _nettoie_prix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable prix.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_prix
>>> import pandas as pd
>>> data = pd.DataFrame(
...     {'prix' : ['Déjà vendu', '28,25\u20Ac', '50\xa0', '84.23\u202f4']}
... )
>>> _nettoie_prix(data)
    prix_euro
0	28.25
1	50.00
2	84.23
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = data[data.loc[:, "prix"] != "Déjà vendu"]
        data = data.drop_duplicates(ignore_index=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].str.replace(",", ".").replace("\u20AC", "", regex=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].str.replace("\xa0", "").replace("\u202f1", "", regex=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].str.replace("\u202f0", "", regex=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].str.replace("\u202f2", "", regex=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].str.replace("\u202f3", "", regex=True)
        data.loc[:, "prix"] = data.loc[:, "prix"].replace("\u202f4", "", regex=True).astype("float")
        data = data.rename(columns={"prix":"prix_euro"})
    return data


def _nettoie_serie(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable serie.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_serie
>>> import pandas as pd
>>> data = pd.DataFrame(
...     {"serie" : ["", "GALAXY", "GaLAxY", "IPhone"]}
... )
>>> _nettoie_serie(data)
    serie
0	
1	galaxy
2	galaxy
3	iphone
    """
    data["serie"] = data["serie"].str.lower()
    return data
    

def _nettoie_couleur(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable couleur.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_couleur
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "couleur" : ["", "(product)RED", "bleu", "GriS"]
... )
>>> _nettoie_couleur(data)
	couleur
0	
1	rouge
2	bleu
3	gris
    """
    data["couleur"] = data["couleur"].str.lower().replace("(product)red", "rouge")
    return data


def _nettoie_taille_ecran(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable taille_ecran.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_taille_ecran
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "taille_ecran" : ["7,20", "6,25", "6.45", "6"]
... )
>>> _nettoie_taille_ecran(data)
	taille_ecran_pouce
0	7.20
1	6.25
2	6.45
3	6.00
    """
    data["taille_ecran"] = data["taille_ecran"].str.replace(",", ".").astype("float")
    data = data.rename(columns={"taille_ecran":"taille_ecran_pouce"})
    return data


def _nettoie_capacite_stockage(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable capacite_stockage.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_capacite_stockage
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "capacite_stockage" : ["28", "126,2", "32 Go", "45"]
... )
>>> _nettoie_capacite_stockage(data)
	capacite_stockage_Go
0	28.0
1	126.2
2	32.0
3	45.0
    """
    data = data.rename(columns={"capacite_stockage":"capacite_stockage_Go"})
    data["capacite_stockage_Go"] = data["capacite_stockage_Go"].str.replace(",", ".")
    data["capacite_stockage_Go"] = data["capacite_stockage_Go"].str.replace(" Go", "").astype("float")
    return data


def _nettoie_modele(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable modele.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_modele
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "modele" : ["", "(GAlaxy s20)", "Galaxy s20 Plus", "Galaxy s20+"]
... )
>>> _nettoie_modele(data)
	modele
0	
1	galaxy s20
2	galaxy s20 plus
3	galaxy s20 plus
    """
    data["modele"] = data["modele"].str.replace("\(", "", regex=True).replace("\)", "", regex=True)
    data["modele"] = data["modele"].str.replace("\+", " plus", regex=True)
    data["modele"] = data["modele"].str.lower()
    return data


def _nettoie_megapixel(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable megapixel.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_megapixel
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "megapixel" : ["", "20MP", " 20+40+2MP", "16"]
... )
>>> _nettoie_megapixel(data)
	megapixel
0	
1	20
2	20/40/2
3	16
    """
    data["megapixel"] = data["megapixel"].str.replace("MP", "").replace("\s", "", regex=True)
    data["megapixel"] = data["megapixel"].replace("\+", "/", regex=True)
    return data


def _nettoie_resolution_ecran(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable resolution_ecran.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_resolution_ecran
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "resolution_ecran" : ["6.42 inches", "2800x1500pixels", " 800*1500", "full hd"]
... )
>>> _nettoie_resolution_ecran(data)
	resolution_ecran
0	
1	1500x2800
2	800x1500
3	1080x1920
    """
    data["resolution_ecran"] = data["resolution_ecran"].str.lower().replace("full hd", "1920x1080")
    data["resolution_ecran"] = data["resolution_ecran"].str.replace("pixels", "")
    data["resolution_ecran"] = data["resolution_ecran"].str.replace("\*", "x", regex=True).replace("\×", "x", regex=True)
    data["resolution_ecran"] = data["resolution_ecran"].str.replace("6.42 inches", "", regex=True)
    data["resolution_ecran"] = data["resolution_ecran"].str.replace("\s", "", regex=True)
    for i in data.index:
        resolution = data.loc[i,"resolution_ecran"]
        if resolution != "":
            resolution_1, resolution_2 = resolution.split("x")
            if int(resolution_1) > int(resolution_2):
                data.loc[i,"resolution_ecran"] = data.loc[i,"resolution_ecran"].replace(resolution, f"{resolution_2}x{resolution_1}")
    return data


def _nettoie_date_de_sortie(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable date_de_sortie.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_date_de_sortie
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "date_de_sortie" : ["2007", "25 Mars 2008", "Août 2010", "Décembre 2020"]
... )
>>> _nettoie_date_de_sortie(data)
	date_de_sortie
0	2007
1	2008
2	2010
3	2020
    """
    data["date_de_sortie"] = data["date_de_sortie"].str.replace("25", "").replace("[A-Z, a-z, é, û]", "", regex=True)
    return data


def _nettoie_memoire(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable memoire.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_memoire
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "memoire" : ["", "beaucoup", "", "16"]
... )
>>> _nettoie_memoire(data)
	memoire
0	NaN
1	NaN
2	NaN
3	16.0
    """
    data["memoire"] = data["memoire"].str.replace("[A-Z, a-z]", "", regex=True).replace("", np.NaN).astype("float")
    return data


def _nettoie_vitesse_processeur(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable vitesse_processeur.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_vitesse_processeur
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "vitesse_processeur" : ["", "", "200 GHz", "20,5"]
... )
>>> _nettoie_vitesse_processeur(data)
	vitesse_processeur_GHz
0	NaN
1	NaN
2	200.0
3	20.5
    """
    data["vitesse_processeur"] = data["vitesse_processeur"].str.replace(",", ".").replace(" GHz", "", regex=True)
    data["vitesse_processeur"] = data["vitesse_processeur"].replace("", np.NaN).astype("float")
    data = data.rename(columns={"vitesse_processeur":"vitesse_processeur_GHz"})
    return data


def _nettoie_port_carte_SD(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable port_carte_SD.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_port_carte_SD
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "port_carte_SD" : ["", "", " Oui", "Non"]
... )
>>> _nettoie_port_carte_SD(data)
	port_carte_SD
0	Non
1	Non
2	Oui
3	Non
    """
    data["port_carte_SD"] = data["port_carte_SD"].str.replace(" ", "").replace("", "Non")
    return data


def _nettoie_pliable(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable pliable.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_pliable
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "pliable" : ["", "", " ", "Oui"]
... )
>>> _nettoie_pliable(data)
	pliable
0	Non
1	Non
2	Non
3	Oui
    """
    data["pliable"] = data["pliable"].str.replace(" ", "").replace("", "Non")
    return data


def _nettoie_poids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable poids.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_poids
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "poids" : ["200", "200 g", "2500", "180"]
... )
>>> _nettoie_poids(data)
	poids_g
0	200
1	200
2	169
3	180
    """
    data["poids"] = data["poids"].str.replace(" g", "").astype("int32")
    data["poids"] = data["poids"].replace(2500, 169)
    data = data.rename(columns={"poids":"poids_g"})
    return data


def _nettoie_hauteur(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable hauteur.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_hauteur
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "hauteur" : ["", "20 cm", "18,6", "22"]
... )
>>> _nettoie_hauteur(data)
	hauteur_cm
0	NaN
1	20.0
2	18.6
3	22.0
    """
    data["hauteur"] = data["hauteur"].str.replace(" cm", "").replace(",", ".", regex=True)
    data["hauteur"] = data["hauteur"].replace("", np.NaN).astype("float")
    data = data.rename(columns={"hauteur":"hauteur_cm"})
    return data


def _nettoie_largeur(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable largeur.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_largeur
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "largeur" : ["", "20 cm", "18,6", "22"]
... )
>>> _nettoie_largeur(data)
	largeur_cm
0	NaN
1	20.0
2	18.6
3	22.0
    """
    data["largeur"] = data["largeur"].str.replace(" cm", "").replace(",", ".", regex=True).replace("", np.NaN).astype("float")
    data = data.rename(columns={"largeur":"largeur_cm"})
    return data


def _nettoie_profondeur(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable profondeur.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_profondeur
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "profondeur" : ["", "20 cm", "18,6", "22"]
... )
>>> _nettoie_profondeur(data)
	profondeur_cm
0	NaN
1	20.0
2	18.6
3	22.0
    """
    data["profondeur"] = data["profondeur"].str.replace(" cm", "").replace(",", ".", regex=True).replace("", np.NaN).astype("float")
    data = data.rename(columns={"profondeur":"profondeur_cm"})
    return data


def _nettoie_lien(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la variable lien.
    
    Exemple :
>>> from Nettoyage.nettoyage import _nettoie_lien
>>> import pandas as pd
>>> data = pd.DataFrame(
...     "lien" : ["https:", "http", "", "https"]
... )
>>> _nettoie_lien(data)

0
1
2
3
    """
    data = data.drop(columns="lien")
    return data