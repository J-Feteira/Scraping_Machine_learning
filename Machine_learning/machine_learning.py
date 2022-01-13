#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Script permettant de planifier un entrainement par crossvalidation via un fichier de configuration en yaml.

Fonctions principales :
- genere_configuration
- chargement_configuration
- affichage_configuration
- modifie_dataframe
- renvoie_echantillons_test_train
- apprentissage_modele
- visualisation_apprentissage
- teste_precision_modele
- matrice_confusion_donnees_test

Dataclass :
- Arbre
- Bayesien
- Strategie
- Benet
- Foret
- KVoisins
- Neurones
- SupportVecteurs
- Config
"""
from dataclasses import dataclass
from enum import Enum
from rich import print
from serde import serialize, deserialize
from serde.yaml import to_yaml, from_yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Optional, List, Dict, Union
import pandas as pd
import numpy as np
import typer
import warnings
import os
import pickle
import time
from rich.table import Table
from rich import print


ModelRegressor = Union[
    DummyClassifier, 
    RandomForestClassifier,  
    MultinomialNB,
    KNeighborsClassifier,
    MLPClassifier,
    SVC,
    DecisionTreeClassifier
]

@serialize
@deserialize
@dataclass
class Arbre:
    """Configuration pour l'arbre de décision."""
    config_vide: bool = True
    longueur_arbre: Optional[List[int]] = None
    nombre_noeuds: Optional[List[int]] = None
    parametre_complexite: Optional[List[int]] = None

        
@serialize
@deserialize
@dataclass
class Bayesien:
    """Configuration pour le modèle Bayésien."""
    config_vide: bool = True
    alpha: Optional[List[int]] = None

        
class Strategie(Enum):
    """Définit les stratégies pour le DummyClassifier."""
    PRIOR = "prior"
    UNIFORM = "uniform"
    STRATIFIED = "stratified"


@serialize
@deserialize
@dataclass
class Benet:
    """Configuration pour le modèle DummyClassifier."""
    config_vide: bool = True
    strategies: Optional[List[Strategie]] = None


@serialize
@deserialize
@dataclass
class Foret:
    """Configuration pour la forêt aléatoire."""
    config_vide: bool = True
    longueur_arbre: Optional[List[int]] = None
    nombre_estimateurs: Optional[List[int]] = None


@serialize
@deserialize
@dataclass
class KVoisins:
    """Configuration pour les k plus proches voisins."""
    config_vide: bool = True
    nombre_voisins: Optional[List[int]] = None
    type_distance: Optional[List[int]] = None


@serialize
@deserialize
@dataclass
class Neurones:
    """Configuration pour les réseaux de neurones."""
    config_vide: bool = True
    architecture: Optional[List[List[int]]] = None
    alpha: Optional[List[int]] = None
    max_iter: Optional[List[int]] = None


@serialize
@deserialize
@dataclass
class SupportVecteurs:
    """Configuration pour le support vecteur."""
    config_vide: bool = True
    regularisation: Optional[List[float]] = None


@serialize
@deserialize
@dataclass
class Config:
    """Classe de configuration de tous les modèles."""
    arbre_decision: Optional[Arbre] = None
    bayesien: Optional[Bayesien] = None
    benet: Optional[Benet] = None
    foret: Optional[Foret] = None
    k_voisins: Optional[KVoisins] = None
    neurones: Optional[Neurones] = None
    support_vecteurs: Optional[SupportVecteurs] = None


app = typer.Typer()


@app.command()
def genere_configuration(nom_fichier: str = "config.yaml") -> None:
    """
    Genere un fichier de configuration par défaut.
    
    Exemple :
>>> from Machine_learning import genere_configuration
>>> genere_configuration('config.yaml')
    """
    config_modeles = Config(
        arbre_decision=Arbre(
            config_vide=False, 
            longueur_arbre=[1, 3, 5, 7],
            nombre_noeuds=[2, 4, 6],
            parametre_complexite=[0., 1., 2.]
        ),
        bayesien=Bayesien(
            config_vide=False, 
            alpha=[0.001, 0.01, 0.1, 1.]
        ),
        benet=Benet(
            config_vide=False, 
            strategies=[
                Strategie.PRIOR, Strategie.STRATIFIED, Strategie.UNIFORM
            ]
        ),
        foret=Foret(
            config_vide=False, 
            longueur_arbre=[1, 5, 10, 25],
            nombre_estimateurs=[10, 100, 250, 400]
        ),
        k_voisins=KVoisins(
            config_vide=False, 
            nombre_voisins=[1, 5, 11, 20],
            type_distance=[1, 2]
        ),
        neurones=Neurones(
            config_vide=False, 
            alpha=[0.001, 0.1],
            architecture=[[100,], [50, 50,]],
            max_iter=[500, 1000, 10000]
        ),
        support_vecteurs=SupportVecteurs(
            config_vide=False, 
            regularisation=[0.1, 1.0, 5.0]
        ),
    )
    
    with open(nom_fichier, "w") as fichier:
        fichier.write(to_yaml(config_modeles))


def chargement_configuration(nom_fichier: str) -> Config:
    """
    Chargement du fichier de configuration.
    
    Exemple :
>>> from Machine_learning import chargement_configuration
>>> chargement_configuration('config.yaml')

Config(arbre_decision=Arbre(config_vide=False, longueur_arbre=[1, 3, 5, 7], nombre_noeuds=[2, 4, 6], parametre_complexite=[0.0, 1.0, 2.0]), bayesien=Bayesien(config_vide=False, alpha=[0.001, 0.01, 0.1, 1.0]), benet=Benet(config_vide=False, strategies=[<Strategie.PRIOR: 'prior'>, <Strategie.STRATIFIED: 'stratified'>, <Strategie.UNIFORM: 'uniform'>]), foret=Foret(config_vide=False, longueur_arbre=[1, 5, 10, 25], nombre_estimateurs=[10, 100, 250, 400]), k_voisins=KVoisins(config_vide=False, nombre_voisins=[1, 5, 11, 20], type_distance=[1, 2]), neurones=Neurones(config_vide=False, architecture=[[100], [50, 50]], alpha=[0.001, 0.1], max_iter=[500, 1000, 10000]), support_vecteurs=SupportVecteurs(config_vide=False, regularisation=[0.1, 1.0, 5.0]))
    """
    with open(nom_fichier, "r") as fichier:
        data = fichier.read()

    config = from_yaml(Config, data)
    return config


@app.command()
def affichage_configuration(nom_fichier: str) -> None:
    """
    Affiche l'objet généré par le fichier de configuration sans apprentissage.
    
    Exemple :
>>> from Machine_learning import affichage_configuration
>>> affichage_configuration("config.yaml")
Config(
    arbre_decision=Arbre(
        config_vide=False,
        longueur_arbre=[1, 3, 5, 7],
        nombre_noeuds=[2, 4, 6],
        parametre_complexite=[0.0, 1.0, 2.0]
    ),
    bayesien_naif=Bayesien(config_vide=False, alpha=[0.001, 0.01, 0.1, 1.0]),
    benet=Benet(
        config_vide=False,
        strategies=[
            <Strategie.PRIOR: 'prior'>,
            <Strategie.STRATIFIED: 'stratified'>,
            <Strategie.UNIFORM: 'uniform'>
        ]
    ),
    foret=Foret(
        config_vide=False,
        longueur_arbre=[1, 5, 10, 25],
        nombre_estimateurs=[10, 100, 250, 400]
    ),
    k_voisins=KVoisins(
        config_vide=False,
        nombre_voisins=[1, 5, 11, 20],
        type_distance=[1, 2]
    ),
    neurones=Neurones(
        config_vide=False,
        architecture=[[100], [50, 50]],
        alpha=[0.001, 0.1],
        max_iter=[500, 1000, 10000]
    ),
    support_vecteurs=SupportVecteurs(config_vide=False, regularisation=[0.1, 1.0, 5.0])
)
    """
    config = chargement_configuration(nom_fichier)
    print(config)
    

def modifie_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Modifie le DataFrame pour qu'il soit utilisable par la fonction apprentissage.
    
    Exemple :
>>> from Machine_learning import modifie_dataframe
>>> import pandas as pd
>>> data = pd.DataFrame(
...     {
...         "prix_euro": [50, 200, 1000],
...         "marque" : ["Samsung", "Apple", "Huawei"]
...     }
... )
>>> modifie_dataframe(data)
   prix_euro	marque_Apple	marque_Huawei	marque_Samsung
0  1	        0	            0	            1
1  0	        1	            0	            0
2  2	        0	            1	            0
    """
    data_dummy = pd.get_dummies(data)
    data_dummy["prix_euro"] = pd.qcut(data_dummy["prix_euro"], q=10)
    data_dummy["prix_euro"] = data_dummy["prix_euro"].astype("string")
    labelencoder_prix = LabelEncoder()
    data_dummy["prix_euro"] = labelencoder_prix.fit_transform(data_dummy["prix_euro"])
    return data_dummy


def renvoie_echantillons_test_train(data_dummy: pd.DataFrame) -> np.ndarray:
    """
    Fonction renvoyant les échantillons des données test et train.
    
    Exemple :
>>> import os
>>> from Machine_learning import modifie_dataframe, renvoie_echantillons_test_train
>>> import pandas as pd
>>> path = os.path.abspath(os.getcwd())
>>> data = pd.read_csv(path + "\Donnees\complete_database.csv", index_col=0)
>>> data_dummy = modifie_dataframe(data)
>>> renvoie_echantillons_test_train(data_dummy)
[array([[  7. ,   6.1,  64. , ...,   0. ,   1. ,   0. ],
        [  3. ,   5.8, 128. , ...,   1. ,   1. ,   0. ],
        [  1. ,   4.7,  64. , ...,   0. ,   1. ,   0. ],
        ...,
        [  1. ,   4. , 128. , ...,   0. ,   1. ,   0. ],
        [  9. ,   6.5, 512. , ...,   0. ,   1. ,   0. ],
        [  6. ,   6.5, 128. , ...,   1. ,   1. ,   0. ]]),
 array([[  8. ,   6.5,  64. , ...,   0. ,   1. ,   0. ],
        [  3. ,   6.2,  64. , ...,   1. ,   1. ,   0. ],
        [  6. ,   6.1, 128. , ...,   0. ,   1. ,   0. ],
        ...,
        [  0. ,   6.5,  32. , ...,   1. ,   1. ,   0. ],
        [  9. ,   6.7, 128. , ...,   0. ,   0. ,   1. ],
        [  0. ,   5.2,  32. , ...,   1. ,   1. ,   0. ]]),
 array([7., 3., 1., ..., 1., 9., 6.]),
 array([8., 3., 6., 4., 0., 6., 5., 9., 9., 1., 7., 9., 9., 3., 4., 9., 1.,
        8., 3., 0., 0., 3., 4., 2., 1., 5., 2., 9., 4., 2., 6., 4., 4., 8.,
        4., 0., 0., 9., 7., 5., 3., 7., 1., 3., 3., 2., 6., 7., 4., 7., 2.,
        6., 1., 2., 0., 8., 5., 3., 2., 1., 5., 0., 4., 8., 8., 5., 6., 5.,
        5., 6., 2., 4., 4., 6., 8., 8., 5., 6., 4., 8., 9., 0., 4., 5., 7.,
        9., 6., 1., 8., 7., 5., 3., 7., 4., 4., 3., 4., 6., 6., 2., 2., 6.,
        9., 3., 9., 5., 1., 6., 9., 6., 6., 3., 8., 7., 7., 0., 3., 3., 1.,
        9., 2., 3., 5., 5., 4., 7., 7., 1., 0., 0., 9., 3., 1., 0., 1., 4.,
        5., 3., 5., 8., 2., 4., 3., 0., 2., 2., 9., 8., 6., 9., 2., 1., 0.,
        7., 6., 6., 7., 8., 7., 5., 5., 1., 5., 0., 6., 5., 7., 9., 7., 0.,
        8., 3., 6., 4., 5., 9., 5., 8., 7., 2., 8., 1., 6., 2., 8., 6., 4.,
        2., 3., 6., 2., 9., 7., 3., 4., 9., 4., 7., 9., 8., 6., 2., 1., 6.,
        0., 3., 2., 8., 5., 3., 2., 2., 7., 6., 2., 9., 4., 3., 9., 7., 6.,
        2., 4., 4., 1., 7., 2., 8., 0., 2., 9., 0., 4., 2., 7., 8., 2., 4.,
        1., 9., 9., 0., 3., 9., 3., 5., 2., 0., 9., 4., 2., 8., 7., 6., 6.,
        2., 5., 0., 1., 3., 0., 3., 8., 8., 5., 8., 5., 1., 8., 2., 0., 3.,
        2., 6., 7., 9., 8., 4., 2., 5., 6., 7., 9., 5., 6., 8., 5., 3., 1.,
        7., 4., 2., 5., 5., 5., 8., 3., 7., 3., 5., 3., 0., 6., 0., 5., 5.,
        0., 6., 0., 7., 5., 1., 3., 0., 5., 6., 1., 0., 6., 4., 8., 6., 0.,
        6., 8., 8., 5., 2., 3., 8., 5., 1., 0., 2., 7., 6., 6., 1., 3., 4.,
        0., 5., 8., 8., 8., 4., 3., 3., 1., 6., 9., 2., 5., 7., 8., 0., 0.,
        5., 3., 6., 8., 9., 1., 8., 9., 6., 3., 7., 4., 0., 5., 3., 2., 5.,
        6., 3., 6., 3., 5., 0., 8., 8., 8., 8., 8., 2., 5., 1., 2., 3., 1.,
        6., 8., 4., 9., 7., 5., 5., 8., 6., 8., 2., 1., 1., 8., 6., 4., 3.,
        1., 7., 3., 6., 1., 6., 8., 7., 4., 2., 4., 0., 4., 3., 0., 1., 9.,
        5., 2., 0., 6., 6., 4., 1., 3., 3., 9., 1., 6., 2., 5., 1., 2., 5.,
        6., 9., 2., 6., 9., 4., 4., 6., 1., 2., 0., 1., 4., 6., 2., 2., 4.,
        9., 1., 3., 3., 3., 0., 4., 8., 4., 3., 9., 0., 4., 2., 9., 4., 5.,
        6., 8., 9., 0., 3., 9., 6., 6., 1., 2., 0., 7., 6., 6., 2., 1., 8.,
        9., 4., 1., 0., 7., 0., 9., 0., 9., 2., 2., 4., 2., 7., 8., 1., 8.,
        9., 4., 9., 8., 4., 0., 7., 2., 5., 6., 8., 1., 6., 6., 2., 8., 8.,
        1., 5., 1., 4., 2., 9., 7., 7., 9., 0., 3., 6., 1., 4., 9., 3., 6.,
        2., 6., 5., 5., 9., 6., 4., 3., 4., 6., 6., 3., 0., 4., 2., 9., 2.,
        6., 0., 9., 7., 8., 8., 7., 0., 3., 3., 5., 6., 2., 9., 5., 6., 4.,
        3., 7., 6., 4., 5., 6., 8., 8., 6., 9., 1., 8., 7., 7., 4., 1., 4.,
        3., 4., 5., 5., 9., 5., 7., 8., 0., 3., 4., 4., 5., 5., 6., 0., 4.,
        3., 3., 5., 7., 4., 7., 5., 4., 9., 2., 2., 0., 9., 0.])]
    """
    matrice_data = data_dummy.values
    X = matrice_data
    y = matrice_data[:, 0]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)
    return X_tr, X_te, y_tr, y_te


@app.command()
def apprentissage_modele(nom_fichier: str, data_dummy: pd.DataFrame) -> Dict[str, str]:
    """
    Lance un apprentissage paramétré par le fichier de configuration.
    
    Exemple :
>>> import os
>>> from Machine_learning import modifie_dataframe, apprentissage_modele
>>> import pandas as pd
>>> path = os.path.abspath(os.getcwd())
>>> data = pd.read_csv(path + "\Donnees\complete_database.csv", index_col=0)
>>> data_dummy = modifie_dataframe(data)
>>> apprentissage_modele("config.yaml", data_dummy)

Arbre de décision fini.
Bayesien fini.
DummyClassifier fini.
Forêt aléatoire finie.
k-plus proches voisins fini.
Réseaux de neurones fini.
Support vecteur fini.

{DecisionTreeClassifier(max_depth=5, max_leaf_nodes=6): '0.61608 (0.0543 minutes)',
 MultinomialNB(alpha=0.001): '0.3637 (0.0045 minutes)',
 DummyClassifier(): '0.1065 (0.0016 minutes)',
 RandomForestClassifier(max_depth=25, n_estimators=400): '0.85676 (1.1033 minutes)',
 KNeighborsClassifier(n_neighbors=1): '0.77636 (0.098 minutes)',
 MLPClassifier(alpha=0.001, hidden_layer_sizes=[100], max_iter=10000): '0.52288 (3.9902 minutes)',
 SVC(C=5.0): '0.16881 (0.2581 minutes)'}
    """
    config = chargement_configuration(nom_fichier)
    X_tr, X_te, y_tr, y_te = renvoie_echantillons_test_train(data_dummy)
    resultats = dict()
    path = os.path.dirname(os.getcwd())
    
    if config.arbre_decision:
        debut = time.time()
        p = DecisionTreeClassifier()
        if config.arbre_decision.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Arbre_decision.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(
                p, 
                param_grid={
                    "max_depth": config.arbre_decision.longueur_arbre,
                    "max_leaf_nodes": config.arbre_decision.nombre_noeuds,
                    "ccp_alpha": config.arbre_decision.parametre_complexite
                }
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Arbre_decision.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("Arbre de décision fini.")
        
    if config.bayesien:
        debut = time.time()
        p = MultinomialNB()
        if config.bayesien.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Bayesien.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(p, param_grid={"alpha": config.bayesien.alpha})
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Bayesien.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("Bayesien fini.")
        
    if config.benet is not None:
        debut = time.time()
        p = DummyClassifier()
        if config.benet.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\DummyClassifier.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(
                p,
                param_grid={
                    "strategy": [
                        strategie.value for strategie in config.benet.strategies
                    ]
                },
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\DummyClassifier.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("DummyClassifier fini.")
        
    if config.foret is not None:
        debut = time.time()
        p = RandomForestClassifier()
        if config.foret.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Foret_aleatoire.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(
                p, 
                param_grid={
                    "n_estimators": config.foret.nombre_estimateurs,
                    "max_depth": config.foret.longueur_arbre
                }
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Foret_aleatoire.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("Forêt aléatoire finie.")
        
    if config.k_voisins is not None:
        debut = time.time()
        p = KNeighborsClassifier()
        if config.k_voisins.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Knn.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(
                p, param_grid={
                    "n_neighbors": config.k_voisins.nombre_voisins,
                    "p": config.k_voisins.type_distance
                }
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Knn.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("k-plus proches voisins fini.")
            
    if config.neurones is not None:
        debut = time.time()
        p = MLPClassifier()
        if config.neurones.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Reseaux_neurones.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(
                p, param_grid={
                    "hidden_layer_sizes": config.neurones.architecture,
                    "alpha": config.neurones.alpha,
                    "max_iter": config.neurones.max_iter
                }
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\Reseau_neurones.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("Réseaux de neurones fini.")
            
    if config.support_vecteurs is not None:
        debut = time.time()
        p = SVC()
        if config.support_vecteurs.config_vide:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = str(round(score, 5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\SVC.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        else:
            g = GridSearchCV(p, param_grid={"C": config.support_vecteurs.regularisation})
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = str(round(g.best_score_,5)) + f" ({round((time.time() - debut) / 60, 4)} minutes)"
            with open(path + "\Rendu_final\Modele\SVC.pkl", "wb") as fichier_modele:
                pickle.dump(g, fichier_modele)
        print("Support vecteur fini.")
        
    return resultats


def visualisation_apprentissage(resultats_apprentissage: Dict[str, str]) -> None:
    """
    Fonction pour visualiser les résultats de l'apprentissage des modèles.
    
    Exemple :
>>> import os
>>> import pickle
>>> from Machine_learning import visualisation_apprentissage
>>> path = os.path.abspath(os.getcwd())
>>> with open(path + '\Resultats_apprentissage.pkl', 'rb') as fichier_modele:
        resultats_apprentissage = pickle.load(fichier_modele)
>>> visualisation_apprentissage(resultats_apprentissage)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Modele                 ┃ Score    ┃ Temps           ┃ Meilleur choix de Paramètres        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ DummyClassifier        │ 0.1065   │ 0.0029 minutes  │                                     │
│ SVC                    │ 0.16881  │ 0.8929 minutes  │ C=5.0                               │
│ MultinomialNB          │ 0.46912  │ 0.0111 minutes  │ alpha=0.001                         │
│ DecisionTreeClassifier │ 0.61608  │ 0.1452 minutes  │ max_depth=5, max_leaf_nodes=6       │
│ MLPClassifier          │ 0.67574  │ 9.2212 minutes  │ alpha=0.1, hidden_layer_sizes=[50,  │
│                        │          │                 │ 50], max_iter=500                   │
│ KNeighborsClassifier   │ 0.77263  │ 0.207 minutes   │ n_neighbors=1                       │
│ RandomForestClassifier │ 0.82427  │ 1.4676 minutes  │ max_depth=25, n_estimators=400      │
└────────────────────────┴──────────┴─────────────────┴─────────────────────────────────────┘
    """
    resume = Table()
    resume.add_column("Modele")
    resume.add_column("Score")
    resume.add_column("Temps")
    resume.add_column("Meilleur choix de Paramètres")
    for score, modele in sorted(
        [(score, modele) for modele, score in resultats_apprentissage.items()], 
        key=lambda x: x[0]
    ):
        score = score.replace(")", "")
        score_modele, temps = score.split("(")
        modele = str(modele).replace(")", "")
        nom_modele, parametres = modele.split("(")
        resume.add_row(
            str(nom_modele), 
            str(score_modele),
            str(temps),
            str(parametres)
        )
    print(resume)
    

def teste_precision_modele(
    data_dummy: pd.DataFrame, 
    modele: ModelRegressor
) -> Dict[ModelRegressor, float]:
    """
    Fonction permettant de voir quel modèle prédit le mieux sur les données test.
    
    Exemple :
>>> import os
>>> import pickle
>>> import pandas as pd
>>> from Machine_learning import visualisation_apprentissage, modifie_dataframe
>>> path = os.path.abspath(os.getcwd())

>>> data = pd.read_csv(path + "\Donnees\complete_database.csv", index_col=0)
>>> data_dummy = modifie_dataframe(data)
>>> with open(path + "\Modele\Foret_aleatoire.pkl", "rb") as fichier_modele:
...     foret_aleatoire = pickle.load(fichier_modele)

>>> teste_precision_modele(data_dummy, foret_aleatoire)
{RandomForestClassifier(max_depth=25, n_estimators=400): 0.8178913738019169}
    """
    _, X_te, _, y_te = renvoie_echantillons_test_train(data_dummy)
    prediction = dict()
    prediction_modele = modele.predict(X_te)
    valeur_prediction = accuracy_score(y_te, prediction_modele)
    prediction[modele.best_estimator_] = valeur_prediction
    return prediction


def matrice_confusion_donnees_test(
    data_dummy: pd.DataFrame, 
    modele: ModelRegressor
) -> confusion_matrix:
    """
    Fonction permettant de ressortir la matrice de confusion sur les données test.
    
    Exemple :
>>> import os
>>> import pickle
>>> import pandas as pd
>>> from Machine_learning import visualisation_apprentissage, modifie_dataframe
>>> path = os.path.abspath(os.getcwd())

>>> data = pd.read_csv(path + "\Donnees\complete_database.csv", index_col=0)
>>> data_dummy = modifie_dataframe(data)
>>> with open(path + "\Modele\Foret_aleatoire.pkl", "rb") as fichier_modele:
...     foret_aleatoire = pickle.load(fichier_modele)

>>> print(matrice_confusion_donnees_test(data_dummy, foret_aleatoire))
[[49  7  0  0  0  1  0  0  0  0]
 [ 5 40  3  0  0  2  0  0  0  0]
 [ 2 10 47  6  1  0  0  0  0  0]
 [ 1  0 12 44  8  0  0  0  0  0]
 [ 0  0  4  9 49  0  6  0  0  0]
 [ 2  1  2  0  0 60  0  0  0  2]
 [ 0  0  0  2  5  0 63  9  0  0]
 [ 0  0  0  1  1  0  4 44  0  0]
 [ 0  1  0  0  0  0  0  5 59  0]
 [ 0  0  0  0  0  0  0  0  2 57]]
    """
    _, X_te, _, y_te = renvoie_echantillons_test_train(data_dummy)
    prediction_modele = modele.predict(X_te)
    return confusion_matrix(y_te, prediction_modele)


if __name__ == "__main__":
    app()
