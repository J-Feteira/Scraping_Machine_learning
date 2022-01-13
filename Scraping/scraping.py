#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Classes pour scrapper des données sur backmarket avec sa fonction permettant 

Fonction :
- main

Classes :
- Session
- Description
"""

from selenium import webdriver
from bs4 import BeautifulSoup as BS
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from requests import get
from time import sleep
from copy import deepcopy
from serde.json import to_json
import json
from typing import List
import os


def main(nom_fichier: str, changer_page: bool=True) -> str:
    """
    Fonction pour scrapper les annonces de smartphones sur Back Market.
    
    Exemple :
>>> from Scrapping import main
>>> main(nom_fichier='data.json')
    """
    lien = "https://www.backmarket.fr/"
    while True:
        nav = Session(lien)
        nav.navigateur.maximize_window()
        nav.accepter_cookies()
        if lien == "https://www.backmarket.fr/":
            nav.chercher_element("smartphone")
        for _ in range(0,5):
            sleep(10)
            nav.clique_annonce(nom_fichier)
            try:
                sleep(5)
                if changer_page:
                    nav.change_page()
                else:
                    nav.navigateur.quit()
                    break
            except NoSuchElementException:
                nav.navigateur.quit()
                break
        try:
            lien = nav.navigateur.current_url
            sleep(5)
            nav.navigateur.quit()
            sleep(5)
        except:
            break
    return "Scrapping terminé."
    

class Session:
    """Classe permettant de parcourir le site Backmarket pour récupérer les annonces de smartphones."""
    
    def __init__(self, lien: str):
        self.navigateur = webdriver.Chrome()
        self.navigateur.get(lien)
        sleep(6)
    
    
    def accepter_cookies(self) -> None:
        """Accepte les cookies."""
        bouton_cookie = self.navigateur.find_element(
            By.XPATH, value="//button[@data-qa='accept-cta']"
        )
        bouton_cookie.click()
        del bouton_cookie
    
    
    def chercher_element(self, mot_a_chercher: str) -> None:
        """Fonction pour chercher un mot spécifique."""
        recherche = self.navigateur.find_element(
            By.XPATH, value="//input[@data-qa='search-bar-input']"
        )
        recherche.send_keys(mot_a_chercher)
        sleep(5)
        recherche.send_keys(Keys.ENTER)
        del recherche
    
    
    def change_page(self) -> None:
        """Fonction pour changer de page """
        changer_page = self.navigateur.find_element(
            By.XPATH, value="//button[@data-test='pagination-next']"
        )
        sleep(4)
        changer_page.click()
        del changer_page


    def clique_annonce(self, nom_fichier: str) -> None:
        """Fonction pour cliquer sur l'annonce."""
        sleep(5)
        annonces_page = self.navigateur.find_elements(
            By.XPATH, value="//a[@data-qa='product-thumb']"
        )
        sleep(5)
        for numero_annonce in range(0, len(annonces_page)):
            sleep(5)
            annonces = self.navigateur.find_elements(
                By.XPATH, value="//a[@data-qa='product-thumb']"
            )
            sleep(6)
            annonces[numero_annonce].click()
            sleep(8)
            try:
                tableau_description = self._recupere_description()
                liste_caracteristiques = self._nettoie_description(tableau_description)
                del tableau_description
                liste_etats_prix = self._recupere_prix_etat()
                for etat_prix in liste_etats_prix:
                    etat, prix = etat_prix.split(" -> ")
                    liste_caracteristiques_2 = deepcopy(liste_caracteristiques)
                    liste_caracteristiques_2.append(etat)
                    liste_caracteristiques_2.append(prix)
                    liste_caracteristiques_2.append(f'Lien ==> {self.navigateur.current_url}')
                    path = os.path.abspath(os.getcwd())
                    path_repertoire = os.path.join(path, "Donnees")
                    if not os.path.exists(path_repertoire):
                        os.mkdir(path_repertoire)
                    with open(path_repertoire + "/" + nom_fichier, "a", encoding='utf8') as fichier:
                        fichier.write(Description(liste_caracteristiques_2).to_json() + ",")
                del annonces, liste_caracteristiques, liste_etats_prix, liste_caracteristiques_2
            except IndexError:
                print(f"Erreur pour l'annonce numéro {numero_annonce}")
            sleep(5)
            self.navigateur.back()
        del annonces_page


    def _recupere_description(self): #-> bs4.element.Tag:
        """Fonction récupérant la description pour un smartphone."""
        soupe = BS(self.navigateur.page_source, "html.parser")
        recherche_tableau = soupe.find_all(name="ul", attrs={"class" : "list-none"})
        tableau_description = recherche_tableau[5]
        del soupe, recherche_tableau
        return tableau_description
    
    
    def _recupere_prix_etat(self) -> List[str]:
        """Fonction qui récupère les prix et les états d'un smartphone."""
        soupe = BS(self.navigateur.page_source, "html.parser")
        etats = soupe.find_all(
            "p", 
            attrs={"class" : "break-words font-body text-2 leading-2 font-light"}
        )
        etats_et_prix = [etat.parent for etat in etats]
        resultat = list()
        for ligne in etats_et_prix:
            etat, prix = ligne.find_all("p")
            resultat.append(
                f'État : {etat.get_text().strip()} -> Prix : {prix.get_text().strip()}'
            )
        del soupe, etats, etats_et_prix
        return resultat
    
    
    @staticmethod
    def _nettoie_description(tableau_description) -> List[str]:
        """Fonction nettoyant le tableau de description."""
        liste_description = tableau_description.get_text().split('\n')
        liste_caracteristiques = [
            (''.join(caracteristique)).replace("  ", "")
            for caracteristique in zip(liste_description[0::2], liste_description[1::2])
        ]
        del liste_description
        return liste_caracteristiques
    

    
class Description:
    """
    Classe permettant de récupérer les informations principales sur les caracéristiques du smartphone.
    """

    def __init__(self, description):
        description_copie = deepcopy(description)
        self.set_etat(description_copie)
        self.set_marque(description_copie)
        self.set_prix(description_copie)
        self.set_serie(description_copie)
        self.set_couleur(description_copie)
        self.set_taille_ecran(description_copie)
        self.set_capacite_stockage(description_copie)
        self.set_modele(description_copie)
        self.set_megapixel(description_copie)
        self.set_systeme_exploitation(description_copie)
        self.set_resolution_ecran(description_copie)
        self.set_reseau(description_copie)
        self.set_date_de_sortie(description_copie)
        self.set_memoire(description_copie)
        self.set_vitesse_processeur(description_copie)
        self.set_nombre_coeur(description_copie)
        self.set_connecteur(description_copie)
        self.set_double_sim(description_copie)
        self.set_port_carte_SD(description_copie)
        self.set_pliable(description_copie)
        self.set_reseau_5G(description_copie)
        self.set_appareil_photo(description_copie)
        self.set_poids(description_copie)
        self.set_hauteur(description_copie)
        self.set_largeur(description_copie)
        self.set_profondeur(description_copie)
        self.set_lien(description_copie)


    def __str__(self):
        return f"""
etat                 : {self.etat}
marque               : {self.marque}
prix                 : {self.prix}
serie                : {self.serie}
couleur              : {self.couleur}
taille_ecran         : {self.taille_ecran}
capacite_stockage    : {self.capacite_stockage}
modele               : {self.modele}
megapixel            : {self.megapixel}
systeme_exploitation : {self.systeme_exploitation}
resolution_ecran     : {self.resolution_ecran}
reseau               : {self.reseau}
date_de_sortie       : {self.date_de_sortie}
memoire              : {self.memoire}
vitesse_processeur   : {self.vitesse_processeur}
nombre_coeur         : {self.nombre_coeur}
connecteur           : {self.connecteur}
double_sim           : {self.double_sim}
port_carte_SD        : {self.port_carte_SD}
pliable              : {self.pliable}
reseau_5G            : {self.reseau_5G}
appareil_photo       : {self.appareil_photo}
poids                : {self.poids}
hauteur              : {self.hauteur}
largeur              : {self.largeur}
profondeur           : {self.profondeur}
lien                 : {self.lien}
"""
    
    
    def set_etat(self, liste_annonce: List[str]) -> None:
        """Récupère état du smartphone."""
        for element in liste_annonce:
            if 'état' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.etat = information
                break
        else: self.etat = ''
    
    
    def set_marque(self, liste_annonce: List[str]) -> None:
        """Récupère marque du smartphone."""
        for element in liste_annonce:
            if 'marque :' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.marque = information
                break
        else: self.marque = ''
    
    
    def set_prix(self, liste_annonce: List[str]) -> None:
        """Récupère prix du smartphone."""
        for element in liste_annonce:
            if 'prix' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.prix = information
                break
        else: self.prix = ''
    
    
    def set_serie(self, liste_annonce: List[str]) -> None:
        """Récupère la série du smartphone."""
        for element in liste_annonce:
            if 'série' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.serie = information
                break
        else: self.serie = ''
    
    
    def set_couleur(self, liste_annonce: List[str]) -> None:
        """Récupère la couleur du smartphone."""
        for element in liste_annonce:
            if 'couleur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.couleur = information
                break
        else: self.couleur = ''
    
    
    def set_taille_ecran(self, liste_annonce: List[str]) -> None:
        """Récupère la taille de l'écran du smartphone."""
        for element in liste_annonce:
            if 'taille écran' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.taille_ecran = information
                break
        else: self.taille_ecran = ''
    
    
    def set_capacite_stockage(self, liste_annonce: List[str]) -> None:
        """Récupère la capacité de stockage du smartphone."""
        for element in liste_annonce:
            if 'stockage' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.capacite_stockage = information
                break
        else: self.capacite_stockage = ''
    
    
    def set_modele(self, liste_annonce: List[str]) -> None:
        """Récupère le modèle du smartphone."""
        for element in liste_annonce:
            if 'modèle' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.modele = information
                break
        else: self.modele = ''
    
    
    def set_megapixel(self, liste_annonce: List[str]) -> None:
        """Récupère les mégapixels du smartphone."""
        for element in liste_annonce:
            if 'megapixel' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.megapixel = information
                break
        else: self.megapixel = ''
    
    
    def set_systeme_exploitation(self, liste_annonce: List[str]) -> None:
        """Récupère le système d'exploitation du smartphone."""
        for element in liste_annonce:
            if 'système' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.systeme_exploitation = information
                break
        else: self.systeme_exploitation = ''
    
    
    def set_resolution_ecran(self, liste_annonce: List[str]) -> None:
        """Récupère la résolution d'écran du smartphone."""
        for element in liste_annonce:
            if 'résolution' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.resolution_ecran = information
                break
        else: self.resolution_ecran = ''
    
    
    def set_reseau(self, liste_annonce: List[str]) -> None:
        """Récupère le réseau du smartphone."""
        for element in liste_annonce:
            if 'réseau' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.reseau = information
                break
        else: self.reseau = ''
    
    
    def set_date_de_sortie(self, liste_annonce: List[str]) -> None:
        """Récupère la date de sortie du smartphone."""
        for element in liste_annonce:
            if ('date' or 'année') in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.date_de_sortie = information
                break
        else: self.date_de_sortie = ''
    
    
    def set_memoire(self, liste_annonce: List[str]) -> None:
        """Récupère la mémoire du smartphone."""
        for element in liste_annonce:
            if 'mémoire' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.memoire = information
                break
        else: self.memoire = ''
    
    
    def set_vitesse_processeur(self, liste_annonce: List[str]) -> None:
        """Récupère la vitesse du processeur du smartphone."""
        for element in liste_annonce:
            if 'vitesse du processeur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.vitesse_processeur = information
                break
        else: self.vitesse_processeur = ''
    
    
    def set_nombre_coeur(self, liste_annonce: List[str]) -> None:
        """Récupère le nombre de coeurs du smartphone."""
        for element in liste_annonce:
            if 'coeur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.nombre_coeur = information
                break
        else: self.nombre_coeur = ''
    
    
    def set_connecteur(self, liste_annonce: List[str]) -> None:
        """Récupère le connecteur du smartphone."""
        for element in liste_annonce:
            if 'connecteur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.connecteur = information
                break
        else: self.connecteur = ''
    
    
    def set_double_sim(self, liste_annonce: List[str]) -> None:
        """Récupère la valeur donnant si le smartphone peut contenir 2 cartes sim."""
        for element in liste_annonce:
            if 'double sim' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.double_sim = information
                break
        else: self.double_sim = ''
    
    
    def set_port_carte_SD(self, liste_annonce: List[str]) -> None:
        """Récupère la valeur donnant si le smartphone peut contenir une carte SD."""
        for element in liste_annonce:
            if 'carte sd' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.port_carte_SD = information
                break
        else: self.port_carte_SD = ''
    
    
    def set_pliable(self, liste_annonce: List[str]) -> None:
        """Récupère la valeur permettant de savoir si le smartphone est pliable."""
        for element in liste_annonce:
            if 'pliable' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.pliable = information
                break
        else: self.pliable = ''
    
    
    def set_reseau_5G(self, liste_annonce: List[str]) -> None:
        """Récupère la compatibilité réseau 5G du smartphone."""
        for element in liste_annonce:
            if '5G' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.reseau_5G = information
                break
        else: self.reseau_5G = ''
    
    
    def set_appareil_photo(self, liste_annonce: List[str]) -> None:
        """Récupère le type d'appareil photo du smartphone."""
        for element in liste_annonce:
            if 'appareil photo' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.appareil_photo = information
                break
        else: self.appareil_photo = ''
    
    
    def set_poids(self, liste_annonce: List[str]) -> None:
        """Récupère le poids du smartphone."""
        for element in liste_annonce:
            if 'poids' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.poids = information
                break
        else: self.poids = ''
    
    
    def set_hauteur(self, liste_annonce: List[str]) -> None:
        """Récupère la hauteur du smartphone."""
        for element in liste_annonce:
            if 'hauteur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.hauteur = information
                break
        else: self.hauteur = ''
    
    
    def set_largeur(self, liste_annonce: List[str]) -> None:
        """Récupère la largeur du smartphone."""
        for element in liste_annonce:
            if 'largeur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.largeur = information
                break
        else: self.largeur = ''
    
    
    def set_profondeur(self, liste_annonce: List[str]) -> None:
        """Récupère la profondeur du smartphone."""
        for element in liste_annonce:
            if 'profondeur' in element.lower():
                _, information = element.split(':')
                liste_annonce.remove(element)
                self.profondeur = information
                break
        else: self.profondeur = ''

    
    def set_lien(self, liste_annonce: List[str]) -> None:
        """Affecte le lien vers la page détaillée."""
        for element in liste_annonce:
            if 'lien' in element.lower():
                _, information = element.split('==>')
                liste_annonce.remove(element)
                self.lien = information
                break
        else: self.lien = ''

    
    def to_json(self) -> str:
        """Renvoit une chaine pour stocker le résultat en json."""
        return json.dumps(self.__dict__)
