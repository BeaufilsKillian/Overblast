"""
Construction de l'état (vecteur d'observation) fourni au DQN.
"""

import numpy as np

from config import DEPLACEMENTS, HAUTEUR, LARGEUR, TAILLE_ETAT
from entrepot import case_autorisee_pour_robot
from robot import Phase


def meilleure_direction_locale(robot, grille, bases):
    """Direction (dx, dy) du voisin autorisé qui minimise la distance à l'objectif.

    Retourne un couple d'entiers dans {-1, 0, 1}.
    """
    dist = robot.carte_distance_objectif()
    meilleure_case = robot.position
    meilleure_distance = dist[robot.position]

    for dx, dy in DEPLACEMENTS[:4]:
        voisin = (robot.position[0] + dx, robot.position[1] + dy)

        if case_autorisee_pour_robot(grille, voisin, robot.id, bases):
            if dist[voisin] < meilleure_distance:
                meilleure_distance = dist[voisin]
                meilleure_case = voisin

    # CORRECTION : ``np.sign`` renvoie des flottants numpy (y compris -0.0) ;
    # on convertit en entiers pour un état de type stable et propre.
    return (
        int(np.sign(meilleure_case[0] - robot.position[0])),
        int(np.sign(meilleure_case[1] - robot.position[1])),
    )


def construire_etat(robot, grille, positions_occupees, bases):
    """Construit le vecteur d'état de taille TAILLE_ETAT pour ``robot``.

    Composition de l'état (13 valeurs) :
      0-1 : position du robot (normalisée)
      2-3 : position de l'objectif (normalisée)
      4-5 : meilleure direction locale vers l'objectif
      6-7 : one-hot de la phase (PICKUP, DELIVERY)
      8-11 : occupation des 4 cases voisines (obstacle/robot ou libre)
      12  : risque de repassage sur une case récente
    """
    objectif = robot.objectif()
    dx_obj, dy_obj = meilleure_direction_locale(robot, grille, bases)

    # Occupation des 4 cases voisines (haut, bas, gauche, droite).
    cases_autour = []
    for dx, dy in DEPLACEMENTS[:4]:
        pos = (robot.position[0] + dx, robot.position[1] + dy)

        if not case_autorisee_pour_robot(grille, pos, robot.id, bases):
            cases_autour.append(1.0)
        elif pos in positions_occupees:
            cases_autour.append(1.0)
        else:
            cases_autour.append(0.0)

    # Risque de repassage : un voisin déjà visité récemment est pénalisé.
    risque_repassage = 0.0
    for dx, dy in DEPLACEMENTS[:4]:
        pos_test = (robot.position[0] + dx, robot.position[1] + dy)
        if pos_test in robot.positions_recentes:
            risque_repassage = 1.0
            break

    etat = np.array([
        robot.position[0] / HAUTEUR,
        robot.position[1] / LARGEUR,
        objectif[0] / HAUTEUR,
        objectif[1] / LARGEUR,
        dx_obj,
        dy_obj,
        1.0 if robot.phase == Phase.PICKUP else 0.0,
        1.0 if robot.phase == Phase.DELIVERY else 0.0,
        cases_autour[0],
        cases_autour[1],
        cases_autour[2],
        cases_autour[3],
        risque_repassage,
    ], dtype=np.float32)

    # CORRECTION : vérification de cohérence de la taille de l'état (au lieu de
    # faire confiance à un "13" implicite) — active uniquement si la constante
    # est désynchronisée de la construction.
    assert len(etat) == TAILLE_ETAT, (
        f"État de taille {len(etat)} au lieu de {TAILLE_ETAT}."
    )

    return etat
