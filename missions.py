"""
Génération des missions : chaque robot reçoit un couple pickup -> delivery.
"""

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from chemins import bfs_distances
from config import DISTANCE_MIN_TOTALE_MISSION, NB_ROBOTS, NB_TIRAGES_MISSION
from entrepot import positions_libres


@dataclass
class Mission:
    """Une mission : aller chercher ``pickup`` puis livrer à ``delivery``.

    CORRECTION : l'original représentait une mission par un dictionnaire. Une
    dataclass apporte le typage, un accès par attribut (``m.pickup``) qui ne
    lève pas de KeyError en cas de faute de frappe (contrairement à
    ``m["pickup"]``) et l'auto-complétion dans l'IDE.
    """
    pickup: tuple
    delivery: tuple
    base: tuple
    dist_pickup: np.ndarray
    dist_delivery: np.ndarray
    dist_base: np.ndarray
    distance_minimale: int


def _matrice_distances(grille, libres, coords, robot_id, bases):
    """Matrice des distances BFS entre toutes les cases libres, pour un robot.

    OPTIMISATION : l'ancienne version lançait un BFS complet par tirage
    (jusqu'à ``NB_TIRAGES_MISSION`` BFS par robot, soit des dizaines de
    milliers). On précalcule ici UNE SEULE FOIS les distances entre toutes les
    cases libres (BFS vectorisé depuis chaque case), puis la génération se
    réduit à de simples lectures de tableau.
    """
    n = len(libres)
    matrice = np.full((n, n), np.inf)

    for i, source in enumerate(libres):
        distances = bfs_distances(grille, source, robot_id, bases)
        # Rassemblement vectorisé : distances vers toutes les cases libres.
        matrice[i, :] = distances[coords[:, 0], coords[:, 1]]

    return matrice


def generer_missions_bloc(grille, bases):
    """Génère une mission assez longue pour chaque robot.

    Le trajet total est ``base -> pickup -> delivery -> base`` ; seules les
    missions dont la distance totale atteint ``DISTANCE_MIN_TOTALE_MISSION``
    sont retenues. Parmi elles, on tire au hasard parmi les plus courtes.
    """
    libres = positions_libres(grille)
    coords = np.array(libres)
    index = {cellule: i for i, cellule in enumerate(libres)}
    # CORRECTION : ``pickup in bases`` / ``delivery in bases`` remplacés par un
    # test sur un set (O(1) au lieu de O(n)).
    bases_set = set(bases)

    missions = []

    for robot_id in range(NB_ROBOTS):
        base = bases[robot_id]
        dist_base = bfs_distances(grille, base, robot_id, bases)
        matrice = _matrice_distances(grille, libres, coords, robot_id, bases)

        candidats_par_distance = defaultdict(list)

        for _ in range(NB_TIRAGES_MISSION):
            pickup = random.choice(libres)
            delivery = random.choice(libres)

            if pickup == delivery:
                continue
            if pickup in bases_set or delivery in bases_set:
                continue

            dist_base_pickup = dist_base[pickup]
            dist_base_delivery = dist_base[delivery]
            if dist_base_pickup == np.inf or dist_base_delivery == np.inf:
                continue

            # OPTIMISATION : lecture directe dans la matrice précalculée
            # (plus aucun BFS dans cette boucle).
            dist_pickup_delivery = matrice[index[pickup], index[delivery]]
            if dist_pickup_delivery == np.inf:
                continue

            distance_totale = int(
                dist_base_pickup + dist_pickup_delivery + dist_base_delivery
            )

            if distance_totale >= DISTANCE_MIN_TOTALE_MISSION:
                candidats_par_distance[distance_totale].append((pickup, delivery))

        distances_valides = list(candidats_par_distance)
        if not distances_valides:
            raise RuntimeError(
                f"Impossible de générer une mission assez longue "
                f"pour le robot {robot_id}."
            )

        distance_choisie = min(distances_valides)
        pickup, delivery = random.choice(candidats_par_distance[distance_choisie])

        mission = Mission(
            pickup=pickup,
            delivery=delivery,
            base=base,
            dist_pickup=bfs_distances(grille, pickup, robot_id, bases),
            dist_delivery=bfs_distances(grille, delivery, robot_id, bases),
            # OPTIMISATION : l'original recalculait ``dist_base`` (un BFS
            # complet) alors qu'il était déjà calculé au-dessus ; on le réutilise.
            dist_base=dist_base,
            distance_minimale=distance_choisie,
        )
        missions.append(mission)

    print("Nouvelles missions générées :")
    for i, m in enumerate(missions):
        print(
            f"Robot {i} | base={m.base} | pickup={m.pickup} | "
            f"delivery={m.delivery} | distance min = {m.distance_minimale}"
        )

    return missions
