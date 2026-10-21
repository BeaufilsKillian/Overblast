"""
Calcul des distances sur la grille par parcours en largeur (BFS).
"""

from collections import deque

import numpy as np

from config import DEPLACEMENTS, HAUTEUR, LARGEUR
from entrepot import case_autorisee_pour_robot, case_libre


def bfs_distances(grille, source, robot_id=None, bases=None):
    """Distances minimales depuis ``source`` vers toutes les cases atteignables.

    Si ``robot_id`` et ``bases`` sont fournis, le robot ne peut pas traverser
    les bases des autres robots. Sinon, seules les cases libres sont franchies.

    OPTIMISATION : BFS vectorisé en numpy. Le BFS Python pur traitait les cases
    une par une (avec un appel de fonction par voisin) et était ~10x plus lent.
    Ici, tout le « front » est propagé d'un coup via des opérations numpy.
    """
    hauteur, largeur = grille.shape
    passable = grille != 1

    # Les bases des autres robots sont marquées infranchissables en une passe
    # (l'ancienne version testait chaque voisin une à une).
    if bases is not None and robot_id is not None:
        for autre_id, base in enumerate(bases):
            if autre_id != robot_id:
                passable[base] = False

    distances = np.full((hauteur, largeur), np.inf)
    distances[source] = 0.0

    front = np.zeros((hauteur, largeur), dtype=bool)
    front[source] = True

    d = 0
    while front.any():
        d += 1
        voisins = np.zeros((hauteur, largeur), dtype=bool)
        voisins[:-1, :] |= front[1:, :]   # haut
        voisins[1:, :] |= front[:-1, :]   # bas
        voisins[:, :-1] |= front[:, 1:]   # gauche
        voisins[:, 1:] |= front[:, :-1]   # droite

        front = voisins & passable & (distances == np.inf)
        distances[front] = d

    return distances


def distance_bfs(grille, depart, arrivee, robot_id=None, bases=None):
    """Distance BFS entre deux cases (inf si aucun chemin).

    Calcule la carte COMPLÈTE des distances : gardée pour le débogage et les
    tests. Pour un simple point à point, préférer ``distance_point_a_point``.
    """
    distances = bfs_distances(grille, arrivee, robot_id, bases)
    return distances[depart]


def distance_point_a_point(grille, depart, arrivee, robot_id=None, bases=None):
    """Distance BFS entre deux cases, avec arrêt anticipé dès que ``depart``
    est atteint.

    OPTIMISATION : l'ancienne ``distance_bfs`` calculait la carte de distance
    COMPLÈTE (toute la grille) avant de lire une seule valeur. Ici le BFS part
    de ``arrivee`` et s'arrête dès que ``depart`` est dépilé — le BFS garantit
    que c'est bien la plus courte distance — sans explorer tout le reste.
    """
    if depart == arrivee:
        return 0.0

    distances = np.full((HAUTEUR, LARGEUR), np.inf)
    file = deque([arrivee])
    distances[arrivee] = 0

    while file:
        pos = file.popleft()

        if pos == depart:
            return distances[pos]

        for dx, dy in DEPLACEMENTS[:4]:
            voisin = (pos[0] + dx, pos[1] + dy)

            if bases is not None and robot_id is not None:
                autorise = case_autorisee_pour_robot(grille, voisin, robot_id, bases)
            else:
                autorise = case_libre(grille, voisin)

            if autorise and distances[voisin] == np.inf:
                distances[voisin] = distances[pos] + 1
                file.append(voisin)

    return np.inf
