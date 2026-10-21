"""Benchmark : temps de génération des missions, avant / après optimisation.

Lancement : ``PYTHONPATH= ./.venv/Scripts/python.exe benchmark.py``
"""

import time
from collections import deque

import matplotlib

matplotlib.use("Agg")

import numpy as np

import chemins
import config
import entrepot
import missions


def bfs_python_pur(grille, source, robot_id, bases):
    """BFS en Python pur (équivalent de l'ancienne version), pour référence."""
    from entrepot import case_autorisee_pour_robot

    hauteur, largeur = grille.shape
    distances = np.full((hauteur, largeur), np.inf)
    file = deque([source])
    distances[source] = 0

    while file:
        pos = file.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            voisin = (pos[0] + dx, pos[1] + dy)
            if (
                case_autorisee_pour_robot(grille, voisin, robot_id, bases)
                and distances[voisin] == np.inf
            ):
                distances[voisin] = distances[pos] + 1
                file.append(voisin)

    return distances


if __name__ == "__main__":
    config.initialiser_graines()
    grille, bases = entrepot.creer_entrepot()

    # --- Nouvelle version (matrice précalculée + BFS vectorisé) ---------
    t0 = time.perf_counter()
    missions_bloc = missions.generer_missions_bloc(grille, bases)
    t_nouveau = time.perf_counter() - t0
    print(f"Nouvelle version (matrice précalculée) : {t_nouveau:6.2f} s")

    # --- Référence : coût d'un BFS Python pur (ancienne version) --------
    libres = entrepot.positions_libres(grille)
    t0 = time.perf_counter()
    for k in range(50):
        bfs_python_pur(grille, libres[k], 0, bases)
    cout_un_bfs = (time.perf_counter() - t0) / 50

    nb_bfs_ancien = config.NB_ROBOTS * config.NB_TIRAGES_MISSION
    t_ancien_estime = cout_un_bfs * nb_bfs_ancien

    print(f"BFS Python pur : {cout_un_bfs * 1000:.2f} ms/BFS")
    print(f"Ancienne version estimée ({nb_bfs_ancien} BFS) : {t_ancien_estime:6.2f} s")
    print(f"\nGain : {t_ancien_estime / t_nouveau:.0f}x plus rapide")

    # --- Correctitude : recalcul de chaque distance minimale (BFS pur) ---
    for i, m in enumerate(missions_bloc):
        bp = chemins.distance_point_a_point(grille, m.base, m.pickup, i, bases)
        pd = chemins.distance_point_a_point(grille, m.pickup, m.delivery, i, bases)
        bd = chemins.distance_point_a_point(grille, m.base, m.delivery, i, bases)
        assert int(bp + pd + bd) == m.distance_minimale, (
            i, int(bp + pd + bd), m.distance_minimale
        )
    print("Correctitude : distances minimales vérifiées (BFS pur) ✓")
