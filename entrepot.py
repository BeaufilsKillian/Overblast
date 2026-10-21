"""
Génération de l'entrepôt (grille) et tests d'occupation des cases.

Valeurs de la grille :
  0 = case libre, 1 = étagère / mur, 2 = base d'un robot.
"""

import numpy as np

from config import HAUTEUR, LARGEUR, NB_ROBOTS


def creer_entrepot():
    """Construit la grille de l'entrepôt et place les bases.

    Retourne ``(grille, bases)``.
    """
    grille = np.zeros((HAUTEUR, LARGEUR), dtype=int)

    # Murs extérieurs
    grille[0, :] = 1
    grille[HAUTEUR - 1, :] = 1
    grille[:, 0] = 1
    grille[:, LARGEUR - 1] = 1

    largeur_allee = 3
    largeur_bloc = 3

    allee_h1 = HAUTEUR // 3
    allee_h2 = 2 * HAUTEUR // 3

    # Positions de départ des blocs d'étagères le long de l'axe horizontal.
    x = 1 + largeur_allee
    blocs_positions = []
    while x + largeur_bloc < LARGEUR - 1:
        blocs_positions.append(x)
        x += largeur_bloc + largeur_allee

    # CORRECTION : ``i in [liste]`` était un test d'appartenance O(n) ; on
    # utilise un ensemble (set) pour un test O(1), plus lisible et plus rapide.
    lignes_allees = {
        allee_h1 - 1, allee_h1, allee_h1 + 1,
        allee_h2 - 1, allee_h2, allee_h2 + 1,
    }

    for bloc_x in blocs_positions:
        for i in range(1, HAUTEUR - 1):
            if i in lignes_allees:
                continue
            for j in range(bloc_x, min(bloc_x + largeur_bloc, LARGEUR - 1)):
                grille[i, j] = 1

    # Zone carrée libre pour les bases
    for i in range(HAUTEUR - 6, HAUTEUR - 1):
        for j in range(1, 9):
            grille[i, j] = 0

    # Bases espacées (une par robot : la taille doit correspondre à NB_ROBOTS).
    bases = [
        (HAUTEUR - 3, 2),
        (HAUTEUR - 3, 4),
        (HAUTEUR - 3, 6),
    ]

    # CORRECTION : le nombre de bases codées en dur doit rester cohérent avec
    # NB_ROBOTS ; on le vérifie explicitement au lieu de le supposer.
    assert len(bases) == NB_ROBOTS, (
        f"{len(bases)} bases définies pour NB_ROBOTS = {NB_ROBOTS}."
    )

    for base in bases:
        grille[base] = 2

    taux = 100 * np.sum(grille == 1) / (HAUTEUR * LARGEUR)
    print(f"Carte générée : {taux:.1f}% d'étagères")

    return grille, bases


def case_libre(grille, pos):
    """Vrai si la case est dans la grille et n'est pas une étagère/mur.

    CORRECTION : le test de bornes et l'accès ``grille[x, y]`` étaient combinés
    sur une seule ligne ; on les sépare pour éviter tout risque d'IndexError et
    rendre l'intention explicite.
    """
    x, y = pos
    if not (0 <= x < HAUTEUR and 0 <= y < LARGEUR):
        return False
    return grille[x, y] != 1


def case_autorisee_pour_robot(grille, pos, robot_id, bases):
    """Vrai si la case est libre ET ne contient pas la base d'un autre robot."""
    if not case_libre(grille, pos):
        return False

    for autre_id, base in enumerate(bases):
        if autre_id != robot_id and pos == base:
            return False

    return True


def positions_libres(grille):
    """Retourne toutes les cases non-étagères (cases libres + bases)."""
    return [
        (i, j)
        for i in range(HAUTEUR)
        for j in range(LARGEUR)
        if grille[i, j] != 1
    ]
