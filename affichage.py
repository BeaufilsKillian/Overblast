"""
Affichage 2D de la simulation et courbe d'évolution de l'énergie.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from config import COULEURS_ROBOTS, PAUSE_ANIMATION, SEUIL_ECHEC_ENERGIE


def dessiner_2d(grille, bases, robots, historique, frame, titre, ax=None):
    """Dessine la grille et les robots à une frame donnée sur l'axes ``ax``.

    CORRECTION : la fonction utilisait la « state machine » pyplot (plt.imshow,
    plt.scatter, ...), ce qui l'empêchait de cohabiter avec d'autres éléments
    de la figure (comme un bouton). Elle cible désormais un axes explicite.
    """
    if ax is None:
        ax = plt.gca()

    couleurs = mcolors.ListedColormap(["white", "black", "lightgreen"])
    bornes = [-0.5, 0.5, 1.5, 2.5]
    norme = mcolors.BoundaryNorm(bornes, couleurs.N)

    ax.imshow(grille, cmap=couleurs, norm=norme)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titre)

    for robot_id, base in enumerate(bases):
        ax.scatter(
            base[1],
            base[0],
            marker="s",
            s=50,
            color=COULEURS_ROBOTS[robot_id],
            edgecolors="black",
            label=f"Base R{robot_id}",
        )

    positions = historique[frame]

    for robot_id, robot in enumerate(robots):
        chemin_robot = [historique[t][robot_id] for t in range(frame + 1)]

        if len(chemin_robot) > 1:
            xs = [p[1] for p in chemin_robot]
            ys = [p[0] for p in chemin_robot]
            ax.plot(
                xs,
                ys,
                color=COULEURS_ROBOTS[robot_id],
                linewidth=1.6,
                alpha=0.8,
            )

        pos = positions[robot_id]
        ax.scatter(
            pos[1],
            pos[0],
            marker="o",
            s=50,
            color=COULEURS_ROBOTS[robot_id],
            edgecolors="black",
            label=f"R{robot_id}" if frame == 0 else None,
        )

        ax.scatter(
            robot.pickup[1],
            robot.pickup[0],
            marker="x",
            s=70,
            color=COULEURS_ROBOTS[robot_id],
        )

        ax.scatter(
            robot.delivery[1],
            robot.delivery[0],
            marker="*",
            s=100,
            color=COULEURS_ROBOTS[robot_id],
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)


def animer_2d(grille, bases, resultat, titre):
    """Rejoue un épisode, avec un bouton « Recommencer » pour le relancer.

    AJOUT : bouton interactif qui relance l'animation depuis la première frame,
    sans relancer le programme. À la fin de l'épisode, cliquez sur
    « Recommencer » pour la rejouer, ou fermez la fenêtre pour poursuivre.
    """
    historique = resultat["historique"]
    robots = resultat["robots"]
    nb_frames = len(historique)

    fig = plt.figure(figsize=(13, 7))
    ax_plot = fig.add_axes([0.05, 0.10, 0.68, 0.86])
    ax_bouton = fig.add_axes([0.83, 0.04, 0.13, 0.06])
    bouton = Button(ax_bouton, "Recommencer")

    etat = {"recommencer": False}

    def on_click(_event):
        etat["recommencer"] = True

    bouton.on_clicked(on_click)

    plt.ion()
    plt.show()

    # Boucle : rejoue l'animation, puis attend un clic ou la fermeture.
    while plt.get_fignums():
        etat["recommencer"] = False

        for frame in range(nb_frames):
            ax_plot.clear()
            dessiner_2d(grille, bases, robots, historique, frame, titre, ax=ax_plot)
            plt.pause(PAUSE_ANIMATION)
            if etat["recommencer"]:
                break

        # Attend « Recommencer » ou la fermeture de la fenêtre.
        while not etat["recommencer"] and plt.get_fignums():
            plt.pause(0.05)

    plt.ioff()


def moyenne_mobile(valeurs, fenetre=100):
    """Moyenne mobile (filtre moyenneur) de ``valeurs``.

    CORRECTION : l'original retournait les valeurs brutes quand
    ``len(valeurs) < fenetre``, puis les traçait décalées ; ici la fenêtre est
    réduite automatiquement et le tableau vide est géré.
    """
    valeurs = np.asarray(valeurs, dtype=float)
    if valeurs.size == 0:
        return np.array([])

    if fenetre > valeurs.size:
        fenetre = valeurs.size

    noyau = np.ones(fenetre) / fenetre
    return np.convolve(valeurs, noyau, mode="valid")


def tracer_energie(energies):
    """Trace l'énergie moyenne par itération (brute + lissée)."""
    if len(energies) == 0:
        return

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, len(energies) + 1),
        energies,
        alpha=0.25,
        linewidth=1,
        label="Énergie brute",
    )

    fenetre = max(10, min(100, len(energies) // 20))
    energies_lissees = moyenne_mobile(energies, fenetre)

    if len(energies_lissees) > 0:
        # CORRECTION : la courbe lissée démarrait à ``fenetre`` (décalée vers la
        # droite) ; on centre chaque point au milieu de sa fenêtre.
        decalage = fenetre // 2
        x_lisse = range(1 + decalage, 1 + decalage + len(energies_lissees))
        plt.plot(
            x_lisse,
            energies_lissees,
            linewidth=2.5,
            label=f"Énergie lissée fenêtre={fenetre}",
        )

    plt.axhline(
        SEUIL_ECHEC_ENERGIE,
        linestyle="--",
        linewidth=1.5,
        label="Seuil échec énergie",
    )

    plt.title("Énergie moyenne")
    plt.xlabel("Itération")
    plt.ylabel("Énergie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
