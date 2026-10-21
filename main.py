"""
Point d'entrée : boucle d'apprentissage DQN et bilan final.

Lancement : ``python main.py``
"""

from affichage import animer_2d, tracer_energie
from config import initialiser_graines
from dqn import AgentDQN
from entrepot import creer_entrepot
from missions import generer_missions_bloc
from simulation import simuler_iteration


NOMBRE_ITERATIONS_DEFAUT = 10000
FREQUENCE_ANIMATION_DEFAUT = 2500


def _saisir_entier(message, valeur_defaut):
    """Lit un entier au clavier avec une valeur par défaut (entrée vide).

    CORRECTION : les deux ``input`` du code original dupliquaient la même
    logique de valeur par défaut ; on la factorise dans une fonction.
    """
    saisie = input(message).strip()
    if saisie == "":
        return valeur_defaut
    return int(saisie)


def main():
    initialiser_graines()

    grille, bases = creer_entrepot()
    agent = AgentDQN()

    nb_iterations = _saisir_entier(
        f"Nombre total d'itérations (défaut {NOMBRE_ITERATIONS_DEFAUT}) : ",
        NOMBRE_ITERATIONS_DEFAUT,
    )
    frequence_animation = _saisir_entier(
        f"Animation toutes les N itérations (défaut {FREQUENCE_ANIMATION_DEFAUT}) : ",
        FREQUENCE_ANIMATION_DEFAUT,
    )

    energies = []
    meilleur_resultat = None
    meilleure_energie = float("inf")
    premiere_reussite = None

    missions_bloc = generer_missions_bloc(grille, bases)

    for iteration in range(1, nb_iterations + 1):
        if (
            frequence_animation > 0
            and iteration > 1
            and (iteration - 1) % frequence_animation == 0
        ):
            missions_bloc = generer_missions_bloc(grille, bases)
            print("\n--- Nouveaux objectifs générés ---\n")

        resultat = simuler_iteration(agent, grille, bases, missions_bloc)
        energies.append(resultat["energie_moyenne"])

        if iteration % 50 == 0 or iteration == 1:
            print(
                f"Itération {iteration:05d}/{nb_iterations} | "
                f"Succès={resultat['succes']} | "
                f"Échec énergie={resultat['echec_energie']} | "
                f"Temps={resultat['temps']} | "
                f"Énergie={resultat['energie_moyenne']:.2f} | "
                f"Collisions={resultat['collisions']} | "
                f"Repassages={resultat['repassages']} | "
                f"AR bloqués={resultat['allers_retours_bloques']} | "
                f"Epsilon={agent.epsilon:.3f}"
            )

        if resultat["succes"] and premiere_reussite is None:
            premiere_reussite = resultat

        if (
            resultat["succes"]
            and resultat["energie_moyenne"] < meilleure_energie
        ):
            meilleure_energie = resultat["energie_moyenne"]
            meilleur_resultat = resultat

        if frequence_animation > 0 and iteration % frequence_animation == 0:
            animer_2d(
                grille,
                bases,
                resultat,
                titre=f"Animation itération {iteration}",
            )

    print("\n=== BILAN FINAL ===")

    if premiere_reussite is not None:
        print(f"Première réussite : {premiere_reussite['energie_moyenne']:.2f}")
        animer_2d(grille, bases, premiere_reussite, "Première réussite")
    else:
        print("Aucune réussite trouvée.")

    if meilleur_resultat is not None:
        print(f"Meilleur résultat : {meilleur_resultat['energie_moyenne']:.2f}")
        animer_2d(grille, bases, meilleur_resultat, "Meilleur résultat")
    else:
        print("Aucun meilleur résultat.")

    tracer_energie(energies)


if __name__ == "__main__":
    main()
