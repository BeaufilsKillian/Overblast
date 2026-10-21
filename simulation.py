"""
Simulation d'une itération (épisode) complète : les robots agissent tour à tour.
"""

import random

import numpy as np

from config import (
    COUT_ATTENTE,
    COUT_COLLISION,
    COUT_DROIT,
    COUT_REPASSAGE_RECENT,
    COUT_VIRAGE,
    DEPLACEMENTS,
    FREQUENCE_APPRENTISSAGE,
    INDICE_ATTENTE,
    MAX_ALLERS_RETOURS,
    NB_ROBOTS,
    RECOMPENSE_CHANGEMENT_PHASE,
    RECOMPENSE_COLLISION,
    RECOMPENSE_ELOIGNEMENT,
    RECOMPENSE_ETAPE,
    RECOMPENSE_MISSION_TERMINEE,
    RECOMPENSE_MOUVEMENT_INTERDIT,
    RECOMPENSE_RAPPROCHEMENT,
    RECOMPENSE_REPASSAGE,
    SEUIL_ECHEC_ENERGIE,
    TEMPS_MAX_EPISODE,
)
from entrepot import case_autorisee_pour_robot
from etat import construire_etat
from robot import Robot


def simuler_iteration(agent, grille, bases, missions_bloc):
    """Joue un épisode complet et retourne un dictionnaire de métriques.

    CORRECTION (bug réel) : l'original calculait ``positions_actuelles`` UNE
    seule fois en début de pas de temps, puis les robots bougeaient
    séquentiellement. Le 2e robot voyait donc des positions OBSOLÈTES (le 1er
    avait déjà bougé) alors que la détection de collision utilisait des
    positions fraîches. On recalcule maintenant l'occupation avant chaque
    décision pour que l'état reflète la réalité.
    """
    robots = [Robot(i, bases[i]) for i in range(NB_ROBOTS)]
    for robot, mission in zip(robots, missions_bloc):
        robot.definir_mission(mission)

    historique = []
    succes = False
    echec_energie = False

    for temps in range(TEMPS_MAX_EPISODE):
        historique.append([robot.position for robot in robots])

        ordre_robots = list(range(NB_ROBOTS))
        random.shuffle(ordre_robots)

        for idx in ordre_robots:
            robot = robots[idx]
            if robot.termine:
                continue

            # Occupation fraîche au moment de la décision (cf. correction).
            positions_actuelles = {
                r.position for r in robots if not r.termine
            }

            etat = construire_etat(robot, grille, positions_actuelles, bases)
            action = agent.choisir_action(etat)

            dx, dy = DEPLACEMENTS[action]
            nouvelle_position = (
                robot.position[0] + dx,
                robot.position[1] + dy,
            )

            recompense = RECOMPENSE_ETAPE
            collision = False
            mouvement_interdit = False

            # Obstacle ou base d'un autre robot.
            if not case_autorisee_pour_robot(grille, nouvelle_position, robot.id, bases):
                collision = True

            # Collision avec un autre robot (positions à jour).
            autres_positions = {
                r.position
                for r in robots
                if r.id != robot.id and not r.termine
            }
            if nouvelle_position in autres_positions:
                collision = True

            aller_retour = robot.est_aller_retour(nouvelle_position)

            if (
                aller_retour
                and robot.allers_retours_consecutifs >= MAX_ALLERS_RETOURS
            ):
                mouvement_interdit = True

            dist_carte = robot.carte_distance_objectif()
            ancienne_distance = dist_carte[robot.position]

            if collision:
                robot.energie += COUT_COLLISION
                robot.collision += 1
                recompense += RECOMPENSE_COLLISION
                nouvelle_position = robot.position

            elif mouvement_interdit:
                robot.energie += COUT_REPASSAGE_RECENT
                robot.allers_retours_bloques += 1
                recompense += RECOMPENSE_MOUVEMENT_INTERDIT
                nouvelle_position = robot.position

            else:
                if nouvelle_position in robot.positions_recentes:
                    robot.energie += COUT_REPASSAGE_RECENT
                    robot.repassages_recents += 1
                    recompense += RECOMPENSE_REPASSAGE

                nouvelle_direction = (dx, dy)

                if action == INDICE_ATTENTE:
                    # CORRECTION : coût d'attente nommé (l'original codait 1 en dur).
                    cout = COUT_ATTENTE
                    robot.allers_retours_consecutifs = 0
                else:
                    cout = COUT_DROIT

                    if (
                        robot.direction_precedente is not None
                        and nouvelle_direction != robot.direction_precedente
                    ):
                        cout += COUT_VIRAGE

                    if aller_retour:
                        robot.allers_retours_consecutifs += 1
                    else:
                        robot.allers_retours_consecutifs = 0

                robot.energie += cout
                robot.position = nouvelle_position
                robot.direction_precedente = nouvelle_direction
                robot.chemin.append(robot.position)
                robot.positions_recentes.append(robot.position)

                nouvelle_distance = dist_carte[robot.position]

                if nouvelle_distance < ancienne_distance:
                    recompense += RECOMPENSE_RAPPROCHEMENT
                else:
                    recompense += RECOMPENSE_ELOIGNEMENT

            ancienne_phase = robot.phase
            robot.verifier_phase()

            if robot.phase != ancienne_phase:
                recompense += RECOMPENSE_CHANGEMENT_PHASE

            # Bonus seulement si le robot est vraiment revenu sur SA base.
            if robot.termine and robot.position == robot.base:
                recompense += RECOMPENSE_MISSION_TERMINEE

            positions_suivantes = {
                r.position for r in robots if not r.termine
            }
            etat_suivant = construire_etat(
                robot, grille, positions_suivantes, bases
            )

            agent.memoire.ajouter((
                etat,
                action,
                recompense,
                etat_suivant,
                1.0 if robot.termine else 0.0,
            ))

        if temps % FREQUENCE_APPRENTISSAGE == 0:
            agent.apprendre()

        energie_moyenne_temp = np.mean([robot.energie for robot in robots])

        if energie_moyenne_temp > SEUIL_ECHEC_ENERGIE:
            echec_energie = True
            succes = False
            break

        # Réussite seulement si tous les robots sont sur leurs propres bases.
        if all(robot.termine and robot.position == robot.base for robot in robots):
            succes = True
            historique.append([robot.position for robot in robots])
            break

    energie_moyenne = np.mean([robot.energie for robot in robots])
    collisions_totales = sum(r.collision for r in robots)
    repassages_totaux = sum(r.repassages_recents for r in robots)
    allers_retours_bloques = sum(r.allers_retours_bloques for r in robots)

    if energie_moyenne > SEUIL_ECHEC_ENERGIE:
        succes = False
        echec_energie = True

    return {
        "succes": succes,
        "echec_energie": echec_energie,
        "historique": historique,
        "robots": robots,
        "energie_moyenne": energie_moyenne,
        "collisions": collisions_totales,
        "repassages": repassages_totaux,
        "allers_retours_bloques": allers_retours_bloques,
        "temps": len(historique),
    }
