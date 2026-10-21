"""
Classe Robot : état interne (position, énergie, phase), mission et déplacement.
"""

from collections import deque
from enum import Enum

from config import NB_POSITIONS_RECENTES


class Phase(Enum):
    """Étapes du parcours d'un robot.

    CORRECTION : les phases étaient des chaînes de caractères ("PICKUP",
    "DELIVERY"...) dispersées dans le code. Un enum évite les fautes de frappe
    et centralise les valeurs possibles.
    """
    PICKUP = "PICKUP"
    DELIVERY = "DELIVERY"
    BASE = "BASE"
    TERMINE = "TERMINE"


class Robot:
    def __init__(self, identifiant, base):
        self.id = identifiant
        self.base = base
        self.reset()

    def reset(self):
        self.position = self.base
        self.direction_precedente = None
        self.energie = 0.0
        self.collision = 0
        self.repassages_recents = 0
        self.allers_retours_bloques = 0
        self.allers_retours_consecutifs = 0
        self.termine = False

        self.phase = Phase.PICKUP
        self.pickup = None
        self.delivery = None
        self.mission = None

        self.chemin = [self.position]
        self.positions_recentes = deque(maxlen=NB_POSITIONS_RECENTES)
        self.positions_recentes.append(self.position)

    def definir_mission(self, mission):
        self.mission = mission
        self.pickup = mission.pickup
        self.delivery = mission.delivery
        self.base = mission.base
        self.position = self.base
        self.chemin = [self.position]
        self.positions_recentes = deque(maxlen=NB_POSITIONS_RECENTES)
        self.positions_recentes.append(self.position)

    def objectif(self):
        """Case cible selon la phase courante."""
        if self.phase == Phase.PICKUP:
            return self.pickup
        if self.phase == Phase.DELIVERY:
            return self.delivery
        if self.phase == Phase.BASE:
            return self.base
        # Phase TERMINE (ou inconnue) : on reste sur place.
        return self.position

    def carte_distance_objectif(self):
        """Carte de distances BFS correspondant à la phase courante."""
        if self.phase == Phase.PICKUP:
            return self.mission.dist_pickup
        if self.phase == Phase.DELIVERY:
            return self.mission.dist_delivery
        if self.phase == Phase.BASE:
            return self.mission.dist_base
        return self.mission.dist_base

    def verifier_phase(self):
        """Fait progresser la phase si la cible courante est atteinte.

        CORRECTION : le if/elif garantit qu'on ne change de phase qu'une fois
        par appel (l'enchaînement PICKUP -> DELIVERY -> BASE ne peut pas se
        produire en une seule étape), ce qui correspond au comportement voulu.
        """
        if self.phase == Phase.PICKUP and self.position == self.pickup:
            self.phase = Phase.DELIVERY
        elif self.phase == Phase.DELIVERY and self.position == self.delivery:
            self.phase = Phase.BASE
        elif self.phase == Phase.BASE and self.position == self.base:
            self.phase = Phase.TERMINE
            self.termine = True

    def est_aller_retour(self, nouvelle_position):
        """Vrai si ``nouvelle_position`` revient sur la case précédente.

        CORRECTION : la fonction libre ``est_aller_retour`` de l'original devient
        une méthode (elle ne dépend que du chemin du robot).
        """
        if len(self.chemin) < 2:
            return False
        return (
            nouvelle_position == self.chemin[-2]
            and nouvelle_position != self.position
        )
