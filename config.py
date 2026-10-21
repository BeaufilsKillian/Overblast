"""
Configuration globale du projet.

CORRECTION : tous les paramètres dispersés en tête du fichier original sont
regroupés ici dans un module dédié (bonne pratique de découpage), organisés en
sections commentées et réutilisables partout.

Une graine unique garantit la reproductibilité des tirages aléatoires.
"""

import os

# CORRECTION : ``setdefault`` n'écrase pas une éventuelle valeur déjà définie
# par l'utilisateur (l'original forçait toujours la valeur).
# Évite un doublon de bibliothèque OpenMP (principalement macOS).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random

import numpy as np
import torch

# CORRECTION : limitation à 1 thread CPU pour des mesures reproductibles
# (l'original l'imposait déjà, on le garde dans le module de config).
torch.set_num_threads(1)

# Appareil d'exécution : GPU si disponible, sinon CPU.
# CORRECTION : l'original ne gérait pas l'appareil ; les tenseurs sont désormais
# placés sur APPAREIL pour éviter les erreurs de "device mismatch" et profiter
# du GPU quand il existe.
APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Environnement (entrepôt + robots)
# ---------------------------------------------------------------------------

NB_ROBOTS = 3

HAUTEUR = 30
LARGEUR = 45

# Coûts en énergie
COUT_DROIT = 1
COUT_VIRAGE = 5
COUT_COLLISION = 100
COUT_REPASSAGE_RECENT = 50
# CORRECTION : le coût de l'action "rester sur place" était codé en dur
# (``cout = 1``) dans la simulation ; on le remonte en paramètre.
COUT_ATTENTE = 1

# Récompenses.
# CORRECTION : les valeurs magiques de récompense dispersées dans la simulation
# (-0.1, -15.0, -20.0, -12.0, +2.5, -1.2, +25.0, +80.0) sont regroupées ici.
RECOMPENSE_ETAPE = -0.1
RECOMPENSE_COLLISION = -15.0
RECOMPENSE_MOUVEMENT_INTERDIT = -20.0
RECOMPENSE_REPASSAGE = -12.0
RECOMPENSE_RAPPROCHEMENT = 2.5
RECOMPENSE_ELOIGNEMENT = -1.2
RECOMPENSE_CHANGEMENT_PHASE = 25.0
RECOMPENSE_MISSION_TERMINEE = 80.0

# Seuil d'énergie au-delà duquel l'épisode est considéré comme un échec.
SEUIL_ECHEC_ENERGIE = 750

# Une mission doit imposer un trajet total d'au moins cette distance.
DISTANCE_MIN_TOTALE_MISSION = 70
TEMPS_MAX_EPISODE = 300

# Nombre d'allers-retours consécutifs tolérés avant blocage du mouvement.
MAX_ALLERS_RETOURS = 3

# Nombre de cases récentes mémorisées pour détecter le repassage.
NB_POSITIONS_RECENTES = 3

# CORRECTION : le "15000" codé en dur (nombre de tirages aléatoires pour
# générer une mission) devient un paramètre.
NB_TIRAGES_MISSION = 15000

# Déplacements : indices 0..3 = haut, bas, gauche, droite ; 4 = rester sur place.
DEPLACEMENTS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (0, 0),
]
NB_ACTIONS = len(DEPLACEMENTS)
INDICE_ATTENTE = 4


# ---------------------------------------------------------------------------
# Apprentissage DQN
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
MEMOIRE_MAX = 8000
GAMMA = 0.95
LR = 0.001

EPSILON_DEBUT = 1.0
EPSILON_FIN = 0.05
EPSILON_DECROISSANCE = 0.992

# CORRECTION : le "50" codé en dur (fréquence de mise à jour du réseau cible)
# est remonté en paramètre.
FREQUENCE_MAJ_RESEAU_CIBLE = 50

# CORRECTION : le "temps % 5" codé en dur (fréquence d'apprentissage) est
# remonté en paramètre.
FREQUENCE_APPRENTISSAGE = 5

# CORRECTION : la dimension de l'état (13) et la taille des couches cachées (64)
# sont nommées au lieu d'être des nombres magiques dans le réseau.
TAILLE_ETAT = 13
TAILLE_CACHEE = 64


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

PAUSE_ANIMATION = 0.035
COULEURS_ROBOTS = ["red", "blue", "orange"]


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

GRAINE = 42


def initialiser_graines():
    """CORRECTION : initialisation des graines factorisée dans une fonction,
    au lieu des trois lignes éparpillées en tête du fichier original."""
    random.seed(GRAINE)
    np.random.seed(GRAINE)
    torch.manual_seed(GRAINE)
