"""
Réseau DQN, mémoire de relecture (replay buffer) et agent.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    APPAREIL,
    BATCH_SIZE,
    EPSILON_DEBUT,
    EPSILON_DECROISSANCE,
    EPSILON_FIN,
    FREQUENCE_MAJ_RESEAU_CIBLE,
    GAMMA,
    LR,
    MEMOIRE_MAX,
    NB_ACTIONS,
    TAILLE_CACHEE,
    TAILLE_ETAT,
)


class ReseauDQN(nn.Module):
    """Réseau de neurones : état -> valeurs Q pour chaque action."""

    def __init__(self):
        super().__init__()
        # CORRECTION : les nombres magiques 13 (entrée) et 64 (couches cachées)
        # sont remplacés par les constantes TAILLE_ETAT et TAILLE_CACHEE.
        self.net = nn.Sequential(
            nn.Linear(TAILLE_ETAT, TAILLE_CACHEE),
            nn.ReLU(),
            nn.Linear(TAILLE_CACHEE, TAILLE_CACHEE),
            nn.ReLU(),
            nn.Linear(TAILLE_CACHEE, NB_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class MemoireReplay:
    """Mémoire circulaire stockant les transitions (état, action, r, état', fin)."""

    def __init__(self):
        self.memoire = deque(maxlen=MEMOIRE_MAX)

    def ajouter(self, transition):
        self.memoire.append(transition)

    def taille(self):
        return len(self.memoire)

    def echantillonner(self):
        batch = random.sample(self.memoire, BATCH_SIZE)
        etats, actions, recompenses, etats_suivants, finis = zip(*batch)

        # CORRECTION : les tenseurs sont créés directement sur APPAREIL pour
        # éviter tout "device mismatch" entre le réseau et les données.
        return (
            torch.tensor(np.array(etats), dtype=torch.float32, device=APPAREIL),
            torch.tensor(actions, dtype=torch.long, device=APPAREIL),
            torch.tensor(recompenses, dtype=torch.float32, device=APPAREIL),
            torch.tensor(np.array(etats_suivants), dtype=torch.float32, device=APPAREIL),
            torch.tensor(finis, dtype=torch.float32, device=APPAREIL),
        )


class AgentDQN:
    """Agent DQN : réseau principal + réseau cible + mémoire de relecture."""

    def __init__(self):
        self.reseau = ReseauDQN().to(APPAREIL)
        self.reseau_cible = ReseauDQN().to(APPAREIL)
        self.reseau_cible.load_state_dict(self.reseau.state_dict())

        self.optimiseur = optim.Adam(self.reseau.parameters(), lr=LR)

        # CORRECTION : la fonction de perte était recréée à chaque appel
        # (``nn.MSELoss()`` dans ``apprendre``) ; on l'instancie une seule fois.
        self.critere = nn.MSELoss()

        self.memoire = MemoireReplay()
        self.epsilon = EPSILON_DEBUT
        self.compteur = 0

    def choisir_action(self, etat):
        """Politique epsilon-greedy : exploration aléatoire ou exploitation."""
        if random.random() < self.epsilon:
            return random.randint(0, NB_ACTIONS - 1)

        with torch.no_grad():
            etat_tenseur = torch.tensor(
                etat, dtype=torch.float32, device=APPAREIL
            ).unsqueeze(0)
            q_values = self.reseau(etat_tenseur)
            return int(torch.argmax(q_values).item())

    def apprendre(self):
        """Une étape d'apprentissage sur un mini-lot. Retourne la perte (ou None)."""
        if self.memoire.taille() < BATCH_SIZE:
            return None

        etats, actions, recompenses, etats_suivants, finis = self.memoire.echantillonner()

        q_pred = self.reseau(etats).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.reseau_cible(etats_suivants).max(1)[0]
            q_cible = recompenses + GAMMA * q_next * (1 - finis)

        perte = self.critere(q_pred, q_cible)

        self.optimiseur.zero_grad()
        perte.backward()
        self.optimiseur.step()

        self.compteur += 1

        if self.compteur % FREQUENCE_MAJ_RESEAU_CIBLE == 0:
            self.reseau_cible.load_state_dict(self.reseau.state_dict())

        self.epsilon = max(EPSILON_FIN, self.epsilon * EPSILON_DECROISSANCE)

        # CORRECTION : retour de la perte pour permettre un suivi éventuel
        # (l'original ne renvoyait rien).
        return float(perte.item())
