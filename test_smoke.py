"""Test de fumée : vérifie que le projet refactorisé tourne de bout en bout.

Lancement : ``.venv/Scripts/python.exe test_smoke.py``
"""

import matplotlib

matplotlib.use("Agg")  # aucun affichage de fenêtre pendant le test

import config
import entrepot
import chemins
import missions
import dqn
import robot
import etat
import simulation
import affichage

config.initialiser_graines()

# --- Entrepôt -----------------------------------------------------------
grille, bases = entrepot.creer_entrepot()
assert grille.shape == (config.HAUTEUR, config.LARGEUR)
assert len(bases) == config.NB_ROBOTS
print(f"[OK] entrepot : {grille.shape}, {len(bases)} bases")

# --- BFS ----------------------------------------------------------------
assert chemins.distance_bfs(grille, bases[0], bases[0]) == 0
d = chemins.distance_bfs(grille, bases[0], bases[1], robot_id=0, bases=bases)
print(f"[OK] BFS : base0 -> base1 (robot 0) = {d}")

# OPTIMISATION : vérifie que le BFS à arrêt anticipé donne EXACTEMENT le même
# résultat que le BFS complet, sur des paires aléatoires.
import random as _r

_r.seed(0)
libres = entrepot.positions_libres(grille)
for _ in range(50):
    a = _r.choice(libres)
    b = _r.choice(libres)
    assert chemins.distance_point_a_point(grille, a, b, 0, bases) == \
        chemins.distance_bfs(grille, a, b, 0, bases)
print("[OK] distance_point_a_point == distance_bfs (50 paires)")

# --- Missions (tirages réduits pour un test rapide) ---------------------
missions.NB_TIRAGES_MISSION = 2000
missions_bloc = missions.generer_missions_bloc(grille, bases)
assert len(missions_bloc) == config.NB_ROBOTS
for m in missions_bloc:
    assert m.distance_minimale >= config.DISTANCE_MIN_TOTALE_MISSION
print(f"[OK] missions : {len(missions_bloc)} missions générées")

# --- État DQN -----------------------------------------------------------
r = robot.Robot(0, bases[0])
r.definir_mission(missions_bloc[0])
etat = etat.construire_etat(r, grille, {r.position}, bases)
assert len(etat) == config.TAILLE_ETAT, f"taille état = {len(etat)}"
print(f"[OK] état DQN : shape={etat.shape}, dtype={etat.dtype}")

# --- Agent + simulation -------------------------------------------------
agent = dqn.AgentDQN()
for i in range(6):
    res = simulation.simuler_iteration(agent, grille, bases, missions_bloc)
    print(
        f"[--] épisode {i}: succes={res['succes']} "
        f"energie={res['energie_moyenne']:.2f} "
        f"collisions={res['collisions']} temps={res['temps']}"
    )

assert agent.memoire.taille() >= config.BATCH_SIZE
perte = agent.apprendre()
assert perte is not None, "la mémoire doit être pleine pour apprendre"
print(f"[OK] apprentissage DQN : perte = {perte:.4f}")

# --- Affichage 2D (axes explicite, sans fenêtre) ------------------------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
affichage.dessiner_2d(grille, bases, res["robots"], res["historique"], 0, "test", ax=ax)
plt.close(fig)
print("[OK] dessiner_2d (axes explicite)")

# --- Moyenne mobile -----------------------------------------------------
lisse = affichage.moyenne_mobile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
assert len(lisse) == 6
print(f"[OK] moyenne mobile : {lisse}")

print("\n=== SMOKE TEST OK ===")
