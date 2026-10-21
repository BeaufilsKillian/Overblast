# Entrepôt multi-robots — DQN

Simulation de 3 robots dans un entrepôt apprenant par DQN à réaliser des
missions (aller chercher un colis puis le livrer) tout en minimisant leur
consommation d'énergie.

## Structure du projet

| Fichier        | Rôle                                                              |
| -------------- | ----------------------------------------------------------------- |
| `config.py`    | Tous les paramètres (environnement, DQN, affichage) + graines     |
| `entrepot.py`  | Génération de la grille + tests d'occupation des cases            |
| `chemins.py`   | Distances BFS sur la grille                                       |
| `missions.py`  | Génération des missions (dataclass `Mission`)                     |
| `dqn.py`       | Réseau, mémoire de relecture, agent DQN                           |
| `robot.py`     | Classe `Robot` + phases (enum `Phase`)                            |
| `etat.py`      | Construction du vecteur d'état                                    |
| `simulation.py`| Simulation d'un épisode (boucle décision/récompense)              |
| `affichage.py` | Affichage 2D + courbe d'énergie                                   |
| `main.py`      | Point d'entrée : boucle d'apprentissage + bilan                   |

## Lancer

```bash
python main.py
```

Dépendances : `torch`, `numpy`, `matplotlib`.

Toutes les corrections par rapport au fichier d'origine sont signalées dans le
code par des commentaires `# CORRECTION : ...`.
