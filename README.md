# Overblast -- Optimisation énergétique de robots autonomes en entrepôt

**Projet de TIPE -- CPGE TSI | Python • PyTorch • NumPy • Matplotlib**

> **Problématique :** Dans quelle mesure l'intelligence artificielle peut-elle améliorer l'efficacité des robots évoluant dans un entrepôt ?

**Overblast** est une simulation en deux dimensions de **trois robots mobiles** chargés de missions de prise en charge et de livraison de colis (*Pick-up & Delivery*). Le projet étudie l'utilisation de l'**apprentissage par renforcement**, à travers un agent **DQN (Deep Q-Network)**, pour apprendre à se déplacer en limitant les dépenses énergétiques et les mouvements inefficaces.

---

## 1. Installation

### Prérequis

- Un ordinateur avec **Python** installé (Python 3.10 ou 3.11 conseillé) et un terminal, par exemple celui de **Visual Studio Code**.
- **Git** pour récupérer le dépôt, ou la possibilité de télécharger l'archive ZIP depuis GitHub.
- Les bibliothèques **PyTorch**, **NumPy** et **Matplotlib**, installées à l'aide du fichier `requirements.txt`.

L'exécution est possible sur **CPU** ; le programme peut également utiliser **CUDA** si une installation compatible de PyTorch et un GPU adapté sont disponibles.

### Télécharger le projet

Dans le terminal **PowerShell** de VS Code :

```powershell
git clone https://github.com/BeaufilsKillian/Overblast.git
cd Overblast
```

Sans Git, choisir **Code → Download ZIP** sur la page du dépôt, extraire l'archive, puis ouvrir le dossier dans VS Code.

### Créer un environnement Python et installer les dépendances

Depuis le dossier `Overblast` :

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si la commande `py` n'est pas reconnue, essayer `python -m venv .venv`. Cette méthode utilise directement le Python de l'environnement virtuel, sans devoir l'activer dans PowerShell.

### Lancer la simulation

```powershell
.\.venv\Scripts\python.exe main.py
```

Le terminal demande alors :

1. Le **nombre total d'itérations** d'apprentissage (valeur par défaut : **10 000**).
2. La **fréquence des animations** (valeur par défaut : une animation toutes les **2 500** itérations).

Appuyer sur **Entrée** pour conserver la valeur proposée. Pour un premier essai, on peut saisir un petit nombre d'itérations. La préparation initiale des missions peut toutefois prendre du temps, car elle implique des calculs de distances sur l'entrepôt.

### Vérifier les principaux modules

```powershell
.\.venv\Scripts\python.exe test_smoke.py
```

Le fichier `test_smoke.py` effectue un ensemble de vérifications élémentaires (entrepôt, distances, missions, état de l'agent, simulation et affichage). Il ne remplace pas une validation expérimentale complète.

**Configuration :** dans la version de ce dépôt, les paramètres se modifient directement dans `config.py` : dimensions de la carte, coûts d'énergie, seuil d'échec, nombre d'itérations, paramètres du DQN et vitesse des animations. **Aucun fichier `.env` n'est requis** par cette version.

---

## 2. Architecture du projet

Le programme est **modulaire** : chaque fichier est responsable d'une partie précise de la simulation.

```text
Overblast/
├── main.py             # Lancement et boucle d'apprentissage
├── config.py           # Constantes, parametres et graine aleatoire
├── entrepot.py         # Generation de l'entrepot et cases autorisees
├── chemins.py          # Recherche de distances par BFS
├── missions.py         # Generation des missions de collecte/livraison
├── robot.py            # Etat interne et phases de chaque robot
├── etat.py             # Informations transmises au DQN
├── dqn.py              # Reseau de neurones, agent et memoire de rejeu
├── simulation.py       # Deplacements, interactions et recompenses
├── affichage.py        # Animations 2D et courbes d'energie
├── benchmark.py        # Mesures de performance de calcul
├── test_smoke.py       # Tests elementaires de fonctionnement
├── requirements.txt    # Dependances Python
├── .gitignore          # Fichiers exclus du suivi Git
└── README.md           # Documentation du projet
```

### Organisation générale

```text
                    main.py
                       |
          +------------+------------+
          |                         |
      config.py                 entrepot.py
                                    |
                             chemins.py (BFS)
                                    |
                               missions.py
                                    |
                     +--------------+--------------+
                     |                             |
                 robot.py                       etat.py
                     |                             |
                     +-----------+-----------------+
                                 |
                              dqn.py
                                 |
                           simulation.py
                                 |
                            affichage.py
```

Ce schéma simplifie les relations : la simulation utilise plusieurs modules simultanément, tandis que `main.py` coordonne les itérations, les missions et les animations.

### Technologies utilisées

| Technologie | Rôle |
| --- | --- |
| **Python** | Langage principal de développement |
| **PyTorch** | Réseau de neurones et entraînement DQN |
| **NumPy** | Représentation et calcul sur les matrices de l'entrepôt |
| **Matplotlib** | Affichage de la simulation et des courbes |
| **BFS** (*Breadth-First Search*) | Calcul de distances minimales en nombre de cases, en tenant compte des obstacles |

---

## 3. Comment fonctionne le projet ?

### 3.1. Modélisation de l'entrepôt

L'environnement est une **grille 2D de 30 × 45 cases** dans la configuration publiée. Des zones bloquées représentent les étagères et des cases libres forment les allées. Trois bases distinctes, associées aux robots **rouge, bleu et orange**, se trouvent dans une zone dégagée.

Chaque robot possède sa propre base de départ et d'arrivée. Il ne peut pas traverser celle d'un autre robot. Les positions des autres robots et les étagères sont également prises en compte pour détecter les déplacements impossibles.

### 3.2. Déroulement d'une mission

Chaque robot doit accomplir les étapes suivantes :

1. **Départ :** quitter sa base individuelle.
2. **Collecte (*pick-up*) :** rejoindre la position du colis.
3. **Livraison (*delivery*) :** rejoindre la destination indiquée.
4. **Retour :** revenir à sa base personnelle.

Une itération est considérée comme **réussie lorsque les trois robots ont terminé leurs missions et retrouvé leur base**. Sinon, la simulation peut s'arrêter à cause d'une limite de pas ou d'un dépassement énergétique.

Les missions sont générées par `missions.py` à partir des zones accessibles. Le code utilise des distances **BFS** pour mesurer les plus courts trajets en nombre de déplacements. Dans la version actuelle, la constante `DISTANCE_MIN_TOTALE_MISSION` vaut **70** ; elle peut être modifiée dans `config.py`.

### 3.3. Comment le DQN prend ses décisions

Le **DQN** est un algorithme d'apprentissage par renforcement : un réseau de neurones attribue une valeur estimée à chaque action possible, à partir de la situation observée.

L'agent reçoit un vecteur de **13 informations**, notamment la position du robot, la position de l'objectif, une direction locale suggérée par le BFS, la phase de mission, l'occupation des cases voisines et un indicateur de risque de repassage récent.

À chaque décision, il choisit parmi **5 actions** : aller en haut, en bas, à gauche, à droite ou rester sur place.

L'apprentissage repose sur :

- une stratégie **epsilon-greedy**, alternant exploration aléatoire et exploitation des valeurs apprises ;
- une **mémoire de rejeu** (*replay buffer*) contenant les transitions vécues ;
- un **réseau principal** et un **réseau cible**, mis à jour périodiquement ;
- des **récompenses et pénalités** orientant le comportement du robot.

Le dépôt publié implémente un **DQN avec réseau cible** ; il ne faut pas le confondre avec un **Double DQN**, qui utilise une règle différente de sélection et d'évaluation des actions futures.

### 3.4. Modèle de consommation énergétique

L'objectif est de limiter les mouvements coûteux. Les valeurs définies dans `config.py` sont :

| Événement | Coût énergétique |
| --- | ---: |
| Déplacement en ligne droite | **+1** |
| Virage | **+5** supplémentaires |
| Collision ou déplacement impossible traité comme collision | **+100** |
| Retour sur une case récemment visitée | **+50** |
| Attente | **+1** |

Le système pénalise également les allers-retours consécutifs. Les récompenses d'apprentissage sont définies séparément des coûts énergétiques : un rapprochement de l'objectif, une étape accomplie ou une mission terminée peut donner une récompense positive.

**Seuil d'échec :** un épisode est déclaré en échec lorsque **l'énergie moyenne dépensée par les trois robots dépasse 750 unités**. La durée maximale d'un épisode est réglée par `TEMPS_MAX_EPISODE` (300 pas dans la version publiée).

### 3.5. Entraînement et visualisation

`main.py` lance une succession d'itérations. Les missions restent identiques entre deux animations, puis de nouveaux objectifs sont générés au début du bloc suivant.

Le terminal permet de suivre notamment le **succès ou l'échec**, le **nombre de pas**, l'**énergie moyenne**, les **collisions**, les **repassages** et le niveau d'exploration **epsilon**.

Les animations réalisées avec Matplotlib montrent les robots, leurs bases, les points de collecte et de livraison ainsi que leurs trajectoires. Elles peuvent être relancées avec le bouton **« Recommencer »**. Le bilan final présente aussi une **courbe d'énergie brute et lissée**, ainsi qu'un repère correspondant au seuil de 750 unités.

---

## 4. À quoi sert Overblast ?

### Objectif scientifique

Overblast constitue un **outil d'expérimentation** pour étudier la question : **un algorithme d'apprentissage par renforcement peut-il améliorer l'efficacité énergétique d'un ensemble de robots mobiles dans un entrepôt ?**

L'entrepôt sert de terrain d'essai contrôlé : les missions, les obstacles et les règles de dépense énergétique sont simulés, ce qui permet d'observer et de comparer les comportements sans disposer de robots physiques.

### Ce que le projet permet d'étudier

- **Consommation :** vérifier si l'énergie moyenne diminue au cours de l'entraînement.
- **Déplacements :** observer l'effet des virages, collisions et répétitions de trajectoire.
- **Coordination :** mettre en évidence les difficultés liées à la présence de trois robots partageant les mêmes allées.
- **Apprentissage :** comparer les trajectoires produites à différentes étapes de l'entraînement.
- **Reproductibilité :** modifier les paramètres et relancer la simulation avec une graine aléatoire fixée.

Le projet dispose également de fichiers de **tests** et de **benchmark** pour contrôler certains comportements et mesurer les performances de calcul.

### Portée et limites

Overblast est un **prototype de simulation à vocation pédagogique et expérimentale**, pas un système prêt à piloter des robots industriels réels. Son modèle énergétique est une approximation volontairement simplifiée : les coûts représentent des **unités simulées**, et non une consommation électrique mesurée en joules ou en wattheures.

L'intérêt du TIPE est de **mesurer les résultats effectivement obtenus** et de discuter les limites de l'apprentissage : un DQN n'assure pas automatiquement le meilleur trajet, et une diminution de l'énergie ne suffit pas, à elle seule, à prouver une amélioration générale du système.

---

**Auteur :** Killian Beaufils  
**Cadre :** TIPE — Classes préparatoires aux grandes écoles, filière TSI  
**Dépôt :** https://github.com/BeaufilsKillian/Overblast
