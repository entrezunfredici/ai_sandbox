# 01 – Régression linéaire : estimer le prix d'un logement

Script : [regression_lineaire.py](regression_lineaire.py) (Python + numpy, sans framework de ML).

## Lancer le script

Depuis la racine du dépôt (`ai_sandbox/`) :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"   # pour afficher correctement le symbole €
python 01_regression_lineaire/regression_lineaire.py
```

Le script doit être lancé depuis la racine : il écrit `01_regression_lineaire/model.json` avec un chemin relatif.

Les données sont tirées au hasard avec une graine fixe (`default_rng(42)`) : les résultats sont identiques à chaque exécution.

## Afficher les graphiques

```powershell
python 01_regression_lineaire/regression_lineaire.py --plot
```

Une fenêtre matplotlib s'ouvre avec trois graphiques : la courbe de loss (train vs validation, échelle log), le prix prédit vs réel sur le jeu de test (les points doivent suivre la diagonale), et les coefficients appris comparés aux vrais.

## Lire la sortie

### 1. Courbe de loss

```
epoch   0  loss train=0.6342  val=0.6010
epoch  40  loss train=0.0154  val=0.0153
```

- La **loss** est l'erreur quadratique moyenne (MSE) calculée sur le prix **normalisé** (moyenne 0, écart-type 1). Ce n'est donc pas une valeur en euros.
- Elle doit baisser puis se stabiliser : le modèle a convergé (ici dès ~40 epochs).
- Compare **train** et **val** :
  - proches l'une de l'autre : bon signe, le modèle généralise ;
  - train très inférieure à val : surapprentissage (overfitting).

### 2. Erreur en euros

```
Erreur absolue moyenne  train=11,702 €  val=11,284 €  test=11,292 €
```

C'est l'écart moyen entre prix prédit et prix réel, en euros. C'est la ligne la plus parlante.

- Le **test** est le chiffre honnête : ces données n'ont servi ni à l'entraînement ni au réglage.
- Ici ~11 k€ correspond au bruit injecté dans les données (écart-type 15 k€) : le modèle ne peut pas faire mieux que ce bruit.

### 3. Coefficients

```
superficie_m2     3034     -> +3 034 € par m² supplémentaire
pieces            5066     -> +5 066 € par pièce
distance_centre  -2426     -> -2 426 € par km d'éloignement
anciennete_ans    -474     -> -474 € par année d'ancienneté
biais            47712     -> prix de base quand toutes les variables valent 0
```

Chaque coefficient se lit « toutes choses égales par ailleurs ». Les vrais coefficients des données synthétiques sont 3000 / 5000 / -2500 / -400 / biais 50000 : l'écart vient du bruit.

Ce sont des corrélations statistiques apprises dans les données, pas une compréhension du marché.

### 4. Inférence

```
Estimation pour {'superficie_m2': 75, 'pieces': 3, ...} : 268,849 €
```

Prédiction pour un logement absent des données d'entraînement. Pour tester un autre logement, modifie le tableau `nouveau` en bas du script (ordre : superficie, pièces, distance au centre, ancienneté).

## Le fichier `model.json`

```json
{ "features": ["superficie_m2", ...], "w": [3034, 5066, ...], "b": 47712 }
```

Ce sont les paramètres appris, **en unités d'origine** (m², km, €), donc utilisables sans normaliser les entrées. L'inférence se réduit à :

```
prix = w[0]*superficie + w[1]*pieces + w[2]*distance + w[3]*anciennete + b
```

Ce fichier sera repris pour le déploiement (Worker Cloudflare). Il est régénéré à chaque exécution.

## Pistes pour expérimenter

- Changer `LR` ou `EPOCHS` et observer la courbe de loss (un `LR` trop grand fait diverger).
- Augmenter le bruit (`rng.normal(0, 15_000, n)`) : l'erreur en euros monte, les coefficients deviennent moins précis.
- Réduire `n` à 30 exemples : les coefficients deviennent instables.
