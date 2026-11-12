"""Régression linéaire from scratch (numpy uniquement) : prix d'un logement.

ŷ = w₁x₁ + ... + wₙxₙ + b, entraîné par descente de gradient sur une MSE.
Les 5 étapes : données -> modèle -> entraînement -> généralisation -> inférence.
"""
import json
import sys
import numpy as np

rng = np.random.default_rng(42)
FEATURES = ["superficie_m2", "pieces", "distance_centre_km", "anciennete_ans"]

# --- 1. Collecter et préparer les données ------------------------------------
# Données synthétiques : on connaît la "vraie" relation, donc on peut vérifier
# que le modèle la retrouve. (Localisation simplifiée en distance au centre.)
def generer_donnees(n=500):
    X = np.column_stack([
        rng.uniform(20, 150, n),   # superficie
        rng.integers(1, 7, n),     # pièces
        rng.uniform(0, 30, n),     # distance au centre
        rng.uniform(0, 80, n),     # ancienneté
    ])
    vrais_w = np.array([3000, 5000, -2500, -400])
    y = X @ vrais_w + 50_000 + rng.normal(0, 15_000, n)  # bruit = réalité
    return X, y

X, y = generer_donnees()

# Répartition train / validation / test (60 / 20 / 20), après mélange.
idx = rng.permutation(len(X))
n_tr, n_va = int(0.6 * len(X)), int(0.2 * len(X))
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

# Normalisation : moyenne/écart-type calculés sur le train UNIQUEMENT
# (sinon fuite d'information du test vers l'entraînement).
mu, sigma = X[tr].mean(axis=0), X[tr].std(axis=0)
y_mu, y_sigma = y[tr].mean(), y[tr].std()
norm_x = lambda a: (a - mu) / sigma
Xtr, Xva, Xte = norm_x(X[tr]), norm_x(X[va]), norm_x(X[te])
ytr, yva, yte = (y[tr] - y_mu) / y_sigma, (y[va] - y_mu) / y_sigma, (y[te] - y_mu) / y_sigma

# --- 2. Choisir un modèle ----------------------------------------------------
w = np.zeros(Xtr.shape[1])   # coefficients, mal adaptés au départ
b = 0.0                      # biais
predire = lambda X_: X_ @ w + b

# --- 3. Entraîner ------------------------------------------------------------
perte = lambda X_, y_: np.mean((predire(X_) - y_) ** 2)  # loss : MSE
LR, EPOCHS = 0.1, 200                                    # hyperparamètres

historique = {"train": [], "val": []}
for epoch in range(EPOCHS):
    erreur = predire(Xtr) - ytr
    w -= LR * (2 / len(ytr)) * Xtr.T @ erreur            # gradient de la MSE
    b -= LR * 2 * erreur.mean()
    historique["train"].append(perte(Xtr, ytr))
    historique["val"].append(perte(Xva, yva))
    if epoch % 40 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d}  loss train={perte(Xtr, ytr):.4f}  val={perte(Xva, yva):.4f}")

# --- 4. Vérifier la généralisation -------------------------------------------
# Le test n'est regardé qu'une fois, à la fin.
eur = lambda X_, y_: np.mean(np.abs(predire(X_) - y_)) * y_sigma
print(f"\nErreur absolue moyenne  train={eur(Xtr, ytr):,.0f} €  "
      f"val={eur(Xva, yva):,.0f} €  test={eur(Xte, yte):,.0f} €")

# RMSE (en €, pénalise davantage les grosses erreurs) et R² (part de la variance
# du prix expliquée par le modèle : 1 = parfait, 0 = pas mieux que prédire la moyenne).
# Le R² est invariant par normalisation, on peut le calculer sur les valeurs normalisées.
# MSE en unités d'origine (€²) : la loss d'entraînement remise à l'échelle des prix.
mse = lambda X_, y_: perte(X_, y_) * y_sigma ** 2
rmse = lambda X_, y_: np.sqrt(mse(X_, y_))
r2 = lambda X_, y_: 1 - np.sum((y_ - predire(X_)) ** 2) / np.sum((y_ - y_.mean()) ** 2)
for nom, X_, y_ in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
    print(f"{nom:5s}  MSE={mse(X_, y_):,.0f} €²   RMSE={rmse(X_, y_):9,.0f} €   R²={r2(X_, y_):.3f}")

# Retour aux unités d'origine pour lire les coefficients.
w_reel = w * y_sigma / sigma
b_reel = y_mu + b * y_sigma - w_reel @ mu
print("Coefficients appris (vrais : 3000, 5000, -2500, -400 ; biais 50000) :")
for nom, c in zip(FEATURES, w_reel):
    print(f"  {nom:20s} {c:9.0f}")
print(f"  {'biais':20s} {b_reel:9.0f}")

# --- 5. Inférence ------------------------------------------------------------
# Les paramètres ne bougent plus : on les applique à un logement inédit.
nouveau = np.array([[75, 3, 5, 20]])
print(f"\nEstimation pour {dict(zip(FEATURES, nouveau[0]))} : "
      f"{(nouveau @ w_reel + b_reel)[0]:,.0f} €")

# Export des paramètres : c'est tout ce dont l'inférence a besoin
# (réutilisable ailleurs, p. ex. dans un Worker Cloudflare).
with open("01_regression_lineaire/model.json", "w") as f:
    json.dump({"features": FEATURES, "w": w_reel.tolist(), "b": float(b_reel)}, f, indent=2)

# --- Visualisation (optionnelle) : python regression_lineaire.py --plot ------
if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(historique["train"], label="train")
    ax[0].plot(historique["val"], label="validation", linestyle="--")
    ax[0].set(title="Courbe de loss (MSE normalisée)", xlabel="epoch", ylabel="loss", yscale="log")
    ax[0].legend()

    reel = yte * y_sigma + y_mu
    estime = predire(Xte) * y_sigma + y_mu
    ax[1].scatter(reel / 1e3, estime / 1e3, s=15, alpha=0.7)
    lim = [min(reel.min(), estime.min()) / 1e3, max(reel.max(), estime.max()) / 1e3]
    ax[1].plot(lim, lim, "r--", label="prédiction parfaite")
    ax[1].set(title="Test : prix prédit vs réel", xlabel="prix réel (k€)", ylabel="prix prédit (k€)")
    ax[1].legend()

    vrais = np.array([3000, 5000, -2500, -400])
    pos = np.arange(len(FEATURES))
    ax[2].bar(pos - 0.2, vrais, 0.4, label="vrais")
    ax[2].bar(pos + 0.2, w_reel, 0.4, label="appris")
    ax[2].set_xticks(pos, ["superficie\n(m²)", "pièces", "distance\ncentre (km)", "ancienneté\n(ans)"])
    ax[2].set(title="Coefficients (€ par unité)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()
