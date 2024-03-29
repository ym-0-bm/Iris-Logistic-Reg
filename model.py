import joblib
import numpy as np


def load(chemin_modele):
    """Charge le modèle entraîné depuis le chemin spécifié."""
    try:
        return joblib.load(chemin_modele)
    except FileNotFoundError:
        raise Exception(f"Fichier modèle introuvable : {chemin_modele}")


def predict(modele, data):
    """Effectue une prédiction en utilisant le modèle de régression logistique chargé et retourne une valeur écrite."""

    # Prédiction des probabilités
    prediction = modele.predict_proba(data)

    # Conversion en valeur écrite basée sur la probabilité la plus élevée
    SpeciesPredict = np.argmax(prediction, axis=1)[0]  # Index d'espèce avec la plus haute probabilité

    if SpeciesPredict == 0:
        return "Iris Setosa"
    elif SpeciesPredict == 1:
        return "Iris Versicolor"
    else:
        return "Iris Virginica"
