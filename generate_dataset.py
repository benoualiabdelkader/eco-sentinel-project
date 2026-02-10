import pandas as pd
import numpy as np
from sklearn.datasets import make_moons

def generate_custom_dataset():
    # Générer des données non-linéairement séparables (forme de lunes)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Créer un DataFrame avec des noms de caractéristiques descriptifs
    # Imaginons un cas d'usage : Classification de signaux (Intensité vs Fréquence)
    df = pd.DataFrame(X, columns=['Intensite', 'Frequence'])
    df['Label'] = y
    
    # Sauvegarder en CSV
    df.to_csv('dataset.csv', index=False)
    print("Dataset 'dataset.csv' généré avec succès.")

if __name__ == "__main__":
    generate_custom_dataset()
