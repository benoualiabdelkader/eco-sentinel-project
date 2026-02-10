import pandas as pd
import numpy as np

def generate_legendary_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    # Caractéristiques : Turbidité (clarté de l'eau) et Niveau d'Oxygène
    # On crée une séparation non-linéaire complexe (cercle/anneau)
    # La classe 0 (Sain) est au centre, la classe 1 (Pollué) est en périphérie
    
    radius = np.random.uniform(0, 1, n_samples)
    angle = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Classe 0 : Eau saine (centre)
    # Classe 1 : Eau polluée (périphérie)
    y = (radius > 0.6).astype(int)
    
    # Ajouter du bruit pour rendre le challenge réel
    noise = np.random.normal(0, 0.1, n_samples)
    turbidity = radius * np.cos(angle) + noise
    oxygen = radius * np.sin(angle) + noise
    
    # Mise à l'échelle pour des valeurs réalistes
    # Turbidité (NTU) : 0 à 10
    # Oxygène (mg/L) : 0 à 14
    turbidity = (turbidity - turbidity.min()) / (turbidity.max() - turbidity.min()) * 10
    oxygen = (oxygen - oxygen.min()) / (oxygen.max() - oxygen.min()) * 14
    
    df = pd.DataFrame({
        'Turbidite_NTU': turbidity,
        'Oxygene_Dissous_mgL': oxygen,
        'Etat_Eau': y
    })
    
    # Ajouter des colonnes de contexte pour le "look" CSV
    df['Station_ID'] = [f"SENSOR_{np.random.randint(100, 999)}" for _ in range(n_samples)]
    df['Timestamp'] = pd.date_range(start='2026-01-01', periods=n_samples, freq='H')
    
    # Réorganiser les colonnes
    df = df[['Timestamp', 'Station_ID', 'Turbidite_NTU', 'Oxygene_Dissous_mgL', 'Etat_Eau']]
    
    df.to_csv('eco_sentinel_dataset.csv', index=False)
    print("Dataset 'eco_sentinel_dataset.csv' généré avec succès.")

if __name__ == "__main__":
    generate_legendary_dataset()
