import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def plot_decision_surface(X, y, model, scaler, title, filename):
    h = .05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Prédire sur la grille (en n'oubliant pas de scaler si nécessaire)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap='RdYlGn_r', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn_r', edgecolors='k', s=20)
    plt.xlabel('Turbidité (NTU)')
    plt.ylabel('Oxygène Dissous (mg/L)')
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    df = pd.read_csv('eco_sentinel_dataset.csv')
    X = df[['Turbidite_NTU', 'Oxygene_Dissous_mgL']].values
    y = df['Etat_Eau'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling est crucial pour SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Eco_LogReg": LogisticRegression(),
        "Eco_SVM_Linear": SVC(kernel='linear'),
        "Eco_SVM_RBF": SVC(kernel='rbf', C=1.0, gamma='scale')
    }
    
    summary = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        summary.append(f"Modèle: {name}\nAccuracy: {acc:.4f}\n")
        
        # Visualisation de la surface (sur les données scalées pour la cohérence)
        plot_decision_surface(X_train_scaled, y_train, model, scaler, f"Zone de Protection : {name}", f"{name}_surface.png")
        
        # Matrice de confusion
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Sain', 'Pollué'], yticklabels=['Sain', 'Pollué'])
        plt.title(f"Matrice de Confusion - {name}")
        plt.savefig(f"{name}_cm.png")
        plt.close()
        
    with open('eco_results.txt', 'w') as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    main()
