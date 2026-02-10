import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def plot_decision_surface(X, y, model, title, filename):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    # Charger les données
    df = pd.read_csv('dataset.csv')
    X = df[['Intensite', 'Frequence']].values
    y = df['Label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modèles
    models = {
        "Regression_Logistique": LogisticRegression(),
        "SVM_Lineaire": SVC(kernel='linear'),
        "SVM_Kernel_RBF": SVC(kernel='rbf')
    }
    
    results = []
    
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Métriques
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            "name": name,
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": cr
        })
        
        # Visualisation
        plot_decision_surface(X, y, model, f"Surface de décision : {name}", f"{name}_surface.png")
        
        # Matrice de confusion (Plot)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matrice de Confusion : {name}")
        plt.ylabel('Réel')
        plt.xlabel('Prédit')
        plt.savefig(f"{name}_cm.png")
        plt.close()
        
        print(f"Modèle {name} terminé.")

    # Sauvegarder les résultats textuels pour le rapport
    with open('results_summary.txt', 'w') as f:
        for res in results:
            f.write(f"--- {res['name']} ---\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(pd.DataFrame(res['report']).transpose().to_string())
            f.write("\n\n")

if __name__ == "__main__":
    main()
