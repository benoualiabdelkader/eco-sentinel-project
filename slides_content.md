# Projet Sentinelle Écologique : Détection Intelligente de la Pollution Aquatique

## Diapositive 1: Titre

# Sentinelle Écologique
## Détection Intelligente de la Pollution Aquatique

*   **Un projet pour un futur plus propre**
*   **Par : [Votre Nom]**

## Diapositive 2: L'Urgence Aquatique

### Nos Eaux en Péril

*   **La pollution aquatique : une menace invisible mais dévastatrice.**
*   **Impacts :** Écosystèmes détruits, santé humaine menacée, ressources vitales compromises.
*   **Le défi :** Détecter rapidement et précisément la pollution pour agir efficacement.

## Diapositive 3: La Solution : Sentinelle Écologique

### Notre Réponse : L'IA au Service de l'Environnement

*   **Sentinelle Écologique :** Un système intelligent basé sur l'apprentissage automatique.
*   **Objectif :** Classifier l'état de l'eau (Sain / Pollué) en temps réel.
*   **Technologie :** Capteurs IoT pour des données précises, modèles ML pour l'analyse.

## Diapositive 4: Le Cœur du Projet : Notre Dataset "Intelligent"

### Des Données qui Racontent une Histoire

*   **Source :** Simulation de capteurs IoT (Turbidité, Oxygène Dissous).
*   **Le "Piège" :** Une séparation non-linéaire complexe (eau saine au centre, polluée en périphérie).
*   **Pourquoi ?** Pour défier les modèles et révéler la puissance des approches avancées.

## Diapositive 5: Les Candidats : Modèles de Classification

### Qui Sera le Gardien de l'Eau ?

*   **Régression Logistique :** Le classique linéaire. Simple, mais suffisant ?
*   **SVM Linéaire :** Le champion de la séparation linéaire. Peut-il faire mieux ?
*   **SVM avec Noyau RBF :** Le magicien non-linéaire. La clé de la solution ?

## Diapositive 6: La Bataille des Modèles : Résultats

### La Vérité Révélée par les Chiffres

| Modèle | Précision (Accuracy) |
| :------------------------- | :------------------- |
| Régression Logistique      | 65.00%               |
| SVM Linéaire               | 64.00%               |
| **SVM avec Noyau RBF**     | **91.50%**           |

*   **Observation :** Les modèles linéaires peinent face à la complexité.
*   **Le Vainqueur :** Le SVM RBF se distingue nettement.

## Diapositive 7: La Magie du Noyau RBF (Visualisation 1)

### Transformer l'Impossible en Évidence

*   **Le Problème :** Des données non-linéairement séparables dans 2D.
*   **La Solution du Noyau :** Projeter les données dans une dimension supérieure.
*   **Le Résultat :** Une séparation linéaire devient possible !

*(Insérer ici une animation ou une série d'images montrant la projection des données 2D en 3D pour les rendre linéairement séparables, puis la surface de décision du SVM RBF)*

## Diapositive 8: Surfaces de Décision : L'Œil de l'IA

### Comment Chaque Modèle Voit le Monde

*   **Régression Logistique :** Une ligne droite impuissante.
    ![Surface de Décision : Régression Logistique](/home/ubuntu/Eco_LogReg_surface.png)

*   **SVM Linéaire :** Une autre ligne droite, tout aussi limitée.
    ![Surface de Décision : SVM Linéaire](/home/ubuntu/Eco_SVM_Linear_surface.png)

*   **SVM avec Noyau RBF :** Une frontière flexible qui épouse la réalité.
    ![Surface de Décision : SVM avec Noyau RBF](/home/ubuntu/Eco_SVM_RBF_surface.png)

## Diapositive 9: Matrices de Confusion : Comprendre les Erreurs

### Où les Modèles se Trompent (ou Excellent)

*   **Régression Logistique :** Beaucoup de confusions.
    ![Matrice de Confusion : Régression Logistique](/home/ubuntu/Eco_LogReg_cm.png)

*   **SVM Linéaire :** Des erreurs similaires.
    ![Matrice de Confusion : SVM Linéaire](/home/ubuntu/Eco_SVM_Linear_cm.png)

*   **SVM avec Noyau RBF :** Presque parfait ! Peu de faux positifs/négatifs.
    ![Matrice de Confusion : SVM avec Noyau RBF](/home/ubuntu/Eco_SVM_RBF_cm.png)

## Diapositive 10: Conclusion : Le Triomphe de la Sentinelle

### L'IA, Gardienne de Nos Eaux

*   **Leçon Apprise :** Le choix du modèle est crucial pour des problèmes complexes.
*   **Le SVM RBF :** Un outil puissant pour la détection non-linéaire.
*   **Impact :** Des systèmes comme Sentinelle Écologique peuvent révolutionner la surveillance environnementale.
*   **Perspectives :** Intégration temps réel, autres polluants, déploiement à grande échelle.

## Diapositive 11: Questions / Discussion

### Votre Avis Compte !

*   **Merci de votre attention.**
*   **Des questions ?**

