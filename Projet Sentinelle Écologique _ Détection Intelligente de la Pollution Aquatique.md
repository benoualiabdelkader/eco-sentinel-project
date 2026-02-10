# Projet Sentinelle Écologique : Détection Intelligente de la Pollution Aquatique

## 1. Introduction
La pollution aquatique représente une menace majeure pour les écosystèmes et la santé humaine. La détection précoce et précise des états de pollution est cruciale pour la mise en œuvre de mesures correctives efficaces. Ce projet vise à développer un système intelligent, baptisé "Sentinelle Écologique", capable de classifier l'état de l'eau (Sain ou Pollué) à partir de données de capteurs environnementaux. Nous explorerons et comparerons les performances de différents modèles de classification, notamment la Régression Logistique et les Machines à Vecteurs de Support (SVM) avec et sans noyau, sur un jeu de données simulant des mesures IoT complexes.

## 2. Méthodologie

### 2.1. Génération du Jeu de Données "Sentinelle Écologique"
Pour ce projet, un jeu de données synthétique a été créé pour simuler des mesures provenant de capteurs IoT déployés dans des cours d'eau. Ce dataset, nommé `eco_sentinel_dataset.csv`, comprend les caractéristiques suivantes :

*   **Turbidité (NTU)** : Mesure de la clarté de l'eau, indiquant la présence de matières en suspension.
*   **Oxygène Dissous (mg/L)** : Quantité d'oxygène disponible dans l'eau, essentielle à la vie aquatique.
*   **État de l'Eau** : Variable cible binaire (0 pour 'Sain', 1 pour 'Pollué').

Le jeu de données a été conçu pour présenter une séparation non-linéaire complexe, où les points représentant l'eau saine sont regroupés au centre, tandis que les points d'eau polluée sont dispersés en périphérie. Cette structure permet de tester la robustesse des modèles face à des problèmes de classification plus réalistes que des données linéairement séparables.

### 2.2. Modèles de Classification
Trois modèles de classification ont été entraînés et évalués :

*   **Régression Logistique** : Un modèle linéaire simple, souvent utilisé comme référence.
*   **Machine à Vecteurs de Support (SVM) Linéaire** : Un classifieur linéaire qui cherche à trouver l'hyperplan optimal séparant les classes.
*   **Machine à Vecteurs de Support (SVM) avec Noyau RBF (Radial Basis Function)** : Un modèle non-linéaire capable de projeter les données dans un espace de dimension supérieure pour trouver une séparation optimale, même pour des données non-linéairement séparables.

### 2.3. Prétraitement des Données
Avant l'entraînement des modèles, les caractéristiques ont été standardisées (mise à l'échelle pour avoir une moyenne nulle et un écart-type unitaire). Cette étape est cruciale, en particulier pour les SVM, afin d'assurer que toutes les caractéristiques contribuent équitablement à la distance euclidienne et d'éviter que les caractéristiques avec de plus grandes valeurs numériques ne dominent le processus d'apprentissage.

### 2.4. Évaluation des Performances
Les modèles ont été évalués sur un ensemble de test distinct, en utilisant les métriques suivantes :

*   **Précision (Accuracy)** : Proportion des prédictions correctes.
*   **Matrices de Confusion** : Tableau résumant les prédictions correctes et incorrectes pour chaque classe.
*   **Surfaces de Décision** : Visualisation graphique des frontières de décision apprises par chaque modèle.

## 3. Résultats et Discussion

### 3.1. Performances des Modèles
Le tableau ci-dessous résume les précisions obtenues par chaque modèle sur le jeu de données "Sentinelle Écologique" :

| Modèle | Précision (Accuracy) |
| :------------------------- | :------------------- |
| Régression Logistique      | 65.00%               |
| SVM Linéaire               | 64.00%               |
| **SVM avec Noyau RBF**     | **91.50%**           |

Comme anticipé, la Régression Logistique et le SVM Linéaire affichent des performances relativement faibles. Cela est dû à la nature non-linéaire du problème de classification, que ces modèles linéaires peinent à résoudre efficacement. En revanche, le SVM avec noyau RBF démontre une précision nettement supérieure, atteignant 91.50%.

### 3.2. Surfaces de Décision
Les surfaces de décision illustrent visuellement la capacité de chaque modèle à séparer les classes 'Sain' et 'Pollué'.

#### Régression Logistique
![Surface de Décision : Régression Logistique](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19Mb2dSZWdfc3VyZmFjZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlNYjJkU1pXZGZjM1Z5Wm1GalpRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=bQDVV5m9-0tVDY6AstlpTb6izn9SuRg-GQa6XdxE1vhkbPikYRoXYedVeAYkxQuivgdqXNsfvy27IQT9x634L6v2qPBmrVDCui1K5ml89rQmeaiM3daWq2wuhMZ~~DeQXrNfViyUoLXat2jd1ua5fOfkgs80EB43M8hbuRCi9o69sLxoKMDruheYbeIQ4oFFw98Y7o4wsFk5yiPoXzecebu3pcgwXcaBLZkH81BL9m9PLoMXyXfCHVoRcnHrnZgbhXxXCiVDyQ4uvtUnmXB7B8ivmYhiW0TQdVab1KEj~PTZJoN8ZtPsO~~gNs6gvN0mSuu3kbOFznbYCd7BNf20jg__)

La Régression Logistique tente de trouver une frontière de décision linéaire, ce qui est clairement insuffisant pour séparer les données en forme d'anneau de notre dataset. De nombreux points sont mal classifiés.

#### SVM Linéaire
![Surface de Décision : SVM Linéaire](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19TVk1fTGluZWFyX3N1cmZhY2U.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlUVmsxZlRHbHVaV0Z5WDNOMWNtWmhZMlUucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=POrXemtO0-SQeeLmfoxgtAT9hUsMJTIksFhaf~CQCpg5It2jh5eudBho8nvHzJAca1Hax68xCTLEU7v3RFnyIE8va3PiBkLD4mLSoNXBBhZucPD3M~D~berUL031MR2UuHfggHFrkrvA1yHlL2SpvIUkoQfe6iuZjAnM0T~lwmHZIDtBSzEpiDOX3Sp95h8r~kjhURF~8jVEOtGy7lOB7OOkCenT2s6Qdb1CkZdXDbVqHU7po~UOCC58lDjtnQpHc1qTvuteHtFWZ2U09xayIDWCO2IBcPiKJ76DuvvfU20LEfq9JPqQKWAk5uWEBq7-HoNjVgShMXBD1eQipxsUaw__)

Similaire à la Régression Logistique, le SVM Linéaire échoue également à capturer la structure non-linéaire des données, résultant en une frontière de décision linéaire inefficace.

#### SVM avec Noyau RBF
![Surface de Décision : SVM avec Noyau RBF](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19TVk1fUkJGX3N1cmZhY2U.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlUVmsxZlVrSkdYM04xY21aaFkyVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=qIkHgZAPkH17K8B7jOyTTQtcIyVQ7q8eSSgtd50kCCUD4rrDHsFYzkdHciPd~k778Te~gDY~AZ-nBB9iDBlEes3N8XABDFPKi6Sx63dP7hJlSOs3Grj6VY0V7yjd4nnY59OyqTuRrEt8jrNY9i7MmsLysn~bGi~7HVNsbZtia5FyG7uIE16CqzFo-zB8kwifE5ADWo50HVvslI7BnVxWCpthQO7iiW8AexsB6qEwv-vnSeKo48trcDcBXwn5YW-mSkpmaPsF~L77Jwa08dwtXLh2IhF9eRALgUM9v3sLaU3DoLQ0XmAWi2YVC-F~QbDHexZ3nXxqGh2dN1opMLPCkw__)

Le SVM avec noyau RBF, grâce à sa capacité à créer des frontières de décision non-linéaires, parvient à encapsuler la classe 'Sain' au centre et à isoler la classe 'Pollué' en périphérie. Cette visualisation confirme sa supériorité pour ce type de problème.

### 3.3. Matrices de Confusion
Les matrices de confusion fournissent une vue détaillée des erreurs de classification.

#### Matrice de Confusion : Régression Logistique
![Matrice de Confusion : Régression Logistique](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19Mb2dSZWdfY20.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlNYjJkU1pXZGZZMjAucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=JJIS4kgKQSJQ080Mlhd2NyYWVFnA8o-lulJIa9ZUrhEO5oF6jA8wq~Q6kfzwvnBXa4t0vyhpcRWwN~qxCKVK9t3ZY4qB1qKCMM~ZXY2GaLJSAf79-6ncvDEJLVwrrntqmtmP6sJ-6vladnOUOerNsGdEHu5QLBxlEIxZgQ8QTNNEH235aXcCypDe1JiYTLQdZdnmGP23x~-HeNaoQhLEiY6IZoyQKE-ltjG5MgFHgwAMWViyVWt9OWMQQNAcgXUCBCbjyD6iDAmt~4g6KILw2wp~fhBNHBTQAsMo8Qq3CRSOMaiXxba-k-N6LLmTtuC3j75OJ5lt9NTF0rrq80u6UA__)

#### Matrice de Confusion : SVM Linéaire
![Matrice de Confusion : SVM Linéaire](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19TVk1fTGluZWFyX2Nt.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlUVmsxZlRHbHVaV0Z5WDJOdC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=kxIeTswgECL9RwP~ZmQRZBb9Y-xZiNv7BdFnnqOwblYqiCq45MR9OlXTtzNsq7G-SwtgWJBrZTF4W-Hr-6SRyhyVILdWkp9az~tbLNY-fl0l6frA2gAMM4vh3hS6VG97gMbEex2mfKPkr9mUOLejJp0JDwTNNtDsVVjjGYfjyyBwd9iAnLjsTUr-0zxUmDemYGfhGiO2ZBHmRFQU5cMlVf9Al4YJ3MayG43iLWapMVvk43ZILBaSsyv501fvvp-XcYzZczM5V44qun2gSb3FwCQ9Wl2HSKiD1apTOAdF9W1ubKA2wN0IybwcyoJ5OJV8AAePluZs055tV2c9qDCrpg__)

#### Matrice de Confusion : SVM avec Noyau RBF
![Matrice de Confusion : SVM avec Noyau RBF](https://private-us-east-1.manuscdn.com/sessionFile/P1uFaSlqHztwsIcdrcZOhY/sandbox/8TmBwevo6RMv3gMD6YGm6i-images_1770759385285_na1fn_L2hvbWUvdWJ1bnR1L0Vjb19TVk1fUkJGX2Nt.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUDF1RmFTbHFIenR3c0ljZHJjWk9oWS9zYW5kYm94LzhUbUJ3ZXZvNlJNdjNnTUQ2WUdtNmktaW1hZ2VzXzE3NzA3NTkzODUyODVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwVmpiMTlUVmsxZlVrSkdYMk50LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=iGIZb4SGPZ7OYCptCEtyJoVUPZ7Q9wOPHXYGb0idPhbh9npuocDxr0~f-zxeSVdgJX0x9SVTuNT3EKaYmb-2eMPDdxYGGLgX83B4J8E~vJTd-uYNrPfFsnX0YQNzpc2NYdUOjAVQpXvUlwGvI7ktAsHHlulPD3JTuQ0pICFUEtRt2dE7J~v9VgzMQZjLtQ7~6Cb6hMrZDUQLyrVVuNM4SkVixIn8KBUFzWHI2CH0KTni1zHtd3O0fl6jfjKOR5kglUBteYbszr9qVoyyeGXEYyDjE3Nj25SKMhP5hDGcXlX6KJmlfGRU6MXhlTtOuq0trTW-wa2aPPj1-gORWrEC9A__)

Les matrices de confusion confirment les observations précédentes. Pour la Régression Logistique et le SVM Linéaire, on observe un nombre significatif de faux positifs et de faux négatifs. En revanche, le SVM avec noyau RBF présente un nombre très faible d'erreurs, ce qui se traduit par une précision élevée et une classification quasi parfaite des états de l'eau.

## 4. Conclusion
Ce projet a démontré l'importance cruciale du choix du modèle de classification en fonction de la nature des données. Pour le problème de détection de pollution aquatique avec des corrélations non-linéaires, les modèles linéaires tels que la Régression Logistique et le SVM Linéaire se sont avérés inefficaces. Le **SVM avec noyau RBF** a, quant à lui, prouvé son excellence en capturant la complexité sous-jacente des données et en atteignant une précision remarquable. Le projet "Sentinelle Écologique" illustre comment l'intelligence artificielle, et en particulier les algorithmes de machine learning avancés, peuvent jouer un rôle vital dans la protection de l'environnement en fournissant des outils de surveillance et d'alerte précis. Ce projet constitue une base solide pour des développements futurs, incluant l'intégration de données en temps réel et l'exploration d'autres caractéristiques environnementales.
