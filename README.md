# Détection de fraude à la carte bancaire

🇫🇷 [Français](#-français) · 🇬🇧 [English](#-english)

![Courbes précision-rappel](courbes_pr.png)

---

## 🇫🇷 Français

### Objectif
Comparer la régression logistique (le modèle classique utilisé en banque) avec trois modèles de machine learning : Random Forest, XGBoost et un réseau de neurones.

Questions :
1. Est-ce que les modèles plus complexes détectent mieux la fraude ?
2. Est-ce qu'on peut expliquer leurs décisions ? (SHAP)
3. Combien ça représente en euros ?

### Données
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) : 284 807 transactions, dont 492 fraudes (**0,17 %**). Les variables V1 à V28 sont anonymisées. Le fichier CSV n'est pas dans ce dépôt, il faut le télécharger sur Kaggle.

### Étapes
1. Exploration des données
2. Séparation : train 60 % / validation 20 % / test 20 %
3. Entraînement des 4 modèles (en donnant plus de poids aux fraudes)
4. Choix du seuil de décision sur la validation
5. Résultats sur le test : AUC-ROC, PR-AUC, précision, rappel, F1 (pas d'accuracy, car avec 0,17 % de fraudes elle n'a pas de sens)
6. Explication des décisions avec SHAP
7. Calcul du gain en euros

### Résultats (test, 99 fraudes)

| Modèle | AUC-ROC | PR-AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| Régression logistique | 0,966 | 0,758 | 0,886 | 0,788 | 0,834 |
| Random Forest | 0,977 | 0,875 | 0,931 | 0,818 | 0,871 |
| XGBoost | 0,977 | 0,872 | 0,931 | 0,818 | 0,871 |
| Réseau de neurones | 0,958 | 0,854 | 0,844 | 0,818 | 0,831 |

- XGBoost et Random Forest détectent mieux les fraudes : PR-AUC d'environ 0,87 contre 0,76 pour la régression logistique. XGBoost trouve 81 fraudes sur 99 avec 6 fausses alertes, la régression logistique 78 avec 10 fausses alertes.
- En euros (fraudes évitées moins 5 € par alerte), la différence est très petite, et la régression logistique gagne même un peu plus (environ 6 800 € contre 6 700 €), car elle a détecté quelques fraudes avec de gros montants.
- Conclusion : un meilleur score ne veut pas toujours dire un meilleur gain. Le coût d'une alerte doit être fixé avec les équipes métier.
- SHAP montre que quelques variables (V14, V4, V12, V10) comptent beaucoup plus que les autres.

![SHAP](shap_global.png)

### Limites
Découpage aléatoire (il faudrait plutôt séparer par date), variables anonymisées, seulement 99 fraudes dans le test.

### Outils
Python, pandas, scikit-learn, XGBoost, SHAP, Matplotlib

### Lancer le projet
```bash
pip install -r requirements.txt
# mettre creditcard.csv (Kaggle) dans le dossier, et créer un dossier "figures"
jupyter notebook fraud_detection_benchmark.ipynb
```

---

## 🇬🇧 English

### Goal
Compare logistic regression (the classic model used in banks) with three machine learning models: Random Forest, XGBoost and a neural network, for credit card fraud detection. Then explain the decisions with SHAP and compute the result in euros.

### Data
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle): 284,807 transactions, 492 frauds (**0.17%**). The CSV is not included; download it from Kaggle.

### Main results
- XGBoost and Random Forest detect fraud better: PR-AUC about 0.87 vs 0.76 for logistic regression, with fewer false alerts.
- In euros, the difference is very small, and logistic regression even earns slightly more, because it caught a few large frauds.
- A better score does not always mean a better financial result.
- SHAP shows that a few features (V14, V4, V12, V10) drive most decisions.

**Author:** Sezen Nisa Dikmen
