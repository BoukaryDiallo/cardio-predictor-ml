# Projet IA — IFOAD : Classification des maladies cardiaques ❤️

> ## 🌐 Application en ligne
> **➡️ https://cardio-predictor-ml.streamlit.app/**
>
> L'application Streamlit est déployée et accessible directement via le lien ci-dessus.

**Cours** : Intelligence Artificielle et Apprentissage Automatique — Dr Arthur Sawadogo
**Dataset** : [Heart Disease UCI (id=45)](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Structure du projet

```
IA/
├── notebook/
│   └── heart_disease.ipynb        # Notebook complet (EDA + 6 modèles + comparaison)
├── models/                         # Artefacts générés (pkl)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── LogisticRegression.pkl
│   ├── KNN.pkl
│   ├── SVM.pkl
│   ├── DecisionTree.pkl
│   ├── RandomForest.pkl
│   ├── AdaBoost.pkl
│   ├── results.pkl
│   └── metadata.pkl
├── app.py                          # Application Streamlit
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate            # Linux/Mac
# venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

## Utilisation

### 1. Exécuter le notebook

```bash
jupyter notebook notebook/heart_disease.ipynb
```

Le notebook réalise :
- chargement via `ucimlrepo`
- binarisation de la cible (`num` 0-4 → 0/1)
- imputation des valeurs manquantes (`ca`, `thal`)
- EDA répondant aux 6 questions du sujet
- entraînement de 6 modèles : LogisticRegression, KNN, SVM, DecisionTree, RandomForest, AdaBoost
- évaluation : Accuracy, Précision, Rappel, F1-Score, AUC-ROC
- sauvegarde des modèles dans `models/`

### 2. Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application propose 4 pages :
- 🏠 **Accueil** : aperçu du dataset
- 📊 **Analyse exploratoire** : graphiques répondant aux 6 questions
- 🤖 **Comparaison des modèles** : tableau, ROC, matrices de confusion
- 🔮 **Prédiction** : formulaire patient → prédiction + probabilité

## Résultats

Classement par F1-Score :

| Modèle              | Accuracy | Précision | Rappel | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| **RandomForest**    | 0.9016   | 0.8438    | 0.9643 | **0.9000** | 0.9551 |
| AdaBoost            | 0.9016   | 0.8438    | 0.9643 | 0.9000   | 0.9740 |
| LogisticRegression  | 0.8689   | 0.8125    | 0.9286 | 0.8667   | 0.9513 |
| SVM                 | 0.8525   | 0.8065    | 0.8929 | 0.8475   | 0.9437 |
| KNN                 | 0.8361   | 0.7647    | 0.9286 | 0.8387   | 0.9453 |
| DecisionTree        | 0.7705   | 0.7059    | 0.8571 | 0.7742   | 0.8030 |

Le modèle déployé dans l'application est **RandomForest** (meilleur F1-Score).

## Auteurs

- DIALLO Boukary
- TIENDREBEOGO Ouahabou.
