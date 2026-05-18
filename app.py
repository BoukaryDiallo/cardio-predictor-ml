# Application Streamlit - Projet IA IFOAD
# Auteurs : DIALLO Boukary, TIENDREBEOGO Ouahabou

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# configuration de la page
st.set_page_config(page_title="Maladies cardiaques", page_icon="❤️", layout="wide")


@st.cache_data
def charger_donnees():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    df = X.copy()
    df['target'] = (y['num'] > 0).astype(int)

    # on remplit les valeurs manquantes comme dans le notebook
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    return df


@st.cache_resource
def charger_modeles():
    scaler = joblib.load('models/scaler.pkl')
    resultats = joblib.load('models/results.pkl')
    meta = joblib.load('models/metadata.pkl')

    modeles = {}
    for nom in ['LogisticRegression', 'KNN', 'SVM', 'DecisionTree', 'RandomForest', 'AdaBoost']:
        modeles[nom] = joblib.load(f'models/{nom}.pkl')

    return scaler, modeles, resultats, meta


df = charger_donnees()
scaler, modeles, resultats, meta = charger_modeles()


st.sidebar.title("❤️ Maladies cardiaques")
st.sidebar.write("Projet IA - IFOAD")
st.sidebar.write("Dr Arthur Sawadogo")
st.sidebar.write("---")

page = st.sidebar.radio(
    "Menu",
    ["Accueil", "Analyse exploratoire", "Comparaison des modeles", "Prediction"]
)

st.sidebar.write("---")
st.sidebar.write(f"**Meilleur modele :** {meta['best_name']}")
st.sidebar.write(f"**Nb patients :** {df.shape[0]}")


if page == "Accueil":
    st.title("Prediction des maladies cardiaques")
    st.write("""
    Cette application a ete realisee dans le cadre du projet d'Intelligence
    Artificielle a l'IFOAD. Elle utilise le jeu de donnees **Heart Disease UCI**
    pour predire si un patient est atteint d'une maladie cardiaque a partir de
    ses caracteristiques cliniques.
    """)

    # quelques chiffres cles
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", df.shape[0])
    col2.metric("Variables", df.shape[1] - 1)
    col3.metric("Malades", int(df['target'].sum()))
    col4.metric("Sains", int((df['target'] == 0).sum()))

    st.write("---")
    st.subheader("Apercu des donnees")
    st.dataframe(df.head(10))

    with st.expander("Description des variables"):
        st.markdown("""
        - **age** : age du patient
        - **sex** : 1 = homme, 0 = femme
        - **cp** : type de douleur thoracique (1 a 4)
        - **trestbps** : pression arterielle au repos (mm Hg)
        - **chol** : cholesterol serique (mg/dl)
        - **fbs** : glycemie a jeun > 120 mg/dl (1=oui, 0=non)
        - **restecg** : ECG au repos (0 a 2)
        - **thalach** : frequence cardiaque max atteinte
        - **exang** : angine d'effort (1=oui, 0=non)
        - **oldpeak** : depression du segment ST a l'effort
        - **slope** : pente du segment ST (1 a 3)
        - **ca** : nombre de gros vaisseaux colores (0 a 3)
        - **thal** : thalassemie (3=normal, 6=defaut fixe, 7=defaut reversible)
        - **target** : 1 = malade, 0 = sain
        """)


elif page == "Analyse exploratoire":
    st.title("Analyse exploratoire")

    onglets = st.tabs([
        "Age", "Sexe", "Douleur (cp)", "Pression/Chol/FC", "Glycemie", "Angine", "Correlation"
    ])

    # Q1 : age
    with onglets[0]:
        st.subheader("Distribution de l'age")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].hist(df['age'], bins=20, color='steelblue', edgecolor='black')
        ax[0].axvline(df['age'].mean(), color='red', linestyle='--',
                      label=f"moyenne = {df['age'].mean():.1f}")
        ax[0].set_xlabel('Age')
        ax[0].set_ylabel('Nb patients')
        ax[0].set_title("Distribution de l'age")
        ax[0].legend()

        sns.boxplot(x='target', y='age', data=df, ax=ax[1])
        ax[1].set_xticklabels(['Sain', 'Malade'])
        ax[1].set_title('Age selon le statut')

        plt.tight_layout()
        st.pyplot(fig)

        st.write("Age moyen sain :", round(df[df['target']==0]['age'].mean(), 2))
        st.write("Age moyen malade :", round(df[df['target']==1]['age'].mean(), 2))

    # Q2 : sexe
    with onglets[1]:
        st.subheader("Difference entre les sexes")

        tableau = pd.crosstab(df['sex'], df['target'])
        tableau.index = ['Femme', 'Homme']
        tableau.columns = ['Sain', 'Malade']
        st.dataframe(tableau)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        tableau.plot(kind='bar', ax=ax[0], color=['green', 'red'])
        ax[0].set_title('Effectifs par sexe et statut')
        ax[0].set_xticklabels(['Femme', 'Homme'], rotation=0)

        prop = df.groupby('sex')['target'].mean()
        prop.index = ['Femme', 'Homme']
        prop.plot(kind='bar', ax=ax[1], color=['orchid', 'royalblue'])
        ax[1].set_title('Proportion de malades par sexe')
        ax[1].set_ylim(0, 1)
        for i, v in enumerate(prop.values):
            ax[1].text(i, v + 0.02, f"{v:.1%}", ha='center')
        ax[1].set_xticklabels(['Femme', 'Homme'], rotation=0)

        plt.tight_layout()
        st.pyplot(fig)

    # Q3 : cp
    with onglets[2]:
        st.subheader("Type de douleur thoracique (cp)")

        labels = ['Angine typique', 'Angine atypique', 'Non angineuse', 'Asymptomatique']
        tableau = pd.crosstab(df['cp'], df['target'])
        tableau.index = labels
        tableau.columns = ['Sain', 'Malade']
        st.dataframe(tableau)

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        tableau.plot(kind='bar', ax=ax[0], color=['green', 'red'])
        ax[0].set_title('Effectifs par type de douleur')
        ax[0].set_xticklabels(labels, rotation=20)

        prop_cp = df.groupby('cp')['target'].mean()
        prop_cp.index = labels
        prop_cp.plot(kind='bar', ax=ax[1], color='steelblue')
        ax[1].set_title('Proportion de malades selon cp')
        ax[1].set_ylim(0, 1)
        for i, v in enumerate(prop_cp.values):
            ax[1].text(i, v + 0.02, f"{v:.1%}", ha='center')
        ax[1].set_xticklabels(labels, rotation=20)

        plt.tight_layout()
        st.pyplot(fig)

    # Q4 : trestbps, chol, thalach
    with onglets[3]:
        st.subheader("Pression, cholesterol et frequence cardiaque")

        moyennes = df.groupby('target')[['trestbps', 'chol', 'thalach']].mean().round(2)
        moyennes.index = ['Sain', 'Malade']
        st.dataframe(moyennes)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        variables = ['trestbps', 'chol', 'thalach']
        titres = ['Pression au repos', 'Cholesterol', 'FC max']
        for i in range(3):
            sns.boxplot(x='target', y=variables[i], data=df, ax=ax[i])
            ax[i].set_xticklabels(['Sain', 'Malade'])
            ax[i].set_title(titres[i])

        plt.tight_layout()
        st.pyplot(fig)

    # Q5 : fbs
    with onglets[4]:
        st.subheader("Glycemie a jeun (fbs)")

        tableau = pd.crosstab(df['fbs'], df['target'], normalize='index') * 100
        tableau.index = ['fbs <= 120', 'fbs > 120']
        tableau.columns = ['% Sain', '% Malade']
        st.dataframe(tableau.round(2))

        fig, ax = plt.subplots(figsize=(7, 4))
        prop_fbs = df.groupby('fbs')['target'].mean()
        prop_fbs.index = ['fbs <= 120', 'fbs > 120']
        prop_fbs.plot(kind='bar', ax=ax, color='tomato')
        ax.set_title('Proportion de malades selon fbs')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(prop_fbs.index, rotation=0)
        for i, v in enumerate(prop_fbs.values):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
        st.pyplot(fig)

        st.info("La glycemie ne semble pas etre un indicateur tres discriminant.")

    # Q6 : exang
    with onglets[5]:
        st.subheader("Angine d'effort (exang)")

        tableau = pd.crosstab(df['exang'], df['target'], normalize='index') * 100
        tableau.index = ['Pas d angine', 'Angine']
        tableau.columns = ['% Sain', '% Malade']
        st.dataframe(tableau.round(2))

        fig, ax = plt.subplots(figsize=(7, 4))
        prop_ex = df.groupby('exang')['target'].mean()
        prop_ex.index = ['Pas d angine', 'Angine']
        prop_ex.plot(kind='bar', ax=ax, color='darkred')
        ax.set_title('Proportion de malades selon exang')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(prop_ex.index, rotation=0)
        for i, v in enumerate(prop_ex.values):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
        st.pyplot(fig)

        st.success("Une angine d'effort est fortement liee a la maladie (~76% de malades).")

    # correlation
    with onglets[6]:
        st.subheader("Matrice de correlation")

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0, ax=ax)
        st.pyplot(fig)


elif page == "Comparaison des modeles":
    st.title("Comparaison des 6 algorithmes")

    st.subheader("Tableau des metriques")
    st.dataframe(resultats.sort_values('F1-Score', ascending=False).style.format("{:.4f}"))

    st.subheader("Graphique comparatif")
    fig, ax = plt.subplots(figsize=(12, 5))
    resultats.plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_title('Comparaison des metriques par modele')
    ax.set_xticklabels(resultats.index, rotation=20)
    ax.legend(loc='lower right', ncol=5)
    plt.tight_layout()
    st.pyplot(fig)

    # on reconstruit le test set pour faire ROC + matrices de confusion
    X = df.drop('target', axis=1)
    y = df['target']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)

    st.subheader("Courbes ROC")
    fig, ax = plt.subplots(figsize=(8, 6))
    for nom, modele in modeles.items():
        y_proba = modele.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{nom} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbes ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    st.subheader("Matrices de confusion")
    colonnes = st.columns(3)
    i = 0
    for nom, modele in modeles.items():
        cm = confusion_matrix(y_test, modele.predict(X_test_sc))
        with colonnes[i % 3]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Sain', 'Malade'],
                        yticklabels=['Sain', 'Malade'], ax=ax)
            ax.set_title(nom)
            ax.set_xlabel('Predit')
            ax.set_ylabel('Reel')
            st.pyplot(fig)
        i += 1

    st.success(f"Le meilleur modele d'apres le F1-Score est : **{meta['best_name']}**")


elif page == "Prediction":
    st.title("Prediction sur un patient")
    st.write("Renseignez les caracteristiques du patient pour obtenir une prediction.")

    # choix du modele
    nom_modele = st.selectbox(
        "Choix du modele",
        list(modeles.keys()),
        index=list(modeles.keys()).index(meta['best_name'])
    )
    modele = modeles[nom_modele]

    with st.form("formulaire"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 20, 90, 55)
            sex = st.radio("Sexe", [0, 1],
                           format_func=lambda v: "Femme" if v == 0 else "Homme",
                           horizontal=True)
            cp = st.selectbox(
                "Type de douleur (cp)",
                [1, 2, 3, 4],
                format_func=lambda v: {
                    1: "1 - Angine typique",
                    2: "2 - Angine atypique",
                    3: "3 - Non angineuse",
                    4: "4 - Asymptomatique"
                }[v]
            )
            trestbps = st.slider("Pression au repos (trestbps)", 90, 200, 130)
            chol = st.slider("Cholesterol (chol)", 100, 600, 245)

        with col2:
            fbs = st.radio("Glycemie > 120 (fbs)", [0, 1],
                           format_func=lambda v: "Non" if v == 0 else "Oui",
                           horizontal=True)
            restecg = st.selectbox(
                "ECG au repos (restecg)", [0, 1, 2],
                format_func=lambda v: {
                    0: "0 - Normal",
                    1: "1 - Anomalie ST-T",
                    2: "2 - Hypertrophie"
                }[v]
            )
            thalach = st.slider("FC max (thalach)", 60, 220, 150)
            exang = st.radio("Angine d'effort (exang)", [0, 1],
                             format_func=lambda v: "Non" if v == 0 else "Oui",
                             horizontal=True)
            oldpeak = st.slider("Depression ST (oldpeak)", 0.0, 7.0, 1.0, 0.1)

        with col3:
            slope = st.selectbox(
                "Pente ST (slope)", [1, 2, 3],
                format_func=lambda v: {
                    1: "1 - Ascendante",
                    2: "2 - Plate",
                    3: "3 - Descendante"
                }[v]
            )
            ca = st.selectbox("Nb gros vaisseaux (ca)", [0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemie (thal)", [3, 6, 7],
                format_func=lambda v: {
                    3: "3 - Normal",
                    6: "6 - Defaut fixe",
                    7: "7 - Defaut reversible"
                }[v]
            )

        valider = st.form_submit_button("Predire", use_container_width=True)

    if valider:
        # on construit le vecteur du patient
        donnees_patient = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])

        # on applique le meme scaler que pour l'entrainement
        donnees_sc = scaler.transform(donnees_patient)

        prediction = int(modele.predict(donnees_sc)[0])
        proba = float(modele.predict_proba(donnees_sc)[0, 1])

        st.write("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            if prediction == 1:
                st.error(f"### Risque de maladie\nProbabilite : **{proba:.1%}**")
            else:
                st.success(f"### Pas de maladie detectee\nProbabilite : **{proba:.1%}**")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 1.5))
            couleur = 'crimson' if prediction == 1 else 'seagreen'
            ax.barh([0], [proba], color=couleur, height=0.5)
            ax.barh([0], [1 - proba], left=[proba], color='lightgrey', height=0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(np.linspace(0, 1, 11))
            ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 11)])
            ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f"Probabilite estimee par {nom_modele}")
            st.pyplot(fig)

        st.write("### Recapitulatif des valeurs saisies")
        st.dataframe(donnees_patient.T.rename(columns={0: 'Valeur'}))

        st.caption("Outil pedagogique - ne remplace pas un avis medical.")
