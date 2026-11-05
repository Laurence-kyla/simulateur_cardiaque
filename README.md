# 🫀 Simulateur de risque cardiaque / Heart Disease Risk Simulator

Application Streamlit qui estime le risque de maladie cardiaque (crise cardiaque ou angine de poitrine) à partir d'un profil de santé, et montre l'effet d'un changement d'habitude : arrêter de fumer, pratiquer une activité physique, ramener son IMC à 24, dormir 7 heures.

*A Streamlit app that estimates heart disease risk (heart attack or angina) from a health profile and shows the effect of changing one habit.*

⚠️ **Outil pédagogique, pas un avis médical.** Rien n'est conservé : les informations saisies ne sont ni enregistrées ni transmises.

## Données / Data

- Enquête BRFSS 2022 du CDC, 246 022 adultes américains.
- Jeu de données Kaggle : [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).
- `heart_2022_simulateur.csv.gz` ne garde que les 14 colonnes utilisées par le simulateur.

## Modèle / Model

Régression logistique (scikit-learn), avec encodage one-hot des variables catégorielles et standardisation de l'IMC et du sommeil. Le modèle est volontairement entraîné **sans rééquilibrage des classes**, pour afficher de vraies probabilités et non un score gonflé.

L'analyse complète se trouve dans le notebook Kaggle : [heart-disease-v2](https://www.kaggle.com/code/laurenceky/heart-disease-v2).

## Lancer en local / Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

Projet de **Laurence YANG** · Kaggle : [laurenceky](https://www.kaggle.com/laurenceky)
