from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Only the columns the simulator needs, gzipped to stay small enough for GitHub
DATA_PATH = Path(__file__).parent / "heart_2022_simulateur.csv.gz"

CATEGORICAL = ["Sex", "AgeCategory", "GeneralHealth", "SmokerStatus", "HadDiabetes",
               "PhysicalActivities", "HadStroke", "HadKidneyDisease", "HadCOPD", "DifficultyWalking"]
NUMERIC = ["BMI", "SleepHours"]

# Interface texts; the model keeps the dataset's English values
TEXT = {
    "fr": {
        "title": "🫀 Simulateur de risque cardiaque",
        "disclaimer": "⚠️ Outil pédagogique, pas un avis médical.",
        "trained_on": "Modèle entraîné sur 246 022 adultes américains (enquête BRFSS 2022).",
        "privacy": "🔒 Rien n'est conservé : les informations saisies ne sont ni enregistrées ni transmises. "
                   "Tout est anonyme et disparaît à la fermeture de la page.",
        "profile": "Ton profil",
        "sex": "Sexe", "age": "Tranche d'âge", "health": "État de santé ressenti", "smoker": "Tabac",
        "diabetes": "Diabète", "activity": "Activité physique ce dernier mois",
        "bmi": "IMC (indice de masse corporelle)", "sleep": "Heures de sommeil par nuit",
        "stroke": "Antécédent d'AVC", "kidney": "Maladie rénale", "copd": "BPCO (maladie pulmonaire)",
        "walking": "Difficulté à marcher",
        "result": "Résultat", "risk": "Risque estimé", "average": "Moyenne de la population",
        "same_group": "Moyenne même sexe et âge",
        "higher": "Ce profil a un risque **{x} fois plus élevé** que la moyenne.",
        "lower": "Ce profil a un risque **{x} fois moins élevé** que la moyenne.",
        "what_if": "Et si… ?",
        "what_if_intro": "Effet d'un seul changement d'habitude, toutes choses égales par ailleurs :",
        "quit_smoking": "Arrêter de fumer", "be_active": "Pratiquer une activité physique",
        "bmi_24": "Ramener l'IMC à 24", "sleep_7": "Dormir 7 heures",
        "change": "Changement", "new_risk": "Nouveau risque", "gap": "Écart (points)",
        "all_good": "Ce profil a déjà toutes les bonnes habitudes testées ici.",
        "footer": "Maladie cardiaque = crise cardiaque ou angine de poitrine déclarée. "
                  "Régression logistique — projet de Laurence YANG (Kaggle : laurenceky).",
        "Female": "Femme", "Male": "Homme", "No": "Non", "Yes": "Oui",
        "Excellent": "Excellente", "Very good": "Très bonne", "Good": "Bonne", "Fair": "Moyenne", "Poor": "Mauvaise",
        "Never smoked": "N'a jamais fumé", "Former smoker": "Ancien fumeur",
        "Current smoker - now smokes some days": "Fume de temps en temps",
        "Current smoker - now smokes every day": "Fume tous les jours",
        "No, pre-diabetes or borderline diabetes": "Pré-diabète",
        "Yes, but only during pregnancy (female)": "Seulement pendant la grossesse",
        "age_range": "{a} à {b} ans", "age_last": "80 ans et plus",
    },
    "en": {
        "title": "🫀 Heart Disease Risk Simulator",
        "disclaimer": "⚠️ Educational tool, not medical advice.",
        "trained_on": "Model trained on 246,022 American adults (BRFSS 2022 survey).",
        "privacy": "🔒 Nothing is stored: the information you enter is neither saved nor sent anywhere. "
                   "Everything is anonymous and disappears when you close the page.",
        "profile": "Your profile",
        "sex": "Sex", "age": "Age group", "health": "Self-rated general health", "smoker": "Smoking",
        "diabetes": "Diabetes", "activity": "Physical activity in the past month",
        "bmi": "BMI (body mass index)", "sleep": "Hours of sleep per night",
        "stroke": "History of stroke", "kidney": "Kidney disease", "copd": "COPD (lung disease)",
        "walking": "Difficulty walking",
        "result": "Result", "risk": "Estimated risk", "average": "Population average",
        "same_group": "Average, same sex and age",
        "higher": "This profile's risk is **{x} times higher** than average.",
        "lower": "This profile's risk is **{x} times lower** than average.",
        "what_if": "What if…?",
        "what_if_intro": "Effect of a single change in habits, all else being equal:",
        "quit_smoking": "Quit smoking", "be_active": "Be physically active",
        "bmi_24": "Bring BMI down to 24", "sleep_7": "Sleep 7 hours",
        "change": "Change", "new_risk": "New risk", "gap": "Difference (points)",
        "all_good": "This profile already has all the healthy habits tested here.",
        "footer": "Heart disease = reported heart attack or angina. "
                  "Logistic regression — a project by Laurence YANG (Kaggle: laurenceky).",
        "Female": "Female", "Male": "Male", "No": "No", "Yes": "Yes",
        "Excellent": "Excellent", "Very good": "Very good", "Good": "Good", "Fair": "Fair", "Poor": "Poor",
        "Never smoked": "Never smoked", "Former smoker": "Former smoker",
        "Current smoker - now smokes some days": "Smokes some days",
        "Current smoker - now smokes every day": "Smokes every day",
        "No, pre-diabetes or borderline diabetes": "Pre-diabetes",
        "Yes, but only during pregnancy (female)": "Only during pregnancy",
        "age_range": "{a} to {b}", "age_last": "80 or older",
    },
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["HD"] = (df["HadHeartAttack"].eq("Yes") | df["HadAngina"].eq("Yes")).astype(int)
    return df


@st.cache_resource
def train_model(df):
    # No class_weight here: the simulator must show real probabilities, not a rebalanced score
    model = make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]),
        LogisticRegression(max_iter=1000),
    )
    model.fit(df[CATEGORICAL + NUMERIC], df["HD"])
    return model


st.set_page_config(page_title="Heart Disease Risk Simulator", page_icon="🫀")

# Soothing blue and sea-green waves along the bottom of the page.
# Each wave starts and ends at the same height so it can tile and drift sideways.
def wave_svg(y, amplitude, color, opacity):
    return quote(f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 400' preserveAspectRatio='none'>
<path d='M0 {y} C240 {y - amplitude} 480 {y - amplitude} 720 {y} S1200 {y + amplitude} 1440 {y} V400 H0Z'
fill='{color}' fill-opacity='{opacity}'/></svg>""")


st.markdown(f"""
<style>
[data-testid="stApp"] {{
    background-color: #F5F9FC;
    background-image: url("data:image/svg+xml,{wave_svg(120, 70, '#B5D4F4', 0.55)}"),
                      url("data:image/svg+xml,{wave_svg(200, 60, '#9FE1CB', 0.6)}"),
                      url("data:image/svg+xml,{wave_svg(290, 45, '#5DCAA5', 0.45)}"),
                      linear-gradient(180deg, #F5F9FC 0%, #E6F1FB 100%);
    background-repeat: repeat-x, repeat-x, repeat-x, no-repeat;
    background-position: 0 100%, 0 100%, 0 100%, top;
    background-size: 1440px 45vh, 1440px 45vh, 1440px 45vh, 100% 100%;
    background-attachment: fixed;
    animation: drift 60s linear infinite;
}}
/* The three waves move at different speeds, which gives a gentle depth effect */
@keyframes drift {{
    from {{ background-position: 0 100%, 0 100%, 0 100%, top; }}
    to {{ background-position: 1440px 100%, -1440px 100%, 2880px 100%, top; }}
}}
@keyframes fade-up {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes heartbeat {{
    0%, 40%, 100% {{ transform: scale(1); }}
    10%, 30% {{ transform: scale(1.18); }}
    20% {{ transform: scale(1.05); }}
}}
.heartbeat {{ display: inline-block; animation: heartbeat 1.6s ease-in-out infinite; }}
[data-testid="stMain"] h1, [data-testid="stAlert"], [data-testid="stMetric"], [data-testid="stTable"] {{
    animation: fade-up 0.7s ease-out both;
}}
[data-testid="stAlert"] {{ animation-delay: 0.15s; }}
[data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] {{ animation-delay: 0.3s; }}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {{ animation-delay: 0.45s; }}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{ animation-delay: 0.6s; }}
[data-testid="stTable"] {{ animation-delay: 0.75s; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ animation: fade-up 0.6s ease-out both; }}
[data-testid="stMetric"], [data-testid="stTable"] {{
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
[data-testid="stMetric"]:hover, [data-testid="stTable"]:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(12, 68, 124, 0.12);
}}
[data-testid="stProgress"] div {{ transition: width 0.6s ease; }}
/* Respect visitors who turned off animations in their system settings */
@media (prefers-reduced-motion: reduce) {{
    *, [data-testid="stApp"] {{ animation: none !important; transition: none !important; }}
}}
[data-testid="stHeader"] {{ background-color: rgba(245, 249, 252, 0.8); }}
[data-testid="stSidebar"] {{ background-color: rgba(255, 255, 255, 0.75); }}
[data-testid="stSidebar"] h2 {{
    font-size: 1.9rem;
    color: #0C447C;
    border-bottom: 3px solid #5DCAA5;
    padding-bottom: 6px;
    margin-bottom: 12px;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #1F2A37;
}}
[data-testid="stMetric"], [data-testid="stTable"] {{
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 12px;
    padding: 12px 16px;
}}
</style>
""", unsafe_allow_html=True)

lang = st.sidebar.radio("Langue / Language", ["fr", "en"], horizontal=True,
                        format_func={"fr": "🇫🇷 Français", "en": "🇬🇧 English"}.get)
T = TEXT[lang]


def label(value):
    return T.get(value, value)


def pct(x):
    # French writes 2,5 % where English writes 2.5%
    return f"{x:.1%}" if lang == "en" else f"{x * 100:.1f} %".replace(".", ",")


def ratio(x, signed=False):
    text = f"{x:+.1f}" if signed else f"{x:.1f}"
    return text if lang == "en" else text.replace(".", ",")


def age_label(category):
    if "older" in category:
        return T["age_last"]
    low, high = category.replace("Age ", "").split(" to ")
    return T["age_range"].format(a=low, b=high)


st.markdown(f"<h1><span class='heartbeat'>🫀</span> {T['title'].removeprefix('🫀 ')}</h1>",
            unsafe_allow_html=True)
st.warning(f"### {T['disclaimer']}\n{T['trained_on']}")
st.info(T["privacy"])

df = load_data()
model = train_model(df)
average_rate = df["HD"].mean()

sb = st.sidebar
sb.header(T["profile"])
profile = {
    "Sex": sb.radio(T["sex"], ["Female", "Male"], format_func=label, horizontal=True),
    "AgeCategory": sb.selectbox(T["age"], sorted(df["AgeCategory"].unique()), index=6, format_func=age_label),
    "GeneralHealth": sb.select_slider(T["health"], ["Poor", "Fair", "Good", "Very good", "Excellent"],
                                      value="Good", format_func=label),
    "SmokerStatus": sb.selectbox(T["smoker"], ["Never smoked", "Former smoker",
                                               "Current smoker - now smokes some days",
                                               "Current smoker - now smokes every day"], format_func=label),
    "HadDiabetes": sb.selectbox(T["diabetes"], ["No", "No, pre-diabetes or borderline diabetes", "Yes",
                                                "Yes, but only during pregnancy (female)"], format_func=label),
    "PhysicalActivities": sb.radio(T["activity"], ["Yes", "No"], format_func=label, horizontal=True),
    "BMI": sb.slider(T["bmi"], 15.0, 50.0, 26.0, 0.5),
    "SleepHours": sb.slider(T["sleep"], 3, 12, 7),
    "HadStroke": sb.radio(T["stroke"], ["No", "Yes"], format_func=label, horizontal=True),
    "HadKidneyDisease": sb.radio(T["kidney"], ["No", "Yes"], format_func=label, horizontal=True),
    "HadCOPD": sb.radio(T["copd"], ["No", "Yes"], format_func=label, horizontal=True),
    "DifficultyWalking": sb.radio(T["walking"], ["No", "Yes"], format_func=label, horizontal=True),
}


def predict(p):
    return model.predict_proba(pd.DataFrame([p]))[0, 1]


risk = predict(profile)
same_group = df[(df["Sex"] == profile["Sex"]) & (df["AgeCategory"] == profile["AgeCategory"])]

st.subheader(T["result"])
col1, col2, col3 = st.columns(3)
col1.metric(T["risk"], pct(risk))
col2.metric(T["average"], pct(average_rate))
col3.metric(T["same_group"], pct(same_group['HD'].mean()))
st.progress(min(risk, 1.0))
if risk >= average_rate:
    st.write(T["higher"].format(x=ratio(risk / average_rate)))
else:
    st.write(T["lower"].format(x=ratio(average_rate / risk)))

st.subheader(T["what_if"])
st.write(T["what_if_intro"])
scenarios = {
    "quit_smoking": {"SmokerStatus": "Former smoker"},
    "be_active": {"PhysicalActivities": "Yes"},
    "bmi_24": {"BMI": 24.0},
    "sleep_7": {"SleepHours": 7},
}
rows = []
for key, change in scenarios.items():
    if all(profile[k] == v for k, v in change.items()):
        continue
    if key == "quit_smoking" and not profile["SmokerStatus"].startswith("Current"):
        continue
    new_risk = predict({**profile, **change})
    rows.append({T["change"]: T[key], T["new_risk"]: pct(new_risk),
                 T["gap"]: ratio(round((new_risk - risk) * 100, 1) + 0.0, signed=True)})
if rows:
    st.table(pd.DataFrame(rows).set_index(T["change"]))
else:
    st.success(T["all_good"])

st.caption(T["footer"])
