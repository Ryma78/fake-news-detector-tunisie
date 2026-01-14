import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News Tunisie",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look plus moderne
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fake-result {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .real-result {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    **Détecteur de Fake News pour la Tunisie**
    
    Ce modèle utilise l'intelligence artificielle pour analyser les textes et déterminer s'ils contiennent des informations fiables ou des fake news.
    
    - **Modèle**: Régression Logistique + TF-IDF
    - **Précision**: F1-Score de 100% sur les données d'entraînement
    - **Langue**: Français (Tunisie)
    """)
    
    st.header("📊 Statistiques")
    st.markdown("""
    - **Données d'entraînement**: 21,353 textes
    - **Fake News**: 5,227 (24.5%)
    - **News Réelles**: 16,126 (75.5%)
    """)
    
    st.header("💡 Conseils d'utilisation")
    st.markdown("""
    - Entrez un texte en français
    - Le modèle analyse le style et le contenu
    - Plus le texte est long, plus l'analyse est précise
    """)

# Contenu principal
st.markdown('<h1 class="main-header">📰 Détecteur de Fake News Tunisie</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analysez la fiabilité des informations en temps réel</p>', unsafe_allow_html=True)

# Zone de saisie
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "📝 Texte à analyser :",
        "Le président tunisien Kais Saied annonce de nouvelles réformes économiques pour stimuler l'investissement étranger.",
        height=150,
        help="Collez ou tapez le texte que vous souhaitez analyser"
    )

with col2:
    st.markdown("### 📋 Exemples rapides")
    if st.button("📰 News réelle"):
        st.session_state.text = "Le gouvernement tunisien annonce une nouvelle loi sur l'éducation qui entrera en vigueur le mois prochain."
    if st.button("🚨 Fake News"):
        st.button("URGENT: Confinement total à Tunis suite à une nouvelle vague de coronavirus!")
    if st.button("🧹 Effacer"):
        st.session_state.text = ""

# Bouton d'analyse
col_center = st.columns([1, 2, 1])[1]
with col_center:
    analyze_button = st.button("🔍 Analyser le texte", use_container_width=True)

# Résultats
if analyze_button and text.strip():
    with st.spinner("Analyse en cours..."):
        try:
            resp = requests.post("http://localhost:8000/predict", json={"text": text}, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                
                # Affichage du résultat
                is_fake = result["is_fake"]
                prob_fake = result["probability_fake"]
                
                if is_fake:
                    st.markdown(f"""
                    <div class="result-box fake-result">
                        <h2>🚨 FAKE NEWS DÉTECTÉE</h2>
                        <p>Ce texte présente des caractéristiques de fake news</p>
                        <div class="metric-value">{prob_fake:.1%}</div>
                        <p>Probabilité de fake news</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box real-result">
                        <h2>✅ INFORMATION FIABLE</h2>
                        <p>Ce texte semble contenir des informations fiables</p>
                        <div class="metric-value">{1-prob_fake:.1%}</div>
                        <p>Probabilité d'authenticité</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Métriques détaillées
                st.subheader("📊 Analyse détaillée")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité Fake", f"{prob_fake:.1%}")
                with col2:
                    st.metric("Probabilité Réel", f"{1-prob_fake:.1%}")
                
                # Texte nettoyé
                with st.expander("🔧 Voir le texte traité"):
                    st.code(result["text"], language="text")
                    
            else:
                st.error(f"❌ Erreur du serveur: {resp.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Impossible de contacter le serveur d'analyse: {e}")
            st.info("Vérifiez que l'API est en cours d'exécution sur http://localhost:8000")

elif analyze_button:
    st.warning("⚠️ Veuillez entrer un texte à analyser")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Développé avec ❤️ pour la lutte contre la désinformation en Tunisie</p>
    <p><small>Modèle entraîné sur des données tunisiennes • Mise à jour: Janvier 2026</small></p>
</div>
""", unsafe_allow_html=True)