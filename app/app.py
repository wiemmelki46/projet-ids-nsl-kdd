import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="IDS - Network Intrusion Detection", page_icon="shield", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4270;
        margin: 5px;
    }
    .metric-value { font-size: 2.5em; font-weight: bold; margin: 5px 0; }
    .metric-label { font-size: 0.9em; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .alert-box {
        background: linear-gradient(135deg, #3d1515, #5c2020);
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .header-box {
        background: linear-gradient(135deg, #1a1f3a, #2d3561);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #4a5080;
    }
    .section-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #7eb3ff;
        border-bottom: 2px solid #3d4270;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
    div[data-testid="stTab"] { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 style="color:#7eb3ff; margin:0; font-size:2.2em;">Shield  Systeme de Detection d Intrusions Reseau</h1>
    <p style="color:#aaa; margin:10px 0 0 0; font-size:1.1em;">Option A — Classification Multi-classe | Dataset NSL-KDD | XGBoost</p>
</div>
""", unsafe_allow_html=True)

COLORS = {"normal":"#4C9BE8","DoS":"#E8593C","Probe":"#F2A623","R2L":"#8B5CF6","U2R":"#10B981"}
ICONS  = {"normal":"green","DoS":"red","Probe":"orange","R2L":"purple","U2R":"brown"}

@st.cache_resource
def load_models():
    xgb = joblib.load("models/xgb_model.pkl")
    sc  = joblib.load("models/scaler.pkl")
    le  = joblib.load("models/label_encoder.pkl")
    with open("models/columns.json") as f:
        cols = json.load(f)
    return xgb, sc, le, cols

xgb_model, scaler, le, feature_columns = load_models()

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    threshold  = st.slider("Seuil de confiance minimum", 0.5, 0.99, 0.7, 0.01)
    show_proba = st.checkbox("Afficher probabilites par classe", value=True)
    st.markdown("---")
    st.markdown("### Classes detectees")
    for cls, color in COLORS.items():
        st.markdown(f"<span style=color:{color}>■</span> **{cls}**", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Projet IDS — 4eme annee Cybersecurite")

columns_nsl = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_score"
]

tab1, tab2, tab3 = st.tabs(["  Detection  ", "  Metriques  ", "  A propos  "])

with tab1:
    st.markdown("<div class=section-title>Charger des donnees reseau</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Format NSL-KDD (.txt ou .csv, sans en-tete)", type=["csv","txt"])

    if uploaded:
        df = pd.read_csv(uploaded, header=None, names=columns_nsl, on_bad_lines="skip")
        if "difficulty_score" in df.columns:
            df = df.drop("difficulty_score", axis=1)
        true_labels = df["label"].copy() if "label" in df.columns else None
        X = df.drop(columns=["label"], errors="ignore")

        with st.spinner("Analyse en cours..."):
            X_enc = pd.get_dummies(X, columns=["protocol_type","service","flag"])
            for col in feature_columns:
                if col not in X_enc.columns:
                    X_enc[col] = 0
            X_enc     = X_enc[feature_columns].astype(float)
            X_scaled  = scaler.transform(X_enc)
            preds     = le.inverse_transform(xgb_model.predict(X_scaled))
            probas    = xgb_model.predict_proba(X_scaled)
            confidence= probas.max(axis=1)

        df["Prediction"] = preds
        df["Confiance"]  = confidence.round(3)
        df["Alerte"]     = df["Confiance"] < threshold

        n_total   = len(df)
        n_normal  = int((df["Prediction"]=="normal").sum())
        n_dos     = int((df["Prediction"]=="DoS").sum())
        n_probe   = int((df["Prediction"]=="Probe").sum())
        n_r2l     = int((df["Prediction"]=="R2L").sum())
        n_u2r     = int((df["Prediction"]=="U2R").sum())
        n_alerts  = int(df["Alerte"].sum())
        pct_attack= round((n_total - n_normal) / n_total * 100, 1)

        if pct_attack > 30:
            st.markdown(f"""
            <div class="alert-box">
                <b style="color:#ff6666; font-size:1.1em;">ALERTE — Trafic suspect eleve</b><br>
                <span style="color:#ffaaaa;">{pct_attack}% du trafic analyse est potentiellement malveillant ({n_total - n_normal:,} connexions)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class=section-title>Tableau de bord</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:15px;">
            <div class="metric-card" style="flex:1; border-top:3px solid #7eb3ff;">
                <div class="metric-label">Total Connexions</div>
                <div class="metric-value" style="color:#7eb3ff;">{n_total:,}</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #4C9BE8;">
                <div class="metric-label">Normal</div>
                <div class="metric-value" style="color:#4C9BE8;">{n_normal:,}</div>
                <div style="color:#888; font-size:0.85em;">{round(n_normal/n_total*100,1)}%</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #E8593C;">
                <div class="metric-label">DoS</div>
                <div class="metric-value" style="color:#E8593C;">{n_dos:,}</div>
                <div style="color:#888; font-size:0.85em;">{round(n_dos/n_total*100,1)}%</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #F2A623;">
                <div class="metric-label">Probe</div>
                <div class="metric-value" style="color:#F2A623;">{n_probe:,}</div>
                <div style="color:#888; font-size:0.85em;">{round(n_probe/n_total*100,1)}%</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #8B5CF6;">
                <div class="metric-label">R2L</div>
                <div class="metric-value" style="color:#8B5CF6;">{n_r2l:,}</div>
                <div style="color:#888; font-size:0.85em;">{round(n_r2l/n_total*100,1)}%</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #10B981;">
                <div class="metric-label">U2R</div>
                <div class="metric-value" style="color:#10B981;">{n_u2r:,}</div>
                <div style="color:#888; font-size:0.85em;">{round(n_u2r/n_total*100,1)}%</div>
            </div>
            <div class="metric-card" style="flex:1; border-top:3px solid #ff4444;">
                <div class="metric-label">Alertes</div>
                <div class="metric-value" style="color:#ff4444;">{n_alerts:,}</div>
                <div style="color:#888; font-size:0.85em;">confiance < {threshold}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class=section-title>Visualisations</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7,4), facecolor="#1e2130")
            ax.set_facecolor("#1e2130")
            counts = df["Prediction"].value_counts()
            bars = ax.bar(counts.index, counts.values,
                         color=[COLORS.get(l,"gray") for l in counts.index],
                         width=0.6, edgecolor="none", zorder=3)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                       f"{val:,}", ha="center", va="bottom", color="white", fontsize=9)
            ax.set_title("Distribution des classes predites", color="white", pad=15, fontsize=12)
            ax.tick_params(colors="white")
            ax.spines[:].set_visible(False)
            ax.yaxis.grid(True, color="#2d3250", zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7,4), facecolor="#1e2130")
            ax.set_facecolor("#1e2130")
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                colors=[COLORS.get(l,"gray") for l in counts.index],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops=dict(edgecolor="#1e2130", linewidth=2)
            )
            for t in texts: t.set_color("white")
            for t in autotexts: t.set_color("white"); t.set_fontsize(9)
            ax.set_title("Repartition en pourcentage", color="white", pad=15, fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12,3), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        ax.hist(confidence, bins=60, color="#4C9BE8", edgecolor="none", alpha=0.8)
        ax.axvline(threshold, color="#ff4444", linestyle="--", linewidth=2,
                   label=f"Seuil = {threshold}")
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1000],
                         0, threshold, alpha=0.1, color="#ff4444")
        ax.set_title("Distribution de la confiance des predictions", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color="#2d3250")
        ax.legend(facecolor="#2d3250", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<div class=section-title>Detail des connexions</div>", unsafe_allow_html=True)
        if show_proba:
            proba_df = pd.DataFrame(probas, columns=[f"P({c})" for c in le.classes_]).round(3)
            display_df = pd.concat([df[["Prediction","Confiance","Alerte"]].reset_index(drop=True), proba_df], axis=1)
        else:
            display_df = df[["Prediction","Confiance","Alerte"]]
        st.dataframe(display_df, use_container_width=True, height=300)

        alerts = df[df["Alerte"]]
        if len(alerts) > 0:
            st.markdown(f"""
            <div class="alert-box">
                <b style="color:#ff6666;">  {len(alerts):,} connexions avec confiance inferieure a {threshold}</b>
            </div>
            """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # PERFORMANCE — VERSION AMELIOREE
        # ═══════════════════════════════════════════════
        if true_labels is not None:
            st.markdown("<div class=section-title>Performance du modele</div>", unsafe_allow_html=True)

            dos_l   = ["back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm","mailbomb"]
            probe_l = ["ipsweep","nmap","portsweep","satan","mscan","saint"]
            r2l_l   = ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]

            def map_label(l):
                l = l.split(".")[0].strip()
                if l=="normal": return "normal"
                elif l in dos_l: return "DoS"
                elif l in probe_l: return "Probe"
                elif l in r2l_l: return "R2L"
                else: return "U2R"

            y_true = true_labels.apply(map_label)

            classes_order = ["DoS","Probe","R2L","U2R","normal"]
            prec, rec, f1, sup = precision_recall_fscore_support(
                y_true, preds, labels=classes_order, zero_division=0
            )
            acc = (np.array(preds) == np.array(y_true)).mean()
            f1_weighted = sum(f1[i]*sup[i] for i in range(len(f1))) / sum(sup)

            # ── Banniere metriques globales ──
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2744,#2d3561);border-radius:12px;
                        padding:20px 30px;margin-bottom:20px;border:1px solid #3d4270;
                        display:flex;align-items:center;gap:40px;flex-wrap:wrap;">
                <div>
                    <div style="color:#aaa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">Accuracy Globale</div>
                    <div style="color:#7eb3ff;font-size:3em;font-weight:bold;line-height:1.1;">{acc:.1%}</div>
                </div>
                <div style="border-left:1px solid #3d4270;padding-left:40px;">
                    <div style="color:#aaa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">F1 Weighted</div>
                    <div style="color:#10B981;font-size:3em;font-weight:bold;line-height:1.1;">{f1_weighted:.2f}</div>
                </div>
                <div style="border-left:1px solid #3d4270;padding-left:40px;">
                    <div style="color:#aaa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">Connexions testees</div>
                    <div style="color:#F2A623;font-size:3em;font-weight:bold;line-height:1.1;">{len(y_true):,}</div>
                </div>
                <div style="border-left:1px solid #3d4270;padding-left:40px;">
                    <div style="color:#aaa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">Classes detectees</div>
                    <div style="color:#8B5CF6;font-size:3em;font-weight:bold;line-height:1.1;">5 / 5</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Cards par classe ──
            cards_html = '<div style="display:flex;gap:10px;margin-bottom:24px;flex-wrap:wrap;">'
            cls_colors = {"DoS":"#E8593C","Probe":"#F2A623","R2L":"#8B5CF6","U2R":"#10B981","normal":"#4C9BE8"}

            for i, cls in enumerate(classes_order):
                color   = cls_colors[cls]
                f1_val  = f1[i]
                p_val   = prec[i]
                r_val   = rec[i]
                s_val   = int(sup[i])
                bar_w   = int(f1_val * 100)
                # Couleur badge F1
                if f1_val >= 0.8:
                    badge_color = "#10B981"
                elif f1_val >= 0.5:
                    badge_color = "#F2A623"
                else:
                    badge_color = "#E8593C"

                cards_html += f"""
                <div style="flex:1;min-width:150px;background:#1e2130;border-radius:12px;
                            padding:18px;border:1px solid #2d3250;border-top:3px solid {color};">
                    <div style="color:{color};font-weight:bold;font-size:1.15em;margin-bottom:12px;">{cls}</div>
                    <div style="margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="color:#aaa;font-size:0.75em;text-transform:uppercase;">F1-Score</span>
                            <span style="background:{badge_color};color:white;padding:1px 7px;
                                         border-radius:10px;font-size:0.8em;font-weight:bold;">{f1_val:.2f}</span>
                        </div>
                        <div style="background:#2d3250;border-radius:4px;height:8px;">
                            <div style="background:linear-gradient(90deg,{color},{badge_color});
                                        width:{bar_w}%;height:8px;border-radius:4px;
                                        transition:width 0.3s;"></div>
                        </div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:12px;
                                border-top:1px solid #2d3250;padding-top:10px;">
                        <div style="text-align:center;">
                            <div style="color:#aaa;font-size:0.68em;text-transform:uppercase;">Precision</div>
                            <div style="color:white;font-size:1em;font-weight:bold;">{p_val:.2f}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#aaa;font-size:0.68em;text-transform:uppercase;">Rappel</div>
                            <div style="color:white;font-size:1em;font-weight:bold;">{r_val:.2f}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#aaa;font-size:0.68em;text-transform:uppercase;">Support</div>
                            <div style="color:white;font-size:1em;font-weight:bold;">{s_val:,}</div>
                        </div>
                    </div>
                </div>"""
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Matrice de confusion ──
            st.markdown("<div style='color:#7eb3ff;font-weight:bold;font-size:1.1em;margin-bottom:12px;'>Matrice de Confusion</div>", unsafe_allow_html=True)
            cls_list = list(le.classes_)
            cm  = confusion_matrix(y_true, preds, labels=cls_list)
            fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1e2130")
            ax.set_facecolor("#1e2130")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=cls_list, yticklabels=cls_list, ax=ax,
                       linewidths=0.5, linecolor="#2d3250",
                       annot_kws={"size":12, "weight":"bold"})
            ax.set_title("Matrice de Confusion", color="white", fontsize=14, pad=15)
            ax.tick_params(colors="white", labelsize=11)
            ax.set_xlabel("Predit", color="#aaa", fontsize=12)
            ax.set_ylabel("Reel", color="#aaa", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px; background:#1e2130; border-radius:15px; border:2px dashed #3d4270;">
            <div style="font-size:3em;">upload</div>
            <h3 style="color:#7eb3ff;">Uploadez votre fichier NSL-KDD</h3>
            <p style="color:#888;">Format : 43 colonnes NSL-KDD (avec difficulty_score)<br>Extensions acceptees : .txt, .csv</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class=section-title>Modeles implementes</div>", unsafe_allow_html=True)
    data = {
        "Modele": ["Logistic Regression","Decision Tree","Random Forest","XGBoost (final)"],
        "Type":   ["Baseline","Baseline","Fine-tune","Fine-tune"],
        "Statut": ["Entraine","Entraine","Entraine","Selectionne"]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)
    st.info("Les metriques exactes (F1, Accuracy, AUC) sont affichees dans le notebook Colab apres entraînement.")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Projet
        **Titre :** Detection d Intrusions Reseau par ML
        **Option :** A — Classification Multi-classe
        **Dataset :** NSL-KDD (125 973 enregistrements)
        **Niveau :** 4eme annee Cybersecurite

        ### Pipeline
        1. Chargement NSL-KDD
        2. Regroupement en 5 classes
        3. One-hot encoding
        4. SMOTETomek (reequilibrage)
        5. RandomizedSearchCV
        6. XGBoost final
        """)
    with col2:
        st.markdown("""
        ### Classes detectees
        | Classe | Description |
        |--------|-------------|
        | Normal | Trafic legitime |
        | DoS | Denial of Service |
        | Probe | Reconnaissance |
        | R2L | Remote to Local |
        | U2R | User to Root |

        ### Modele final
        **XGBoost** avec hyperparametres optimises par RandomizedSearchCV sur 3-fold CV
        """)

st.markdown("---")
st.caption("Projet IDS — 4eme annee Cybersecurite | ML-based Network Intrusion Detection System")
