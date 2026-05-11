import torch
torch.set_num_threads(1)

import streamlit as st
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FishLens · Species Identifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f0a00;
    color: #f5e6c8;
}
.stApp {
    background: radial-gradient(ellipse at top left, #1a0f00 0%, #0f0a00 40%, #0a0800 100%);
}

/* ── Navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 2.5rem;
    background: rgba(20, 12, 0, 0.9);
    border-bottom: 1px solid rgba(255, 160, 0, 0.3);
    backdrop-filter: blur(20px);
    margin-bottom: 2rem;
    border-radius: 0 0 16px 16px;
}
.nav-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(90deg, #ffa500, #ffd700, #ff8c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}
.nav-sub {
    font-size: 0.68rem; color: #7a5c2a;
    letter-spacing: 3px; text-transform: uppercase; margin-top: 2px;
}
.nav-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,165,0,0.1);
    border: 1px solid rgba(255,165,0,0.4);
    border-radius: 100px; padding: 6px 16px;
    font-size: 0.7rem; color: #ffa500; letter-spacing: 1px;
    font-weight: 600;
}
.pulse-dot {
    width: 8px; height: 8px; background: #ffa500;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity:1; box-shadow: 0 0 0 0 rgba(255,165,0,0.5); }
    50%       { opacity:0.7; box-shadow: 0 0 0 6px rgba(255,165,0,0); }
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem; letter-spacing: 3px;
    text-transform: uppercase; color: #ffa500;
    margin-bottom: 0.8rem; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block; width: 24px; height: 2px;
    background: linear-gradient(90deg, #ffa500, transparent);
}

/* ── Upload ── */
.upload-zone {
    background: rgba(255,165,0,0.03);
    border: 1.5px dashed rgba(255,165,0,0.3);
    border-radius: 20px; padding: 2.5rem 2rem;
    text-align: center; margin-bottom: 1rem;
}
.stFileUploader label { color: #7a5c2a !important; font-size: 0.8rem !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff8c00, #ffd700) !important;
    color: #0f0a00 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 1.5px !important; border: none !important;
    border-radius: 12px !important; padding: 0.9rem 2rem !important;
    width: 100% !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(255,140,0,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,140,0,0.6) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(255,140,0,0.08), rgba(255,215,0,0.04));
    border: 1px solid rgba(255,165,0,0.4);
    border-radius: 20px; padding: 2rem; text-align: center;
    margin-bottom: 1rem; animation: glowIn 0.5s ease;
    box-shadow: 0 0 40px rgba(255,140,0,0.1);
}
@keyframes glowIn {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-emoji { font-size: 4rem; margin-bottom: 10px; filter: drop-shadow(0 0 16px rgba(255,165,0,0.6)); }
.result-title {
    font-family: 'Playfair Display', serif;
    font-weight: 800; font-size: 1.7rem;
    background: linear-gradient(90deg, #ffa500, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.result-conf { font-size: 0.82rem; color: #7a5c2a; }

/* ── Confidence bar ── */
.conf-wrap {
    background: rgba(255,255,255,0.05); border-radius: 100px;
    height: 8px; margin: 14px 0; overflow: hidden;
    border: 1px solid rgba(255,165,0,0.15);
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff4500, #ffa500, #ffd700);
    border-radius: 100px;
    box-shadow: 0 0 10px rgba(255,165,0,0.6);
}

/* ── Metrics ── */
.metric-row { display: flex; gap: 10px; margin: 1rem 0; }
.metric-card {
    flex: 1; background: rgba(255,165,0,0.06);
    border: 1px solid rgba(255,165,0,0.2);
    border-radius: 14px; padding: 1rem 0.7rem; text-align: center;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700; color: #ffa500;
    text-shadow: 0 0 12px rgba(255,165,0,0.4);
}
.metric-label { font-size: 0.65rem; color: #7a5c2a; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }

/* ── Info grid ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
.info-card {
    background: rgba(255,165,0,0.05);
    border: 1px solid rgba(255,165,0,0.2);
    border-radius: 14px; padding: 1rem 1.2rem;
}
.info-card-title {
    font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase;
    color: #ffa500; margin-bottom: 8px; font-weight: 600;
}
.info-card-body { font-size: 0.82rem; color: #c8a87a; line-height: 1.7; }
.info-card-body strong { color: #f5e6c8; }

/* ── Edibility badge ── */
.badge-yes { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); color: #4ade80; border-radius: 8px; padding: 6px 14px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-top: 6px; }
.badge-no  { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #f87171; border-radius: 8px; padding: 6px 14px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-top: 6px; }

/* ── Species pills ── */
.species-pill {
    display: inline-block;
    background: rgba(255,165,0,0.08); border: 1px solid rgba(255,165,0,0.25);
    border-radius: 100px; padding: 4px 12px;
    font-size: 0.7rem; color: #c8a87a; margin: 3px; font-weight: 500;
}

/* ── Placeholder ── */
.placeholder {
    min-height: 500px; background: rgba(255,165,0,0.02);
    border: 1.5px dashed rgba(255,165,0,0.15); border-radius: 20px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 2rem;
}

.disclaimer {
    font-size: 0.7rem; color: #4a3010;
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,165,0,0.1);
    border-radius: 8px; padding: 10px 14px; margin-top: 1rem; line-height: 1.6;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Fish database ─────────────────────────────────────────────────────────────
FISH_DB = {
    "Black Sea Sprat": {
        "emoji": "🐟",
        "edible": True,
        "desc": "A small silvery schooling fish from the Black Sea. Rich in Omega-3.",
        "nutrition": "Protein: 17g · Fat: 6g · Omega-3: High · Calories: ~120 kcal/100g",
        "cooking": "Best grilled, fried, or smoked. Also great in salads.",
        "tip": "Very fresh when eyes are clear and flesh is firm. Often eaten whole.",
    },
    "Gilt-Head Bream": {
        "emoji": "🐠",
        "edible": True,
        "desc": "One of the most prized Mediterranean fish with golden eye markings.",
        "nutrition": "Protein: 20g · Fat: 5g · Omega-3: Medium · Calories: ~130 kcal/100g",
        "cooking": "Best baked whole with herbs, grilled, or pan-seared with lemon.",
        "tip": "Look for shiny skin and red gills for freshness. Avoid dull eyes.",
    },
    "Hourse Mackerel": {
        "emoji": "🐟",
        "edible": True,
        "desc": "A fast-swimming pelagic fish, widely used in canned products.",
        "nutrition": "Protein: 19g · Fat: 9g · Omega-3: High · Calories: ~150 kcal/100g",
        "cooking": "Best grilled, smoked, or canned in olive oil. Great with tomatoes.",
        "tip": "Should smell fresh like the sea. Avoid ammonia smell.",
    },
    "Red Mullet": {
        "emoji": "🦐",
        "edible": True,
        "desc": "A bottom-dwelling red fish highly valued in Mediterranean cuisine.",
        "nutrition": "Protein: 18g · Fat: 4g · Omega-3: Medium · Calories: ~115 kcal/100g",
        "cooking": "Pan-fry with butter and capers, or grill with herbs.",
        "tip": "The liver is considered a delicacy in French cuisine. Very perishable.",
    },
    "Red Sea Bream": {
        "emoji": "🐡",
        "edible": True,
        "desc": "Popular in Japanese and Mediterranean cuisine for its delicate flesh.",
        "nutrition": "Protein: 20g · Fat: 3g · Omega-3: Medium · Calories: ~110 kcal/100g",
        "cooking": "Steam, bake, or serve as sashimi/sushi. Pairs well with ginger.",
        "tip": "Fresh when gills are bright red and eyes are clear and bulging.",
    },
    "Sea Bass": {
        "emoji": "🐟",
        "edible": True,
        "desc": "A prized fish with firm white flesh. Widely farmed in Europe.",
        "nutrition": "Protein: 21g · Fat: 3g · Omega-3: Medium · Calories: ~97 kcal/100g",
        "cooking": "Bake en papillote, grill with fennel, or pan-fry crispy skin.",
        "tip": "One of the best choices for beginners. Mild flavor, easy to cook.",
    },
    "Shrimp": {
        "emoji": "🦐",
        "edible": True,
        "desc": "Marine crustaceans found worldwide. Among the most consumed seafood.",
        "nutrition": "Protein: 24g · Fat: 1g · Omega-3: Low · Calories: ~99 kcal/100g",
        "cooking": "Stir-fry, grill, boil, or add to curries and pasta.",
        "tip": "Fresh shrimp should smell briny, not fishy. Avoid black spots on shell.",
    },
    "Striped Red Mullet": {
        "emoji": "🐠",
        "edible": True,
        "desc": "Similar to Red Mullet with horizontal stripes. Common in Mediterranean.",
        "nutrition": "Protein: 18g · Fat: 4g · Omega-3: Medium · Calories: ~118 kcal/100g",
        "cooking": "Best grilled or pan-fried whole. Serve with ratatouille or salsa.",
        "tip": "Highly perishable — cook within 24 hours of purchase for best flavor.",
    },
    "Trout": {
        "emoji": "🐟",
        "edible": True,
        "desc": "A freshwater relative of salmon, beloved in aquaculture worldwide.",
        "nutrition": "Protein: 20g · Fat: 5g · Omega-3: High · Calories: ~119 kcal/100g",
        "cooking": "Pan-fry with almonds, bake with lemon and dill, or smoke.",
        "tip": "Rainbow trout is the most common variety. Flesh should be pink and firm.",
    },
}

CLASS_NAMES = sorted(FISH_DB.keys())

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔥 Loading FishLens model...")
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256),          nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES))
    )
    checkpoint = torch.load("fish_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

try:
    model        = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        conf, pred = probs.max(0)
    return CLASS_NAMES[pred.item()], conf.item() * 100, probs.numpy()

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div>
        <div class="nav-brand">🐟 FishLens</div>
        <div class="nav-sub">Fish Species Identifier · ResNet50 · Transfer Learning</div>
    </div>
    <div class="nav-badge">
        <div class="pulse-dot"></div>
        9 Species · 98.3% Accuracy
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model not found: {model_error}")
    st.info("Place `fish_model.pth` in the same folder as `fish_app.py`")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.1], gap="large")

# ══ LEFT ══════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="section-label">Upload Fish Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)
        predict_btn = st.button("🔍  Identify Fish Species")
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem; margin-bottom:10px;">🐠</div>
            <div style="font-family:'Playfair Display',serif; font-size:1rem;
                        color:#ffa500; margin-bottom:6px;">Drop a fish image here</div>
            <div style="font-size:0.75rem; color:#7a5c2a;">JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Species Database</div>', unsafe_allow_html=True)
    pills = "".join([f'<span class="species-pill">{FISH_DB[c]["emoji"]} {c}</span>' for c in CLASS_NAMES])
    st.markdown(f'<div style="line-height:2.8;">{pills}</div>', unsafe_allow_html=True)

# ══ RIGHT ═════════════════════════════════════════════════════════════════════
with right:
    if predict_btn and uploaded:
        with st.spinner("Identifying species..."):
            pred_class, confidence, probs = predict(image)

        fish = FISH_DB.get(pred_class, {})
        emoji   = fish.get("emoji", "🐟")
        edible  = fish.get("edible", True)
        top2_idx = np.argsort(probs)[::-1]

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-title">{pred_class}</div>
            <div class="result-conf">Confidence: {confidence:.2f}%</div>
            <div class="conf-wrap">
                <div class="conf-fill" style="width:{min(confidence,100):.1f}%;"></div>
            </div>
            {'<span class="badge-yes">✅ Safe to Eat</span>' if edible else '<span class="badge-no">⚠️ Not Recommended</span>'}
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ──
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{probs[top2_idx[1]]*100:.1f}%</div>
                <div class="metric-label">2nd Match</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">98.3%</div>
                <div class="metric-label">Model Acc</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Info grid ──
        st.markdown('<div class="section-label" style="margin-top:1rem;">Fish Intelligence</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-card">
                <div class="info-card-title">🥗 Nutrition (per 100g)</div>
                <div class="info-card-body">{fish.get("nutrition","—")}</div>
            </div>
            <div class="info-card">
                <div class="info-card-title">🍳 Best Cooking Methods</div>
                <div class="info-card-body">{fish.get("cooking","—")}</div>
            </div>
            <div class="info-card" style="grid-column: span 2;">
                <div class="info-card-title">💡 Freshness & Buying Tip</div>
                <div class="info-card-body">{fish.get("tip","—")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability chart ──
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class Probabilities</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0f0a00")
        ax.set_facecolor("#0f0a00")
        colors = ["#ffa500" if c == pred_class else "#2a1a00" for c in CLASS_NAMES]
        labels = [f"{FISH_DB[c]['emoji']}  {c}" for c in CLASS_NAMES]
        bars = ax.barh(labels, probs * 100, color=colors, height=0.6, edgecolor="#2a1a00")
        ax.set_xlabel("Confidence (%)", color="#7a5c2a", fontsize=8)
        ax.set_xlim(0, 115)
        ax.tick_params(colors="#7a5c2a", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2a1a00")
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%", va="center", color="#7a5c2a", fontsize=7.5)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="disclaimer">
        ⚠ FishLens is trained on the Large Scale Fish Dataset (Kaggle) with 98.3% accuracy.
        Nutritional values are approximate. Always consult a professional for dietary advice.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div style="font-size:4rem; margin-bottom:16px;
                        filter:drop-shadow(0 0 20px rgba(255,165,0,0.4));">🐠</div>
            <div style="font-family:'Playfair Display',serif; font-size:1rem;
                        color:#ffa500; margin-bottom:10px;">Awaiting Fish Image</div>
            <div style="font-size:0.8rem; color:#4a3010; max-width:260px; line-height:1.9;">
                Upload a fish image on the left and click
                <strong style="color:#ffa500;">Identify Fish Species</strong>
                to get species info, nutrition, and cooking tips.
            </div>
        </div>
        """, unsafe_allow_html=True)