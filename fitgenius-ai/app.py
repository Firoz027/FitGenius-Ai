# app.py - Full, corrected Personal Trainer Chatbot with robust Gemini (google-genai) integration
# Requirements: streamlit, pandas, numpy, scikit-learn, google-genai (optional)
#
# Run:
#   streamlit run app.py

import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import traceback

# Try to import google genai SDK (Gemini). If not installed, LM features will show a helpful UI message.
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# ----- Config / data directory -----
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------- Helpers ----------
def load_csv(name):
    return pd.read_csv(DATA_DIR / name)

def load_json(name, default):
    p = DATA_DIR / name
    if not p.exists():
        p.write_text(json.dumps(default, indent=2))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # fallback if file corrupted
        return default

def save_json(name, obj):
    (DATA_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# DEBUG helpers (robust JSON parsing from LM outputs)
DEBUG = True

def try_parse_json_from_text(text):
    """
    Try strict json.loads first, then try to extract the first {...} or [...] block.
    Returns (obj, raw_text) where obj is parsed JSON or None.
    """
    if not text or not isinstance(text, str):
        return None, text
    # Try direct
    try:
        return json.loads(text), text
    except Exception:
        pass
    # Try fenced code block with json
    mcode = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if mcode:
        snippet = mcode.group(1)
        try:
            return json.loads(snippet), text
        except Exception:
            pass
    # try to find first {...} or [...]
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text, flags=re.DOTALL)
    if m:
        snippet = m.group(1)
        try:
            return json.loads(snippet), text
        except Exception:
            # cleanup common issues
            san = snippet.replace("\u2013", "-").replace("\u2014", "-")
            san = re.sub(r",\s*([}\]])", r"\1", san)
            try:
                return json.loads(san), text
            except Exception:
                return None, text
    return None, text

def log_debug(msg, obj=None):
    if DEBUG:
        try:
            st.sidebar.write(f"DEBUG: {msg}")
            if obj is not None:
                st.sidebar.write(obj)
        except Exception:
            print("DEBUG:", msg)
            if obj is not None:
                print(obj)

# ---------------- Activity handling (canonical keys + labels + legacy map) ----------------
# Canonical activity multipliers (the *keys* are what we store in JSON)
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

# Pretty labels for UI display
ACTIVITY_LABELS = {
    "sedentary": "Sedentary (little or no exercise)",
    "light": "Lightly active (1–3 days/wk)",
    "moderate": "Moderately active (3–5 days/wk)",
    "active": "Very active (6–7 days/wk)",
    "very_active": "Extra active (hard training / physical job)",
}

# Map legacy/saved labels (including bad dash encodings) back to canonical keys
LEGACY_ACTIVITY_MAP = {
    "Sedentary": "sedentary",
    "Sedentary (little or no exercise)": "sedentary",
    "Lightly active": "light",
    "Lightly active (1-3 days/wk)": "light",
    "Lightly active (1–3 days/wk)": "light",
    "Lightly active (1â€“3 days/wk)": "light",
    "Moderately active": "moderate",
    "Moderately active (3-5 days/wk)": "moderate",
    "Moderately active (3–5 days/wk)": "moderate",
    "Moderately active (3â€“5 days/wk)": "moderate",
    "Very active": "active",
    "Very active (6-7 days/wk)": "active",
    "Very active (6–7 days/wk)": "active",
    "Very active (6â€“7 days/wk)": "active",
    "Extra active": "very_active",
    "Extra active (very hard exercise)": "very_active",
}

def normalize_activity_key(stored_value: str) -> str:
    if not isinstance(stored_value, str):
        return "sedentary"
    val = stored_value.strip()
    if val in ACTIVITY_MULTIPLIERS:
        return val
    return LEGACY_ACTIVITY_MAP.get(val, "sedentary")

# ---------------- Calorie & macros ----------------
# BMR (Mifflin-St Jeor)
def calc_bmr(sex, weight_kg, height_cm, age):
    if isinstance(sex, str) and sex.lower().startswith('m'):
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def calc_tdee(bmr, activity_key: str):
    key = normalize_activity_key(activity_key)
    return float(bmr) * ACTIVITY_MULTIPLIERS.get(key, 1.2)

def macro_split(kcal, split=(0.30, 0.40, 0.30)):
    p, c, f = split
    return {
        'protein_g': round((kcal * p) / 4, 1),
        'carbs_g': round((kcal * c) / 4, 1),
        'fat_g': round((kcal * f) / 9, 1)
    }

# Food filters
def is_food_allowed(row, diet_pref, allergies):
    name = str(row['name']).lower()
    allergies = allergies or []
    allergies = [a.strip().lower() for a in allergies if a.strip()]
    for a in allergies:
        if a in name:
            return False
    dp = (diet_pref or "").lower()
    if dp == 'vegan':
        nonvegan_terms = ['paneer', 'cheese', 'milk', 'yogurt', 'egg', 'honey', 'butter', 'ghee']
        for t in nonvegan_terms:
            if t in name:
                return False
        if 'veg' in row and str(row.get('veg', "")).strip() in ("0", "false", "n", "no"):
            return False
    if dp == 'vegetarian':
        # If veg column exists and is falsy, reject
        if 'veg' in row and str(row.get('veg', "")).strip() in ("0", "false", "n", "no"):
            return False
        # crude filter for meats
        meat_terms = ['chicken', 'beef', 'pork', 'fish', 'mutton', 'shrimp', 'bacon', 'ham', 'turkey']
        for t in meat_terms:
            if t in name:
                return False
    return True

# Greedy meal planner
def simple_meal_plan(foods_df, target_kcal, days=3, diet_pref='None', allergies=None):
    if allergies is None:
        allergies = []
    foods = foods_df.copy()
    allowed = foods[foods.apply(lambda r: is_food_allowed(r, diet_pref, allergies), axis=1)]
    if allowed.empty:
        return {'error': 'No foods match dietary preferences or allergies.'}

    splits = [0.25, 0.40, 0.35]
    tolerance = 0.08
    plans = {}

    # Precompute kcal per gram to scale servings
    allowed = allowed.copy()
    allowed["serving_g"] = allowed.get("serving_g", pd.Series([100]*len(allowed))).replace(0, 100).fillna(100)
    allowed["kcal_per_g"] = allowed["calories"] / allowed["serving_g"]

    for day in range(1, days + 1):
        day_plan = []
        for meal_idx, split in enumerate(splits):
            meal_target = float(target_kcal) * split if target_kcal else 600.0
            meal_items = []
            current_kcal = 0.0
            choices = allowed.sort_values('kcal_per_g', ascending=False).to_dict("records")
            i = 0

            if meal_target <= 0 or np.isnan(meal_target):
                meal_target = max(1.0, (target_kcal or 1800) * split)

            # Greedy select up to 6 items
            while (abs(current_kcal - meal_target) / max(meal_target, 1.0) > tolerance) and len(meal_items) < 6 and choices:
                item = choices[i % len(choices)]
                kcal_per_g = float(item.get("kcal_per_g", 0) or 0)
                if kcal_per_g <= 0:
                    kcal_per_g = 1.0  # fallback
                need_kcal = meal_target - current_kcal
                if need_kcal <= 0:
                    break
                grams = min(max(need_kcal / kcal_per_g, 20.0), 400.0)
                serv_g = float(item.get("serving_g", 100) or 100)
                scale = grams / serv_g
                meal_items.append({
                    "food_id": int(item.get("food_id", -1) if str(item.get("food_id","")).strip() != "" else -1),
                    "name": item.get("name", "Unknown"),
                    "grams": round(grams, 1),
                    "kcal": round(kcal_per_g * grams, 1),
                    "protein_g": round(float(item.get("protein_g", 0) or 0) * scale, 1),
                    "carbs_g": round(float(item.get("carbs_g", 0) or 0) * scale, 1),
                    "fat_g": round(float(item.get("fat_g", 0) or 0) * scale, 1),
                })
                current_kcal += kcal_per_g * grams
                i += 1
                if i > len(choices) * 3:
                    break

            # small booster if still under by > tolerance
            if abs(current_kcal - meal_target) / max(meal_target, 1.0) > tolerance:
                boosters = allowed[allowed['name'].str.contains('olive|oil|nut|almond|peanut', case=False, na=False)]
                if not boosters.empty:
                    booster = boosters.iloc[0].to_dict()
                    grams = 10.0
                    kcal_per_g = float(booster.get("kcal_per_g", 9.0) or 9.0)
                    serv_g = float(booster.get("serving_g", 10) or 10)
                    scale = grams / serv_g
                    meal_items.append({
                        'food_id': int(booster.get('food_id', -1) if str(booster.get('food_id','')).strip() != "" else -1),
                        'name': booster.get('name','Booster'),
                        'grams': grams,
                        'kcal': round(kcal_per_g * grams, 1),
                        'protein_g': round(float(booster.get('protein_g', 0) or 0) * scale, 1),
                        'carbs_g': round(float(booster.get('carbs_g', 0) or 0) * scale, 1),
                        'fat_g': round(float(booster.get('fat_g', 0) or 0) * scale, 1)
                    })
                    current_kcal += kcal_per_g * grams

            day_plan.append({
                "meal_name": ["Breakfast", "Lunch", "Dinner"][meal_idx],
                "target_kcal": round(meal_target, 1),
                "items": meal_items,
                "total_kcal": round(current_kcal, 1)
            })
        plans[f"Day {day}"] = {
            "meals": day_plan,
            "daily_kcal": round(sum(m["total_kcal"] for m in day_plan), 1)
        }

    return plans

# ---------- RAG Lite ----------
class RAGLite:
    def __init__(self, path):
        p = Path(path) if path is not None else None
        if not p or not p.exists():
            self.snippets = []
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf = None
            return
        text = p.read_text(encoding="utf-8")
        self.snippets = [s.strip() for s in text.split("\n\n") if s.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        if self.snippets:
            self.tfidf = self.vectorizer.fit_transform(self.snippets)
        else:
            self.tfidf = None

    def answer(self, query, topk=2, min_score=0.0):
        if not query or not isinstance(query, str) or not query.strip():
            return {"answer": "", "sources": []}
        if self.tfidf is None:
            return {"answer": "", "sources": []}
        try:
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self.tfidf).flatten()
        except Exception as e:
            st.error(f"RAG error: {e}")
            return {"answer": "", "sources": []}
        if sims.size == 0:
            return {"answer": "", "sources": []}
        top_idx = sims.argsort()[-topk:][::-1]
        sources = [{"text": self.snippets[i], "score": float(sims[i])} for i in top_idx]
        filtered = [s for s in sources if s["score"] >= min_score]
        if not filtered:
            filtered = sources
        short = ""
        if filtered:
            short = ". ".join(filtered[0]["text"].split(". ")[:2]).strip()
        return {"answer": short, "sources": filtered}

# Guardrails
RED_FLAGS = ["injury", "chest pain", "dizziness", "faint", "shortness of breath", "ed", "eating disorder"]

def check_guardrails(text):
    t = (text or "").lower()
    for r in RED_FLAGS:
        if r in t:
            return {"refuse": True, "message": "I can't provide medical advice. Please consult a professional."}
    return {"refuse": False}

# ---------- Gemini LM helper (robust) ----------
def ensure_genai_key_from_env_or_ui(ui_key: str | None):
    """
    Priority:
      1) UI-provided (temporary for session)
      2) environment GEMINI_API_KEY
      3) environment GENAI_API_KEY
    """
    if ui_key and ui_key.strip():
        return ui_key.strip()
    return os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")

def lm_query_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Robust Gemini caller adapted to multiple google-genai SDK shapes.
    Tries several call patterns in order and extracts text from responses.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai")

    key = ensure_genai_key_from_env_or_ui(api_key)
    if not key:
        raise RuntimeError("No Gemini API key found. Paste your key in the sidebar or set GEMINI_API_KEY.")

    # Create client explicitly with key if supported
    try:
        client = genai.Client(api_key=key)
    except TypeError:
        client = genai.Client()
        os.environ.setdefault("GENAI_API_KEY", key)

    def _extract_text(resp):
        # Try common spots for generated text
        try:
            if resp is None:
                return ""
            if hasattr(resp, "text") and getattr(resp, "text"):
                return str(resp.text).strip()
            if hasattr(resp, "output_text") and getattr(resp, "output_text"):
                return str(resp.output_text).strip()
            # dict-like shapes
            if isinstance(resp, dict):
                # candidates
                if "candidates" in resp and resp["candidates"]:
                    c = resp["candidates"][0]
                    if isinstance(c, dict):
                        return str(c.get("content") or c.get("text") or "").strip()
                # outputs
                if "outputs" in resp and resp["outputs"]:
                    out0 = resp["outputs"][0]
                    if isinstance(out0, dict):
                        if "content" in out0 and isinstance(out0["content"], str):
                            return out0["content"].strip()
                        content = out0.get("content")
                        if isinstance(content, list) and len(content) > 0:
                            for piece in content:
                                if isinstance(piece, dict) and piece.get("text"):
                                    return piece["text"].strip()
                                if isinstance(piece, str) and piece.strip():
                                    return piece.strip()
                return str(resp).strip()
            # objects with .outputs
            if hasattr(resp, "outputs") and getattr(resp, "outputs"):
                outs = getattr(resp, "outputs")
                try:
                    first = outs[0]
                    if isinstance(first, dict) and "content" in first:
                        cont = first["content"]
                        if isinstance(cont, str):
                            return cont.strip()
                        if isinstance(cont, list) and len(cont) > 0:
                            for piece in cont:
                                if isinstance(piece, dict) and "text" in piece:
                                    return piece["text"].strip()
                                if isinstance(piece, str) and piece.strip():
                                    return piece.strip()
                except Exception:
                    pass
            return str(resp).strip()
        except Exception:
            return ""

    last_exc = None

    # 0) Preferred newer pattern
    try:
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            resp = client.responses.create(model=model, input=prompt, max_output_tokens=int(max_output_tokens), temperature=float(temperature))
            txt = _extract_text(resp)
            if txt:
                return txt
    except Exception as e:
        last_exc = e

    # 1) models.generate_content with temperature
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model=model, contents=prompt, temperature=float(temperature))
            txt = _extract_text(resp)
            if txt:
                return txt
    except Exception as e:
        last_exc = e

    # 2) models.generate_content minimal
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model=model, contents=prompt)
            txt = _extract_text(resp)
            if txt:
                return txt
    except Exception as e:
        last_exc = e

    # 3) chats.create variants
    try:
        if hasattr(client, "chats") and hasattr(client.chats, "create"):
            msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            try:
                resp = client.chats.create(model=model, messages=msgs, temperature=float(temperature), max_output_tokens=int(max_output_tokens))
            except TypeError:
                msgs2 = [{"role": "user", "content": prompt}]
                resp = client.chats.create(model=model, messages=msgs2)
            txt = _extract_text(resp)
            if txt:
                return txt
    except Exception as e:
        last_exc = e

    # 4) older helper
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt, max_tokens=int(max_output_tokens))
            txt = _extract_text(resp)
            if txt:
                return txt
    except Exception as e:
        last_exc = e

    client_dir = ", ".join(sorted([n for n in dir(client) if not n.startswith("_")])[:200])
    sdk_diag = f"Client methods (sample): {client_dir}"
    raise RuntimeError(
        "Gemini call failed. No known client method succeeded.\n"
        f"Last exception: {last_exc}\n\n"
        f"Diagnostic: {sdk_diag}\n\n"
        "Please share diagnostics if you want adaptation to your SDK version."
    )

# ---------- Load Data (defensive) ----------
# load required CSVs; stop with a helpful message if missing
try:
    foods_df = load_csv("foods.csv")
except Exception:
    st.error("Missing or invalid data/foods.csv. Please place foods.csv in the data/ folder.")
    st.stop()

try:
    ex_df = load_csv("exercises.csv")
except Exception:
    st.error("Missing or invalid data/exercises.csv. Please place exercises.csv in the data/ folder.")
    st.stop()

# Load json stores
users = load_json("users.json", {})
diary = load_json("diary.json", {})

# Normalize any legacy activity values on load
for uname, pdata in list(users.items()):
    if isinstance(pdata, dict):
        act = pdata.get("activity")
        if act is not None:
            users[uname]["activity"] = normalize_activity_key(act)

# RAG setup (guidelines.md optional)
guidelines_path = DATA_DIR / "guidelines.md"
rag = RAGLite(guidelines_path if guidelines_path.exists() else None)

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="FitGenius AI",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Enhanced Custom CSS -----
st.markdown("""
<style>
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 700;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ----- Load Data -----
try:
    foods_df = load_csv("foods.csv")
except Exception:
    st.error("⚠️ Missing data/foods.csv. Please add the file to continue.")
    st.stop()

try:
    ex_df = load_csv("exercises.csv")
except Exception:
    st.error("⚠️ Missing data/exercises.csv. Please add the file to continue.")
    st.stop()

users = load_json("users.json", {})
diary = load_json("diary.json", {})

# Normalize legacy activity values
for uname, pdata in list(users.items()):
    if isinstance(pdata, dict):
        act = pdata.get("activity")
        if act is not None:
            users[uname]["activity"] = normalize_activity_key(act)

# RAG setup
guidelines_path = DATA_DIR / "guidelines.md"
rag = RAGLite(guidelines_path if guidelines_path.exists() else None)

# ----- Header -----
st.markdown('<h1 class="main-header">💪 FitGenius AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Intelligent Fitness & Nutrition Companion Powered by AI</p>', unsafe_allow_html=True)

# ----- Enhanced Sidebar -----
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    # User section
    st.markdown("### 👤 User Profile")
    username = st.text_input("Username", value="guest", key="username_input", help="Enter your unique username")
    users.setdefault(username, {})
    
    st.markdown("---")
    
    # Gemini settings
    st.markdown("### 🤖 AI Settings")
    ui_gemini_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        key="gemini_key",
        help="Get your key from Google AI Studio"
    )
    
    with st.expander("🔧 Advanced AI Options", expanded=False):
        gemini_model = st.text_input("Model", value="gemini-2.5-flash", key="gemini_model")
        gemini_temp = st.slider("Temperature", 0.0, 1.0, 0.2, key="gemini_temp", help="Lower = more focused")
        gemini_max_tokens = st.number_input("Max Tokens", 128, 2048, 512, key="gemini_max_tokens")
    
    st.markdown("---")
    
    # Save button
    if st.button("💾 Save All Data", use_container_width=True, help="Save all profiles and diary entries"):
        save_json("users.json", users)
        save_json("diary.json", diary)
        st.success("✅ Data saved successfully!")
    
    st.markdown("---")
    
    # System info
    with st.expander("📊 System Status", expanded=False):
        st.write(f"**Guidelines:** {len(rag.snippets)} snippets")
        rag_status = '✅ Active' if rag.tfidf is not None else '❌ Inactive'
        st.write(f"**RAG System:** {rag_status}")
        gemini_status = '✅ Available' if GENAI_AVAILABLE else '❌ Not installed'
        st.write(f"**Gemini SDK:** {gemini_status}")
        st.write(f"**Active User:** {username}")
        
        # Show user stats
        if username in users and users[username]:
            st.write(f"**Profile:** Complete")
        else:
            st.write(f"**Profile:** Incomplete")
        
        diary_count = len(diary.get(username, {}).get(datetime.date.today().isoformat(), []))
        st.write(f"**Today's Entries:** {diary_count}")

# ----- Main Tabs -----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 My Profile", 
    "🧮 Calorie Calculator", 
    "🍽️ Meal Planner", 
    "🏋️ Workout Builder", 
    "📔 Food Tracker", 
    "💬 AI Coach"
])

# ===== TAB 1: PROFILE =====
with tab1:
    st.markdown("## 📋 Personal Profile")
    st.markdown("*Complete your profile for personalized recommendations*")
    st.markdown("---")
    
    p = users.get(username, {})
    
    # Profile completion indicator
    required_fields = ["age", "sex", "height_cm", "weight_kg", "goal", "activity"]
    completed = sum(1 for f in required_fields if p.get(f))
    completion_pct = (completed / len(required_fields)) * 100
    
    st.progress(completion_pct / 100)
    st.markdown(f"**Profile Completion: {completion_pct:.0f}%**")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Physical Information")
        age = st.number_input("Age (years)", 10, 100, int(p.get("age", 30)), key="age")
        sex = st.selectbox("Sex", ["Male", "Female"], 
                          index=0 if p.get("sex", "Male") == "Male" else 1, key="sex")
        height_cm = st.number_input("Height (cm)", 100, 250, int(p.get("height_cm", 170)), key="height")
        weight_kg = st.number_input("Weight (kg)", 30.0, 300.0, float(p.get("weight_kg", 70.0)), key="weight", step=0.1)
        
        # BMI Calculator
        if height_cm > 0 and weight_kg > 0:
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.info(f"📏 **Your BMI:** {bmi:.1f}")
            if bmi < 18.5:
                st.warning("Underweight")
            elif 18.5 <= bmi < 25:
                st.success("Normal weight")
            elif 25 <= bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
    
    with col2:
        st.markdown("#### 🎯 Goals & Preferences")
        goal = st.selectbox("Fitness Goal", ["Lose", "Maintain", "Gain"], 
                           index=["Lose", "Maintain", "Gain"].index(p.get("goal", "Maintain")), key="goal")
        
        goal_descriptions = {
            "Lose": "🔥 Fat loss & weight reduction",
            "Maintain": "⚖️ Maintain current weight",
            "Gain": "💪 Muscle building & weight gain"
        }
        st.caption(goal_descriptions.get(goal, ""))
        
        stored_activity = normalize_activity_key(p.get("activity", "sedentary"))
        activity_options = list(ACTIVITY_MULTIPLIERS.keys())
        try:
            default_idx = activity_options.index(stored_activity)
        except ValueError:
            default_idx = 0
        
        activity_key = st.selectbox(
            "Activity Level", 
            options=activity_options, 
            index=default_idx,
            format_func=lambda k: ACTIVITY_LABELS.get(k, k), 
            key="activity_select"
        )
        
        diet_pref = st.selectbox("Diet Preference", ["None", "Vegetarian", "Vegan"],
                                index=["None", "Vegetarian", "Vegan"].index(p.get("diet_pref", "None")), 
                                key="diet_pref")
        
        allergies = st.text_input("Allergies (comma-separated)", 
                                 value=p.get("allergies", ""), 
                                 key="allergies",
                                 placeholder="e.g., peanuts, shellfish, dairy")
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("💾 Save Profile", key="save_profile_btn", use_container_width=True):
            users[username] = {
                "age": age, "sex": sex.title(), "height_cm": height_cm,
                "weight_kg": weight_kg, "goal": goal, "activity": activity_key,
                "diet_pref": diet_pref, "allergies": allergies
            }
            save_json("users.json", users)
            st.balloons()
            st.success("✅ Profile saved successfully!")

# ===== TAB 2: CALCULATOR =====
with tab2:
    st.markdown("## 🧮 Smart Calorie & Macro Calculator")
    st.markdown("*Scientifically calculated using Mifflin-St Jeor equation*")
    st.markdown("---")
    
    p = users.get(username)
    if not p:
        st.warning("⚠️ Please complete your profile first in the **My Profile** tab!")
        st.stop()
    
    required = ["sex", "weight_kg", "height_cm", "age", "goal", "activity"]
    missing = [f for f in required if f not in p or p.get(f) in (None, "", [])]
    
    if missing:
        st.error(f"❌ Incomplete profile. Missing: {', '.join(missing)}")
        st.info("👉 Complete your profile in the **My Profile** tab")
        st.stop()
    
    # Calculate values
    bmr = calc_bmr(p["sex"], p["weight_kg"], p["height_cm"], p["age"])
    tdee = calc_tdee(bmr, p["activity"])
    
    if p["goal"] == "Lose":
        target = tdee - 400
        goal_emoji = "🔥"
    elif p["goal"] == "Gain":
        target = tdee + 300
        goal_emoji = "💪"
    else:
        target = tdee
        goal_emoji = "⚖️"
    
    # Display main metrics with cards
    st.markdown("### 📊 Your Daily Energy Needs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔥 BMR</h3>
            <h2>{} kcal</h2>
            <p>Calories at rest</p>
        </div>
        """.format(round(bmr)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ TDEE</h3>
            <h2>{} kcal</h2>
            <p>Total daily expenditure</p>
        </div>
        """.format(round(tdee)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>{} Target</h3>
            <h2>{} kcal</h2>
            <p>For {} goal</p>
        </div>
        """.format(goal_emoji, round(target), p["goal"].lower()), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Macro breakdown
    st.markdown("### 📊 Recommended Macro Split (30/40/30)")
    macros = macro_split(target)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥩 Protein", f"{macros['protein_g']}g", help="30% of calories")
        st.progress(0.30)
    
    with col2:
        st.metric("🍞 Carbohydrates", f"{macros['carbs_g']}g", help="40% of calories")
        st.progress(0.40)
    
    with col3:
        st.metric("🥑 Healthy Fats", f"{macros['fat_g']}g", help="30% of calories")
        st.progress(0.30)
    
    # Warnings and recommendations
    st.markdown("---")
    warn_msgs = []
    
    if (p["sex"].lower().startswith("f") and target < 1200) or (p["sex"].lower().startswith("m") and target < 1500):
        warn_msgs.append("⚠️ Target calories below recommended minimum. Consult a professional.")
    
    if p["goal"] == "Lose":
        protein_floor = round(1.2 * p["weight_kg"], 1)
        if macros['protein_g'] < protein_floor:
            warn_msgs.append(f"💡 For optimal fat loss, aim for at least {protein_floor}g protein daily.")
    
    if warn_msgs:
        for msg in warn_msgs:
            if "⚠️" in msg:
                st.warning(msg)
            else:
                st.info(msg)
    else:
        st.success("✅ Your nutrition targets look great!")

# ===== TAB 3: MEAL PLAN =====
with tab3:
    st.markdown("## 🍽️ AI-Powered Meal Planner")
    st.markdown("*Generate personalized meal plans based on your goals*")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        days = st.slider("📅 Planning Duration (days)", 1, 7, 3, key="plan_days")
    with col2:
        use_gemini_meal = st.toggle("🤖 Use AI", key="use_gemini_meal", 
                                     help="Generate with Gemini AI for better variety")
    
    if st.button("🍽️ Generate Meal Plan", key="generate_plan", use_container_width=True):
        p = users.get(username)
        if not p:
            st.error("❌ Please create your profile first!")
            st.stop()
        
        # Calculate target calories
        target_kcal = calc_tdee(calc_bmr(p["sex"], p["weight_kg"], p["height_cm"], p["age"]), p["activity"])
        if p["goal"] == "Lose":
            target_kcal -= 400
        elif p["goal"] == "Gain":
            target_kcal += 300
        
        allergies_list = [a.strip() for a in p.get("allergies", "").split(",") if a.strip()]
        
        if use_gemini_meal:
            api_key = ensure_genai_key_from_env_or_ui(ui_gemini_key)
            if not GENAI_AVAILABLE or not api_key:
                st.error("❌ Gemini SDK or API key not available! Please configure in sidebar.")
                st.stop()
            
            prompt = f"""Create a {days}-day meal plan as JSON only. Target: {round(target_kcal)}kcal/day.
User: {p['sex']}, {p['age']}yo, {p['weight_kg']}kg, goal={p['goal']}, diet={p['diet_pref']}, allergies={allergies_list or 'none'}.
Format: {{"Day 1": {{"meals": [{{"meal_name": "Breakfast", "target_kcal": 500, "items": [{{"name": "...", "grams": 100, "kcal": 200, "protein_g": 10, "carbs_g": 20, "fat_g": 5}}]}}]}}}}"""
            
            with st.spinner("🤖 AI is creating your personalized meal plan..."):
                try:
                    lm_text = lm_query_gemini(prompt, model=gemini_model, api_key=api_key,
                                             max_output_tokens=int(gemini_max_tokens), 
                                             temperature=float(gemini_temp))
                    parsed, _ = try_parse_json_from_text(lm_text)
                    
                    if parsed:
                        st.success("✅ Meal plan generated successfully!")
                        st.markdown("---")
                        for day, info in parsed.items():
                            daily_kcal = info.get("daily_kcal") or round(sum(m.get("target_kcal",0) for m in info.get("meals",[])),1)
                            st.markdown(f"### 📅 {day} — {daily_kcal} kcal")
                            
                            for meal in info.get("meals", []):
                                with st.expander(f"🍽️ {meal.get('meal_name', 'Meal')} ({meal.get('target_kcal', '?')} kcal)", expanded=False):
                                    items = meal.get("items", [])
                                    if items:
                                        df = pd.DataFrame(items)
                                        st.dataframe(df, use_container_width=True, hide_index=True)
                                        
                                        # Show meal totals
                                        total_p = sum(item.get('protein_g', 0) for item in items)
                                        total_c = sum(item.get('carbs_g', 0) for item in items)
                                        total_f = sum(item.get('fat_g', 0) for item in items)
                                        
                                        col1, col2, col3 = st.columns(3)
                                        col1.caption(f"🥩 Protein: {total_p:.1f}g")
                                        col2.caption(f"🍞 Carbs: {total_c:.1f}g")
                                        col3.caption(f"🥑 Fat: {total_f:.1f}g")
                                    else:
                                        st.warning("No items in this meal")
                            st.markdown("---")
                    else:
                        st.error("❌ Failed to parse AI response. Try again.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            with st.spinner("📋 Creating meal plan from database..."):
                plan = simple_meal_plan(foods_df, target_kcal, days, p.get("diet_pref", "None"), allergies_list)
                
                if isinstance(plan, dict) and plan.get("error"):
                    st.error(plan["error"])
                else:
                    st.success("✅ Meal plan created successfully!")
                    st.markdown("---")
                    for day, info in plan.items():
                        st.markdown(f"### 📅 {day} — {info['daily_kcal']} kcal")
                        
                        for meal in info["meals"]:
                            with st.expander(f"🍽️ {meal['meal_name']} ({meal['total_kcal']} kcal)", expanded=False):
                                df = pd.DataFrame(meal['items'])
                                if not df.empty:
                                    st.dataframe(df[["name", "grams", "kcal", "protein_g", "carbs_g", "fat_g"]], 
                                               use_container_width=True, hide_index=True)
                                else:
                                    st.warning("No items in this meal")
                        st.markdown("---")

# Save on exit
save_json("users.json", users)
save_json("diary.json", diary)
# This is PART 2 - Add this after Part 1 in your app.py file
# The code continues from tab4 onwards...

# ===== TAB 4: WORKOUT =====
with tab4:
    st.markdown("## 🏋️ Personalized Workout Builder")
    st.markdown("*Science-based training programs for your goals*")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        template = st.selectbox("🎯 Select Program Type", [
            "Beginner Full Body (3x/wk)",
            "Upper/Lower Split (4x/wk)",
            "Push/Pull/Legs (6x/wk)"
        ], key="workout_template")
        
        # Program descriptions
        program_info = {
            "Beginner Full Body (3x/wk)": "Perfect for beginners. Work all muscle groups 3x per week.",
            "Upper/Lower Split (4x/wk)": "Intermediate split. Alternate upper and lower body training.",
            "Push/Pull/Legs (6x/wk)": "Advanced routine. High volume and frequency for experienced lifters."
        }
        st.caption(program_info[template])
    
    with col2:
        use_gemini_workout = st.toggle("🤖 AI Mode", key="use_gemini_workout")
    
    if st.button("🏋️ Generate Workout", key="show_workout", use_container_width=True):
        p = users.get(username, {})
        
        if use_gemini_workout:
            api_key = ensure_genai_key_from_env_or_ui(ui_gemini_key)
            if not GENAI_AVAILABLE or not api_key:
                st.error("❌ Gemini SDK or API key not available!")
                st.stop()
            
            days_week = {"Beginner Full Body (3x/wk)": 3, "Upper/Lower Split (4x/wk)": 4, "Push/Pull/Legs (6x/wk)": 6}.get(template, 3)
            pretty_activity = ACTIVITY_LABELS.get(normalize_activity_key(p.get('activity','sedentary')), p.get('activity','sedentary'))
            
            prompt = f"""Create a {template} workout as JSON. User: {p.get('sex','')}, {p.get('age','')}yo, goal={p.get('goal','Maintain')}.
Format: {{"Day 1": {{"exercises": [{{"name": "Squat", "sets": "3", "reps": "8-12", "notes": "Progressive overload"}}]}}}}"""
            
            with st.spinner("🤖 AI is designing your workout program..."):
                try:
                    lm_text = lm_query_gemini(prompt, model=gemini_model, api_key=api_key,
                                             max_output_tokens=int(gemini_max_tokens), 
                                             temperature=float(gemini_temp))
                    parsed, _ = try_parse_json_from_text(lm_text)
                    
                    if parsed:
                        st.success("✅ Workout program generated!")
                        st.markdown("---")
                        for day, info in parsed.items():
                            st.markdown(f"### 💪 {day}")
                            for idx, ex in enumerate(info.get("exercises", []), 1):
                                st.markdown(f"**{idx}. {ex.get('name', '?')}**")
                                st.caption(f"Sets: {ex.get('sets', '?')} × Reps: {ex.get('reps', '?')}")
                                if ex.get('notes'):
                                    st.info(f"💡 {ex.get('notes')}")
                            st.markdown("---")
                        
                        st.info("💡 **Pro Tips:** Warm up 5-10 minutes before training. Stop if you experience sharp pain. Consult a professional for personalized guidance.")
                    else:
                        st.error("❌ Failed to parse AI response")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            def pick(cat, n):
                subset = ex_df[ex_df["category"].astype(str).str.contains(cat, case=False, na=False)]
                return subset.sample(min(len(subset), n)) if len(subset) else subset
            
            st.success("✅ Workout program ready!")
            st.markdown("---")
            
            if template.startswith("Beginner"):
                for d in ["Monday", "Wednesday", "Friday"]:
                    st.markdown(f"### 💪 {d}")
                    picks = pick("Upper|Lower|Full|Core", 5)
                    for idx, (_, r) in enumerate(picks.iterrows(), 1):
                        st.markdown(f"**{idx}. {r['name']}**")
                        st.caption("3 sets × 8-12 reps — Progressive overload")
                    st.markdown("---")
            elif "Upper/Lower" in template:
                for i in range(1, 5):
                    cat = "Upper" if i % 2 == 1 else "Lower"
                    st.markdown(f"### 💪 Day {i} — {cat} Body")
                    picks = pick(cat, 5)
                    for idx, (_, r) in enumerate(picks.iterrows(), 1):
                        st.markdown(f"**{idx}. {r['name']}**")
                        st.caption("3 sets × 6-10 reps — Progressive overload")
                    st.markdown("---")
            else:
                for i, d in enumerate(["Push", "Pull", "Legs"] * 2, 1):
                    st.markdown(f"### 💪 Day {i} — {d}")
                    picks = pick(d, 5)
                    for idx, (_, r) in enumerate(picks.iterrows(), 1):
                        st.markdown(f"**{idx}. {r['name']}**")
                        st.caption("3 sets × 6-12 reps — Progressive overload")
                    st.markdown("---")
            
            st.info("💡 **Pro Tips:** Warm up 5-10 minutes before training. Use RIR ~2 initially. Stop if you experience sharp pain.")

# ===== TAB 5: FOOD DIARY =====
with tab5:
    st.markdown("## 📔 Daily Food Tracker")
    st.markdown("*Log your meals and track your nutrition intake*")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        name = st.text_input("🔍 Food Name or ID", key="diary_name", placeholder="e.g., Chicken Breast or food ID")
    with col2:
        grams = st.number_input("⚖️ Grams", 1.0, 2000.0, 100.0, key="diary_grams", step=10.0)
    with col3:
        use_gemini_food = st.toggle("🤖 AI Lookup", key="use_gemini_food", help="Use AI to estimate nutrition")
    
    if st.button("➕ Add to Diary", key="add_food", use_container_width=True):
        today = datetime.date.today().isoformat()
        
        if use_gemini_food:
            api_key = ensure_genai_key_from_env_or_ui(ui_gemini_key)
            if not GENAI_AVAILABLE or not api_key:
                st.error("❌ Gemini not available! Configure API key in sidebar.")
                st.stop()
            
            prompt = f"""Return JSON nutrition for "{name}": {{"name": "...", "serving_g": 100, "calories": 200, "protein_g": 10, "carbs_g": 20, "fat_g": 5}}"""
            
            with st.spinner("🤖 AI is analyzing food nutrition..."):
                try:
                    lm_text = lm_query_gemini(prompt, model=gemini_model, api_key=api_key,
                                             max_output_tokens=256, temperature=0.2)
                    parsed, _ = try_parse_json_from_text(lm_text)
                    
                    if parsed:
                        nut = parsed
                        serving = float(nut.get("serving_g", 100) or 100)
                        scale = grams / serving
                        entry = {
                            "user": username, "date": today, "food_id": -1,
                            "name": nut.get("name", name), "grams": grams,
                            "kcal": round(float(nut.get("calories", 0)) * scale, 1),
                            "protein_g": round(float(nut.get("protein_g", 0)) * scale, 1),
                            "carbs_g": round(float(nut.get("carbs_g", 0)) * scale, 1),
                            "fat_g": round(float(nut.get("fat_g", 0)) * scale, 1),
                            "note": "ai_estimate"
                        }
                        diary.setdefault(username, {}).setdefault(today, []).append(entry)
                        save_json("diary.json", diary)
                        st.success(f"✅ Logged **{name}** ({grams}g) - AI estimated values")
                    else:
                        st.error("❌ Failed to parse AI response")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            matched = pd.DataFrame()
            if name:
                try:
                    fid = int(name.strip())
                    matched = foods_df[foods_df["food_id"] == fid]
                except:
                    matched = foods_df[foods_df["name"].str.contains(name, case=False, na=False)]
            
            if matched.empty:
                st.error("❌ Food not found in database. Try AI Lookup or use exact food name/ID.")
            else:
                r = matched.iloc[0]
                serving = float(r.get("serving_g", 100) or 100)
                scale = grams / serving
                entry = {
                    "user": username, "date": today,
                    "food_id": int(r.get("food_id", -1)),
                    "name": r.get("name", "Unknown"), "grams": grams,
                    "kcal": round(float(r.get("calories", 0)) * scale, 1),
                    "protein_g": round(float(r.get("protein_g", 0)) * scale, 1),
                    "carbs_g": round(float(r.get("carbs_g", 0)) * scale, 1),
                    "fat_g": round(float(r.get("fat_g", 0)) * scale, 1)
                }
                diary.setdefault(username, {}).setdefault(today, []).append(entry)
                save_json("diary.json", diary)
                st.success(f"✅ Logged **{r.get('name')}** ({grams}g)")
    
    st.markdown("---")
    
    # Show today's log
    today = datetime.date.today().isoformat()
    entries = diary.get(username, {}).get(today, [])
    
    if entries:
        st.markdown("### 📊 Today's Food Log")
        df = pd.DataFrame(entries)
        st.dataframe(df[["name", "grams", "kcal", "protein_g", "carbs_g", "fat_g"]], 
                    use_container_width=True, hide_index=True)
        
        # Calculate totals
        totals = df[["kcal", "protein_g", "carbs_g", "fat_g"]].sum()
        
        st.markdown("---")
        st.markdown("### 🎯 Daily Totals")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔥 Calories", f"{int(totals['kcal'])}")
        col2.metric("🥩 Protein", f"{int(totals['protein_g'])}g")
        col3.metric("🍞 Carbs", f"{int(totals['carbs_g'])}g")
        col4.metric("🥑 Fat", f"{int(totals['fat_g'])}g")
        
        # Compare with targets if profile exists
        p = users.get(username)
        if p and all(k in p for k in ["sex", "weight_kg", "height_cm", "age", "goal", "activity"]):
            bmr = calc_bmr(p["sex"], p["weight_kg"], p["height_cm"], p["age"])
            tdee = calc_tdee(bmr, p["activity"])
            if p["goal"] == "Lose":
                target = tdee - 400
            elif p["goal"] == "Gain":
                target = tdee + 300
            else:
                target = tdee
            
            remaining = target - totals['kcal']
            st.markdown("---")
            if remaining > 0:
                st.info(f"📊 **Remaining Calories:** {int(remaining)} kcal (Target: {int(target)} kcal)")
            elif remaining < -200:
                st.warning(f"⚠️ **Over Target:** {int(abs(remaining))} kcal (Target: {int(target)} kcal)")
            else:
                st.success(f"✅ **Perfect!** You're right on target ({int(target)} kcal)")
    else:
        st.info("📭 No entries for today. Start logging your meals above!")

# ===== TAB 6: AI COACH =====
with tab6:
    st.markdown("## 💬 AI Fitness Coach")
    st.markdown("*Ask questions about nutrition, training, and fitness*")
    st.markdown("---")
    
    q = st.text_input("🔍 Ask a nutrition/training question", 
                      key="ask_q", 
                      placeholder="e.g., How much protein should I eat daily?")
    
    use_lm_always = st.checkbox(
        "Always use Gemini fallback when local RAG is weak", 
        value=True, 
        key="use_lm_always",
        help="Enable AI for more detailed answers"
    )
    
    if st.button("🚀 Ask Coach", key="ask_button", use_container_width=True):
        if not q or not q.strip():
            st.warning("⚠️ Please enter a question")
            st.stop()
        
        # Check guardrails
        guard = check_guardrails(q)
        if guard["refuse"]:
            st.error(f"❌ {guard['message']}")
            st.info("💡 For medical concerns, please consult a qualified healthcare professional.")
            st.stop()
        
        # Try RAG first
        rag_res = rag.answer(q, topk=2, min_score=0.0)
        top_score = max([s.get("score", 0.0) for s in rag_res.get("sources", [])], default=0.0)
        
        RAG_SCORE_THRESHOLD = 0.08
        
        if rag_res.get("answer") and (not use_lm_always or top_score >= RAG_SCORE_THRESHOLD):
            # RAG answer is good enough
            st.success("📚 **Answer from Local Guidelines:**")
            st.write(rag_res['answer'])
            
            with st.expander("📖 View Sources", expanded=False):
                for s in rag_res["sources"]:
                    st.write(f"**Relevance Score:** {s['score']:.3f}")
                    st.info(s["text"])
                    st.markdown("---")
        else:
            # Use Gemini fallback
            api_key = ensure_genai_key_from_env_or_ui(ui_gemini_key)
            if not GENAI_AVAILABLE or not api_key:
                st.error("❌ Gemini not available for AI fallback! Configure API key in sidebar.")
                if rag_res.get("answer"):
                    st.info("💡 Here's what we found in local guidelines:")
                    st.write(rag_res['answer'])
                st.stop()
            
            # Build prompt with context
            prompt_parts = []
            if rag_res.get("sources"):
                prompt_parts.append("Context from local fitness guidelines:")
                for s in rag_res["sources"]:
                    prompt_parts.append(f"- {s['text']}")
                prompt_parts.append("")
            
            prompt_parts.extend([
                f"User question: {q}",
                "",
                "Instructions:",
                "1) Provide a helpful, concise answer (2-4 sentences).",
                "2) If you used the context above, mention it briefly.",
                "3) If the question requires medical advice, politely decline and recommend consulting a professional.",
                "4) Be encouraging and supportive."
            ])
            
            with st.spinner("🤖 AI Coach is thinking..."):
                try:
                    lm_out = lm_query_gemini(
                        "\n".join(prompt_parts), 
                        model=gemini_model,
                        api_key=api_key, 
                        max_output_tokens=int(gemini_max_tokens),
                        temperature=float(gemini_temp)
                    )
                    
                    if lm_out:
                        st.success("🤖 **AI Coach Response:**")
                        st.write(lm_out)
                        
                        if rag_res.get("sources"):
                            with st.expander("📚 Local Guidelines Used as Context", expanded=False):
                                for s in rag_res["sources"]:
                                    st.write(f"**Score:** {s['score']:.3f}")
                                    st.info(s["text"])
                    else:
                        st.warning("⚠️ AI returned no response. Try rephrasing your question.")
                except Exception as e:
                    st.error(f"❌ AI Coach Error: {e}")
                    if rag_res.get("answer"):
                        st.info("💡 Here's what we found in local guidelines:")
                        st.write(rag_res['answer'])

# ----- Footer -----
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p style='font-size: 1.1rem;'><strong>💪 FitGenius AI</strong></p>
    <p>Built with Streamlit & Google Gemini AI</p>
    <p style='font-size: 0.9rem;'>⚠️ <em>This app provides general fitness information only, not medical advice.</em></p>
    <p style='font-size: 0.9rem;'>Always consult healthcare professionals for personalized medical guidance.</p>
</div>
""", unsafe_allow_html=True)

# Final save
save_json("users.json", users)
save_json("diary.json", diary)