import streamlit as st
import joblib
import numpy as np
import os
from datetime import date
import logging

# ====================== Logging ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== Load Models ======================
try:
    from tensorflow import keras
    DL_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError:
    DL_AVAILABLE = False
    logger.warning("TensorFlow not available — Neural Network disabled")

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    log_reg_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_cols  = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    dl_model = None
    if DL_AVAILABLE:
        dl_model = keras.models.load_model(
            os.path.join(MODELS_DIR, 'neural_network_model.keras')
        )
        logger.info("Neural Network model loaded successfully")

    logger.info(f"All models loaded | Features: {len(feature_cols)}")
except Exception as e:
    st.error(f"error: {e}")
    st.stop()

# ====================== Feature Engineering ======================
def build_features(data: dict) -> np.ndarray:
    """Build features exactly as done during training"""
    try:
        nd   = int(data['no_of_dependents'])
        edu  = 1 if str(data['education']).strip().lower() == 'graduate' else 0
        emp  = 1 if str(data['self_employed']).strip().lower() == 'yes' else 0
        
        inc  = float(data['income_annum'])
        la   = float(data['loan_amount'])
        lt   = float(data['loan_term'])
        cs   = float(data['cibil_score'])
        
        ra   = float(data['residential_assets_value'])
        ca   = float(data['commercial_assets_value'])
        lxa  = float(data['luxury_assets_value'])
        ba   = float(data['bank_asset_value'])

        # Engineered Features 
        total_assets         = ra + ca + lxa + ba
        debt_to_income       = la / (inc + 1e-8)
        loan_to_assets       = la / (total_assets + 1e-8)
        income_per_dependent = inc / (nd + 1)
        
        # CIBIL Category 
        if cs >= 750:
            cibil_category = 3
        elif cs >= 700:
            cibil_category = 2
        elif cs >= 500:
            cibil_category = 1
        else:
            cibil_category = 0

        monthly_loan_burden = la / (lt * 12 + 1)

        row = [
            nd, edu, emp, inc, la, lt, cs,
            ra, ca, lxa, ba,
            total_assets, debt_to_income, loan_to_assets,
            income_per_dependent, cibil_category, monthly_loan_burden
        ]

        return np.array(row, dtype=float).reshape(1, -1)

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid input data: {str(e)}")


# ====================== Streamlit Configuration ======================
st.set_page_config(
    page_title="Al-Salam Bank - New Loan Application",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== Full CSS & JavaScript ======================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    direction: ltr;
    margin: 0; padding: 0;
}
#MainMenu, footer, header { visibility: hidden; }

/* ───────────────────────────────────────────
   BASE LAYOUT (Desktop ≥1024px)
─────────────────────────────────────────── */
.block-container {
    max-width: 100% !important;
    padding-top: 80px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
}
.main {
    margin-left: 400px; 
    padding: 20px;
}

/* ── Navbar ── */
.navbar {
    background-color: white;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding: 0 28px 0 240px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 1000;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}

.navbar-links {
    display: flex;
    gap: 28px;
    align-items: center;
    list-style: none;
    margin: 0; padding: 0;
    flex-wrap: wrap;
}

.navbar-links li a {
    color: #1d3557;
    text-decoration: none;
    font-size: 13.5px;
    font-weight: 600;
    white-space: nowrap;
    padding-bottom: 4px;
    transition: all .3s ease;
}

.navbar-links li a:hover { color: #0b5ed7; }

.navbar-links li a.active {
    color: #003b8e;
    font-weight: 700;
    border-bottom: 2.5px solid #ffd43b;
    padding-bottom: 2px;
}

.navbar-right {
    display: flex;
    align-items: center;
    margin-right: 80px;
    gap: 12px;
    flex-shrink: 0;
}

.search-box {
    border: 1px solid rgba(200,210,230,0.8);
    border-radius: 20px;
    padding: 7px 14px 7px 36px;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    color: #333;
    width: 190px;
    outline: none;
    background:
        #f8f9fe
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2.5'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E")
        no-repeat 11px center;
}

.nav-filter-btn {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: rgba(255,255,255,0.25);
    border: 1px solid rgba(200,210,230,0.6);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 14px; color: #1d3557;
    transition: all .3s ease;
    margin: 0 6px;
}
.nav-filter-btn:hover { background: rgba(255,255,255,0.45); transform: translateY(-2px); }

.nav-bell {
    position: relative; width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 18px; color: #1d3557;
}
.nav-bell-dot {
    position: absolute; top: 4px; right: 4px;
    width: 9px; height: 9px;
    background: #ff3b30; border-radius: 50%;
    border: 2px solid white;
}

.nav-avatar {
    width: 40px; height: 40px;
    border-radius: 50%; object-fit: cover;
    border: 2px solid #ffd43b; cursor: pointer;
    box-shadow: 0 0 12px rgba(255,212,59,0.4);
    margin: 0 8px;
}

/* ── Hamburger (mobile only) ── */
.hamburger-btn {
    display: none;
    flex-direction: column; justify-content: center;
    align-items: center; gap: 5px;
    width: 36px; height: 36px; cursor: pointer;
    background: none; border: none; padding: 0;
}
.hamburger-btn span {
    display: block; width: 22px; height: 2px;
    background: #1d3557; border-radius: 2px;
    transition: all .3s;
}

/* ── Sidebar ── */
.sidebar-nav {
    position: fixed; top: 0; left: 0;
    width: 220px; height: 100vh;
    background: linear-gradient(135deg, #1f2d46 0%, #2a3346 35%, #444 75%, #59554b 100%);
    padding: 20px 0 19px 0;
    display: flex; flex-direction: column; gap: 2px;
    overflow-y: auto; z-index: 1100;
    box-shadow: 3px 0 18px rgba(0,0,0,0.25);
    border-right: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.3s ease;
}

.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0 16px 18px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
.sidebar-logo-icon { font-size: 30px; }
.sidebar-logo-text { font-size: 15px; font-weight: 800; }
.sidebar-logo-text .t1 { color: #fff; }
.sidebar-logo-text .t2 { color: #c9a227; }

.nav-item {
    display: flex; align-items: center; gap: 11px;
    padding: 10px 14px; color: rgba(255,255,255,0.65);
    font-size: 13px; font-weight: 600; cursor: pointer;
    border-left: 3px solid transparent; white-space: nowrap;
    margin: 1px 8px 20px 0; border-radius: 9px;
    transition: background 0.18s, color 0.18s;
}
.nav-item:hover { background: rgba(255,255,255,0.07); color: #fff; }
.nav-item.active {
    background: rgba(201,162,39,0.15);
    color: #f0c842;
    border-left: 3px solid #c9a227;
    font-weight: 700;
}
.nav-icon { font-size: 15px; }

/* ── Page title ── */
.page-title {
    font-size: 36px; font-weight: 800;
    color: #1a3a6b; margin-bottom: 0;
}

/* ── Card border wrapper ── */
div[data-testid="stForm"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
div[data-testid="stForm"] > div {
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(26,58,107,0.09) !important;
    border: 1.5px solid #e2e8f4 !important;
    background: #fff !important;
    padding: 6px 10px 16px !important;
}

/* ── Card header ── */
.card-header {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0 10px 0;
    border-bottom: 1.5px solid #e8edf5;
    margin-bottom: 14px;
}
.card-num {
    background: linear-gradient(135deg, #1a3a6b, #2756a8);
    color: #fff; border-radius: 50%;
    width: 24px; height: 24px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; flex-shrink: 0;
}
.card-title { font-size: 20px; font-weight: 800; color: #1a3a6b; }

/* ── Inputs ── */
.stTextInput > label, .stNumberInput > label,
.stSelectbox > label, .stDateInput > label, .stTextArea > label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; color: #374151 !important;
    font-size: 12px !important;
}
.stTextInput input, .stNumberInput input,
.stSelectbox select, .stDateInput input, .stTextArea textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    border: 1px solid #d0d8ec !important;
    border-radius: 8px !important;
    background: #f8f9fe !important;
    padding: 6px 10px !important;
}

/* ── Buttons ── */
div[data-testid="stFormSubmitButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; border-radius: 8px !important;
    height: 42px; font-size: 14px !important;
    background: linear-gradient(135deg, #1a3a6b, #2756a8) !important;
    color: #fff !important; border: none !important; width: 100%;
}
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; border-radius: 8px !important;
    height: 42px; font-size: 14px !important;
    background: #fff !important; color: #1a3a6b !important;
    border: 2px solid #1a3a6b !important; width: 100%;
}

/* ── Result badges ── */
.model-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 12px; font-weight: 700; margin-bottom: 6px;
}
.badge-lr { background: #e3f2fd; color: #1565c0; }
.badge-nn { background: #f3e5f5; color: #6a1b9a; }

/* ───────────────────────────────────────────
   TABLET  (768px – 1023px)
─────────────────────────────────────────── */
@media (max-width: 1023px) {

    .block-container {
        padding-top: 72px !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
    }

    /* Sidebar slides off-screen by default on tablets */
    .sidebar-nav {
        transform: translateX(-100%);
        width: 200px;
        z-index: 1200;
    }
    .sidebar-nav.open { transform: translateX(0); }

    /* Navbar no longer needs left-padding for sidebar */
    .navbar {
        padding: 0 16px;
        margin-left: 0 !important;
    }

    .navbar-right { margin-right: 0; }

    .search-box { width: 140px; }

    .hamburger-btn { display: flex; }

    .page-title { font-size: 26px; }
    .card-title { font-size: 17px; }

    /* Overlay when sidebar is open */
    .sidebar-overlay {
        display: none;
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.45);
        z-index: 1150;
    }
    .sidebar-overlay.active { display: block; }
}

/* ───────────────────────────────────────────
   MOBILE  (≤767px)
─────────────────────────────────────────── */
@media (max-width: 767px) {

    .block-container {
        padding-top: 68px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    .navbar {
        padding: 0 10px;
        height: 58px;
    }

    /* Hide long nav links on small screens */
    .navbar-links { display: none; }

    /* Show hamburger */
    .hamburger-btn { display: flex; }

    /* Search hidden on mobile to save space */
    .search-box { display: none; }

    .navbar-right { gap: 6px; margin-right: 0; }

    .nav-avatar { width: 32px; height: 32px; margin: 0 4px; }

    .page-title { font-size: 20px; margin-bottom: 12px; }

    .card-header { flex-direction: row; }
    .card-title { font-size: 15px; }

    /* Stack Streamlit columns vertically on mobile */
    /* Streamlit uses flex for columns; override widths */
    section[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    section[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 6px 8px 12px !important;
        margin-bottom: 12px !important;
    }

    /* Inputs – make them taller for touch */
    .stTextInput input, .stNumberInput input,
    .stSelectbox select, .stDateInput input, .stTextArea textarea {
        font-size: 14px !important;
        padding: 10px 12px !important;
        min-height: 42px !important;
    }

    .stTextInput > label, .stNumberInput > label,
    .stSelectbox > label, .stDateInput > label, .stTextArea > label {
        font-size: 13px !important;
    }

    div[data-testid="stFormSubmitButton"] > button,
    .stButton > button {
        height: 46px;
        font-size: 15px !important;
    }

    /* Result cards stack */
    .result-row {
        flex-direction: column !important;
        gap: 10px !important;
    }
}

@media (max-width: 400px) {
    .page-title { font-size: 17px; }
    .card-title { font-size: 13px; }
    .navbar { height: 54px; }
    .block-container { padding-top: 62px !important; }
}

</style>

<script>
/* Toggle sidebar on mobile/tablet */
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar-nav');
    const overlay = document.querySelector('.sidebar-overlay');
    if (sidebar) sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('active');
}
</script>
""", unsafe_allow_html=True)

# Sidebar Overlay
st.markdown('<div class="sidebar-overlay" onclick="toggleSidebar()"></div>', unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
  <!-- Hamburger: only visible on tablet/mobile -->
  <button class="hamburger-btn" onclick="toggleSidebar()" aria-label="Open menu">
    <span></span><span></span><span></span>
  </button>

  <ul class="navbar-links">
    <li><a href="#" class="active">Home</a></li>
    <li><a href="#">Services</a></li>
    <li><a href="#">Products</a></li>
    <li><a href="#">Contact Us</a></li>
    <li><a href="#">User Profile</a></li>
  </ul>

  <div class="navbar-right">
    <input class="search-box" type="text" placeholder="Search...">
    <div class="nav-filter-btn">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2.2">
        <line x1="4" y1="6" x2="20" y2="6"/>
        <line x1="8" y1="12" x2="16" y2="12"/>
        <line x1="11" y1="18" x2="13" y2="18"/>
      </svg>
    </div>
    <div class="nav-bell">
      🔔
      <span class="nav-bell-dot"></span>
    </div>
    <img class="nav-avatar" src="https://i.pravatar.cc/40?img=12" alt="User Avatar">
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
col_sidebar, col_main = st.columns([1, 6])
with col_sidebar:
    st.markdown("""
    <div class="sidebar-nav" id="sidebarNav">
      <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🌙</span>
        <span class="sidebar-logo-text">
          <span class="t1">Al-Salam </span><span class="t2">Bank</span>
        </span>
      </div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">🏠</span> Dashboard</div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">📋</span> My Applications</div>
      <div class="nav-item active" onclick="toggleSidebar()"><span class="nav-icon">➕</span> Create New Loan</div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">👥</span> Client Database</div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">🏦</span> Bank Accounts</div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">📊</span> Detailed Reports</div>
      <div class="nav-item" onclick="toggleSidebar()"><span class="nav-icon">⚙️</span> System Settings</div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
with col_main:
    st.markdown('<div class="page-title">New Loan Application</div>', unsafe_allow_html=True)

    with st.form("loan_form"):

        # ── ROW 1 ──
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                  <span class="card-num">1</span>
                  <span class="card-title">Customer Information</span>
                </div>
                """, unsafe_allow_html=True)
                full_name   = st.text_input("Full Name", placeholder="Full Name")
                national_id = st.text_input("National ID Number", placeholder="National ID Number")
                dob         = st.date_input("Date of Birth (DOB)",
                                            min_value=date(1960, 1, 1),
                                            max_value=date.today())
                phone       = st.text_input("Phone Number", placeholder="Phone Number")

        with col2:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                  <span class="card-num">2</span>
                  <span class="card-title">Loan Details</span>
                </div>
                """, unsafe_allow_html=True)
                loan_type    = st.selectbox("Loan Type", ["Personal", "Mortgage", "Commercial", "Auto"])
                loan_amount  = st.number_input("Requested Amount (€)", min_value=0, value=15_000_000, step=100_000)
                loan_term    = st.number_input("Repayment Period (years)", min_value=1, max_value=30, value=5)
                cibil_score  = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
                loan_purpose = st.text_area("Loan Purpose",
                                            placeholder="Describe the purpose of this loan",
                                            height=80)

        # ── ROW 2 ──
        col3, col4 = st.columns([1, 1])

        with col3:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                  <span class="card-num">3</span>
                  <span class="card-title">Financial Info</span>
                </div>
                """, unsafe_allow_html=True)
                monthly_income = st.number_input("Monthly Income")
                employer       = st.text_input("Employer")
                education      = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed  = st.selectbox("Self Employed", ["No", "Yes"])

        with col4:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                  <span class="card-num">4</span>
                  <span class="card-title">Assets</span>
                </div>
                """, unsafe_allow_html=True)
                no_of_dependents   = st.number_input("Dependents")
                residential_assets = st.number_input("Residential Assets")
                commercial_assets  = st.number_input("Commercial Assets")
                luxury_assets      = st.number_input("Luxury Assets")
                bank_asset_value   = st.number_input("Bank Assets")

        # ── Action Buttons ──
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            draft_btn  = st.form_submit_button("💾  Save as Draft")
        with b2:
            submit_btn = st.form_submit_button("📤  Submit Application")

# ====================== Prediction ======================
if submit_btn:
    payload = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": monthly_income * 12,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets,
        "commercial_assets_value": commercial_assets,
        "luxury_assets_value": luxury_assets,
        "bank_asset_value": bank_asset_value,
        "model": "both"
    }

    try:
        X_raw = build_features(payload)
        X_scaled = scaler.transform(X_raw)

        result = {}

        # Logistic Regression
        proba = log_reg_model.predict_proba(X_scaled)[0]
        pred = log_reg_model.predict(X_scaled)[0]
        label = label_encoder.inverse_transform([pred])[0]

        result['logistic_regression'] = {
            "prediction": label,
            "confidence": round(float(max(proba)), 4),
            "probability_approved": round(float(proba[1]), 4),
            "probability_rejected": round(float(proba[0]), 4)
        }

        # Neural Network
        if dl_model and DL_AVAILABLE:
            dl_prediction = dl_model.predict(X_scaled, verbose=0)
            dl_proba = float(dl_prediction[0][0])
            dl_pred = 1 if dl_proba >= 0.5 else 0
            dl_label = label_encoder.inverse_transform([dl_pred])[0]

            result['neural_network'] = {
                "prediction": dl_label,
                "confidence": round(max(dl_proba, 1 - dl_proba), 4),
                "probability_approved": round(dl_proba, 4),
                "probability_rejected": round(1 - dl_proba, 4)
            }

        # Display Results
        st.markdown("---")
        st.markdown('<p style="font-size:1rem;font-weight:800;color:#1a3a6b;margin-bottom:0.8rem;">' \
        '📊 Analysis Results</p>', unsafe_allow_html=True)



        lr_pred = result['logistic_regression']['prediction']
        nn_pred = result['neural_network']['prediction']
        is_consistent = (lr_pred == nn_pred)
        if is_consistent:
            
            rc1, rc2 = st.columns([1,1])
            cols_to_display = [
                (rc1, "logistic_regression", "badge-lr", "Logistic Regression"),
                (rc2, "neural_network", "badge-nn", "Neural Network")
        ]
        else:
            rc1 = st.container() 
            cols_to_display = [
                (rc1, "logistic_regression", "badge-lr", "Logistic Regression (Priority Decision)")
        ]
        for col, key, badge_cls, label in cols_to_display:
            if key in result:
                pred_info = result[key]
                approved = pred_info["prediction"].lower() == "approved"
                color = "#2e7d32" if approved else "#c62828"
                bg = "#e8f5e9" if approved else "#ffebee"
                with col:
                    st.markdown(f"""
                    <div style="background:{bg}; border-radius:12px; padding:1rem; 
                    text-align:center; border:1px solid {color}33;">
                        <span class="model-badge {badge_cls}">{label}</span><br>
                        <span style="font-size:1.4rem; font-weight:800; color:{color};">
                        {pred_info['prediction']}</span><br>
                        <span style="font-size:0.9rem; color:#555;">
                        Confidence: {pred_info['confidence']*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                   
        # Consensus
        if 'logistic_regression' in result and 'neural_network' in result:
            lr_pred = result['logistic_regression']['prediction']
            nn_pred = result['neural_network']['prediction']
            
            is_consistent = (lr_pred == nn_pred)
            
            final_decision = lr_pred 
            
            approved = final_decision.lower() == "approved"
            bg_color = "#e8f5e9" if approved else "#ffebee"
            border_color = "#2e7d32" if approved else "#c62828"
            text_color = "#1b5e20" if approved else "#b71c1c"

            st.markdown(f"""
            <div style="background:{bg_color}; border:2px solid {border_color}; border-radius:12px;
                        padding:1.5rem; text-align:center; margin-top:1.5rem;
                        font-weight:800; color:{text_color}; font-size:1.2rem;">
                Final Decision: {final_decision}
            </div>
            """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Error during prediction: {e}")

if draft_btn:
    st.success("Application saved as draft successfully")