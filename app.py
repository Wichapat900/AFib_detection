"""
app.py — AFib Detection Web App
================================
Streamlit app for detecting Atrial Fibrillation from ECG signals.

Models supported:
  - XGBoost  (HRV features — loads models/xgb.pkl)
  - CatBoost (HRV features — loads models/catboost.pkl)
  - CNN / LSTM / RNN / CNN+LSTM  (raw signal — loads models/*.pth)
  - HRV heuristic (no trained model required — always available)

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import io, pickle

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

# ── optional heavy deps ─────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AFib Detector",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .stCode, .stMarkdown code { font-family: 'DM Mono', monospace !important; }
.stApp { background: #0b0f1a; color: #e0e6f0; }
section[data-testid="stSidebar"] { background: #0f1424; border-right: 1px solid #1e2740; }
section[data-testid="stSidebar"] * { color: #c8d4e8 !important; }
.result-afib {
    background: linear-gradient(135deg,#3d0a0a,#1a0505);
    border: 2px solid #e53e3e; border-radius: 12px;
    padding: 1.6rem 2rem; text-align: center;
}
.result-normal {
    background: linear-gradient(135deg,#0a2e1a,#051210);
    border: 2px solid #38a169; border-radius: 12px;
    padding: 1.6rem 2rem; text-align: center;
}
.result-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0.3rem 0; }
.prob-bar-wrap { background:#1e2740; border-radius:99px; height:10px; overflow:hidden; margin:0.5rem 0; }
.prob-bar-fill  { height:100%; border-radius:99px; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; letter-spacing:-0.02em; }
.stButton>button { background:#2563eb; color:white; border:none; border-radius:8px;
    font-family:'Syne',sans-serif; font-weight:600; padding:0.5rem 1.4rem; }
.stButton>button:hover { background:#1d4ed8; }
.disclaimer { background:#1a1f30; border-left:3px solid #f6ad55; border-radius:4px;
    padding:0.7rem 1rem; font-size:0.82rem; color:#a0aec0; margin-top:0.5rem; }
.model-badge { display:inline-block; background:#1e2740; border:1px solid #2a3558;
    border-radius:6px; padding:0.2rem 0.6rem; font-size:0.75rem;
    font-family:'DM Mono',monospace; color:#63b3ed; margin-right:4px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
FS     = 128
WINDOW = 3840   # 30 s × 128 Hz

FEATURE_NAMES = [
    "mean_rr","median_rr","sdnn","rmssd","pnn50","cv_rr",
    "mean_hr","std_hr","min_hr","max_hr",
    "mean_diff","max_diff","sd1","sd2","sd_ratio",
    "skewness","kurtosis","iqr","irr_score",
    "lf_hf","lf_norm","hf_norm","dominant_freq","n_beats",
]
FEATURE_UNITS = {
    "mean_rr":"ms","median_rr":"ms","sdnn":"ms","rmssd":"ms","pnn50":"%",
    "cv_rr":"","mean_hr":"bpm","std_hr":"bpm","min_hr":"bpm","max_hr":"bpm",
    "mean_diff":"ms","max_diff":"ms","sd1":"ms","sd2":"ms","sd_ratio":"",
    "skewness":"","kurtosis":"","iqr":"ms","irr_score":"","lf_hf":"",
    "lf_norm":"","hf_norm":"","dominant_freq":"Hz","n_beats":"",
}
FEATURE_DESCRIPTIONS = {
    "mean_rr":"Average RR interval","median_rr":"Median RR interval",
    "sdnn":"Std dev of RR intervals","rmssd":"Root mean square of successive differences",
    "pnn50":"% RR differences > 50 ms","cv_rr":"Coefficient of variation",
    "mean_hr":"Mean heart rate","std_hr":"HR standard deviation",
    "min_hr":"Minimum heart rate","max_hr":"Maximum heart rate",
    "mean_diff":"Mean absolute successive difference",
    "max_diff":"Max absolute successive difference",
    "sd1":"Poincaré SD1 (short-term variability)","sd2":"Poincaré SD2 (long-term variability)",
    "sd_ratio":"SD1/SD2 ratio","skewness":"RR interval skewness",
    "kurtosis":"RR interval kurtosis","iqr":"Interquartile range of RR",
    "irr_score":"Irregularity score","lf_hf":"LF/HF power ratio",
    "lf_norm":"Normalized LF power","hf_norm":"Normalized HF power",
    "dominant_freq":"Dominant frequency","n_beats":"Number of detected R-peaks",
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_filter(signal, fs=FS, low=0.5, high=40.0):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)

def preprocess(signal, fs=FS):
    sig = bandpass_filter(signal, fs)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

def detect_rpeaks(signal, fs=FS):
    sig = preprocess(signal, fs)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig))
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3*fs), prominence=sig_max*0.2)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.2, sig_max*0.2), distance=int(0.3*fs))
    return peaks

def extract_hrv(signal, fs=FS):
    peaks = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2000)]
    if len(rr) < 4:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    mean_rr   = float(np.mean(rr));  median_rr = float(np.median(rr))
    sdnn      = float(np.std(rr));   rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
    pnn50     = float(np.mean(np.abs(np.diff(rr)) > 50))
    cv_rr     = sdnn / mean_rr if mean_rr > 0 else 0.0
    hr        = 60000.0 / rr
    mean_hr   = float(np.mean(hr)); std_hr = float(np.std(hr))
    min_hr    = float(np.min(hr));  max_hr = float(np.max(hr))
    diff_rr   = np.diff(rr)
    mean_diff = float(np.mean(np.abs(diff_rr)))
    max_diff  = float(np.max(np.abs(diff_rr))) if len(diff_rr) > 0 else 0.0
    sd1       = float(np.sqrt(0.5) * np.std(diff_rr))
    sd2_sq    = max(0, 2*sdnn**2 - 0.5*np.std(diff_rr)**2)
    sd2       = float(np.sqrt(sd2_sq))
    sd_ratio  = sd1 / (sd2 + 1e-8)
    sk        = float(skew(rr)); ku = float(kurtosis(rr))
    iqr_val   = float(np.percentile(rr,75) - np.percentile(rr,25))
    irr       = float(np.sum(np.abs(diff_rr)/(rr[:-1]+1e-9) > 0.10) / max(len(diff_rr),1))
    try:
        t_rr  = np.cumsum(rr) / 1000.0
        t_uni = np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(t_uni) > 8:
            rr_uni = interp1d(t_rr, rr, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(t_uni)
            rr_uni -= np.mean(rr_uni)
            freqs, psd = welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
            pos = freqs > 0; freqs, psd = freqs[pos], psd[pos]
            lf  = np.sum(psd[(freqs>=0.04)&(freqs<0.15)])
            hf  = np.sum(psd[(freqs>=0.15)&(freqs<0.40)])
            tot = lf + hf + 1e-9
            lf_hf=lf/(hf+1e-9); lf_norm=lf/tot; hf_norm=hf/tot
            dom_freq=float(freqs[np.argmax(psd)])
        else:
            lf_hf=lf_norm=hf_norm=dom_freq=0.0
    except Exception:
        lf_hf=lf_norm=hf_norm=dom_freq=0.0
    return np.array([
        mean_rr,median_rr,sdnn,rmssd,pnn50,cv_rr,
        mean_hr,std_hr,min_hr,max_hr,
        mean_diff,max_diff,sd1,sd2,sd_ratio,
        sk,ku,iqr_val,irr,
        lf_hf,lf_norm,hf_norm,dom_freq,float(len(peaks)),
    ], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  (cached)
# ═══════════════════════════════════════════════════════════════════════════

def _load_pkl(path: str):
    """Load any .pkl model file using joblib (preferred) or pickle."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        if JOBLIB_AVAILABLE:
            return joblib.load(str(p))
        else:
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

@st.cache_resource
def load_xgb_model(path: str):
    if not XGB_AVAILABLE:
        return None
    return _load_pkl(path)

@st.cache_resource
def load_catboost_model(path: str):
    if not CATBOOST_AVAILABLE:
        return None
    return _load_pkl(path)

@st.cache_resource
def load_deep_model(model_name: str, weights_path: str):
    if not TORCH_AVAILABLE:
        return None
    p = Path(weights_path)
    if not p.exists():
        return None
    try:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from model import AFibCNN, AFibLSTM, AFibRNN, AFibCNNLSTM
        cls_map = {"CNN":AFibCNN,"LSTM":AFibLSTM,"RNN":AFibRNN,"CNN+LSTM":AFibCNNLSTM}
        m = cls_map[model_name]()
        m.load_state_dict(torch.load(weights_path, map_location="cpu"))
        m.eval()
        return m
    except Exception as e:
        st.warning(f"Deep model load failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_xgb(model, features: np.ndarray):
    x = features.reshape(1, -1)
    prob = float(model.predict_proba(x)[0][1])
    return ("AFib" if prob >= 0.5 else "Normal"), prob

def predict_catboost(model, features: np.ndarray):
    x = features.reshape(1, -1)
    prob = float(model.predict_proba(x)[0][1])
    return ("AFib" if prob >= 0.5 else "Normal"), prob

def predict_deep(model, signal: np.ndarray):
    sig = signal.copy().astype(np.float32)
    sig = sig[:WINDOW] if len(sig) >= WINDOW else np.pad(sig, (0, WINDOW-len(sig)))
    x = torch.tensor(sig).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()[0]
    return ("AFib" if probs[1] >= 0.5 else "Normal"), float(probs[1])

def hrv_heuristic(features: np.ndarray):
    feat = dict(zip(FEATURE_NAMES, features))
    score = 0.0; reasons = {}
    sdnn_n  = min(feat["sdnn"]/150.0, 1.0);      reasons["SDNN (variability)"]  = sdnn_n;  score += sdnn_n*0.25
    pnn50_n = min(feat["pnn50"]/0.5, 1.0);        reasons["pNN50"]               = pnn50_n; score += pnn50_n*0.20
    irr_n   = min(feat["irr_score"]/0.4, 1.0);    reasons["Irregularity score"]  = irr_n;   score += irr_n*0.25
    rmssd_n = min(feat["rmssd"]/120.0, 1.0);      reasons["RMSSD"]               = rmssd_n; score += rmssd_n*0.15
    lfhf_i  = max(0.0,1.0-min(feat["lf_hf"]/2.0,1.0)); reasons["LF/HF imbalance"] = lfhf_i; score += lfhf_i*0.10
    cv_n    = min(feat["cv_rr"]/0.25, 1.0);       reasons["CV of RR"]            = cv_n;    score += cv_n*0.05
    prob = float(np.clip(score, 0.0, 1.0))
    return ("AFib" if prob >= 0.45 else "Normal"), prob, reasons

# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DEMO
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_ecg(afib=False, seed=42, fs=FS, duration_s=30):
    rng = np.random.default_rng(seed)
    n   = fs * duration_s
    rr_intervals = (rng.exponential(60/90, 200) if afib
                    else rng.normal(60/65, 0.02, 200))
    rr_intervals = np.clip(rr_intervals, 0.25, 1.5)
    beat_times   = np.cumsum(rr_intervals)
    beat_times   = beat_times[beat_times < duration_s]
    beat_samples = (beat_times * fs).astype(int)
    ecg = rng.normal(0, 0.05, n)
    for bs in beat_samples:
        if bs + 30 < n:
            ecg[bs:bs+5] += np.array([-0.1,0.3,1.5,0.3,-0.2])
            ecg[bs+10:bs+30] += np.sin(np.linspace(0,np.pi,20))*0.3
        if bs - 25 >= 0:
            ecg[bs-25:bs-5] += np.sin(np.linspace(0,np.pi,20))*0.15
    return bandpass_filter(ecg, fs).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_ecg(signal, peaks, fs=FS, title="ECG Signal"):
    t = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines",
        line=dict(color="#4299e1",width=1.2), name="ECG",
        hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>"))
    if len(peaks):
        fig.add_trace(go.Scatter(x=t[peaks], y=signal[peaks], mode="markers",
            marker=dict(color="#fc8181",size=7), name="R-peaks"))
    fig.update_layout(title=dict(text=title,font=dict(color="#e0e6f0",size=15)),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        xaxis=dict(title="Time (s)",color="#8fa3cc",gridcolor="#1e2740",zeroline=False),
        yaxis=dict(title="Amplitude (norm.)",color="#8fa3cc",gridcolor="#1e2740",zeroline=False),
        legend=dict(bgcolor="#111827",font=dict(color="#8fa3cc")),
        margin=dict(l=50,r=20,t=40,b=40), hovermode="x unified")
    return fig

def plot_rr(rr_ms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rr_ms, mode="lines+markers",
        line=dict(color="#68d391",width=1.5), marker=dict(size=4,color="#68d391")))
    fig.add_hrect(y0=600,y1=1000,fillcolor="#2d6a4f",opacity=0.15,line_width=0,
        annotation_text="Normal range",annotation_position="top left",
        annotation=dict(font_color="#68d391",font_size=11))
    fig.update_layout(title=dict(text="RR Interval Tachogram",font=dict(color="#e0e6f0",size=15)),
        paper_bgcolor="#111827",plot_bgcolor="#111827",
        xaxis=dict(title="Beat index",color="#8fa3cc",gridcolor="#1e2740"),
        yaxis=dict(title="RR (ms)",color="#8fa3cc",gridcolor="#1e2740"),
        margin=dict(l=50,r=20,t=40,b=40))
    return fig

def plot_poincare(rr_ms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_ms[:-1],y=rr_ms[1:],mode="markers",
        marker=dict(color="#b794f4",size=4,opacity=0.7)))
    fig.update_layout(title=dict(text="Poincaré Plot (RRₙ vs RRₙ₊₁)",font=dict(color="#e0e6f0",size=15)),
        paper_bgcolor="#111827",plot_bgcolor="#111827",
        xaxis=dict(title="RRₙ (ms)",color="#8fa3cc",gridcolor="#1e2740"),
        yaxis=dict(title="RRₙ₊₁ (ms)",color="#8fa3cc",gridcolor="#1e2740"),
        margin=dict(l=50,r=20,t=40,b=40))
    return fig

def plot_radar(features):
    keys   = ["sdnn","rmssd","pnn50","irr_score","cv_rr","sd1","sd2","lf_hf"]
    norms  = {"sdnn":150,"rmssd":120,"pnn50":1,"irr_score":0.5,
               "cv_rr":0.3,"sd1":80,"sd2":150,"lf_hf":5}
    labels = ["SDNN","RMSSD","pNN50","Irregularity","CV(RR)","SD1","SD2","LF/HF"]
    feat   = dict(zip(FEATURE_NAMES, features))
    vals   = [min(feat[k]/norms[k],1.0) for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]], fill="toself",
        fillcolor="rgba(99,179,237,0.15)", line=dict(color="#63b3ed",width=2)))
    fig.update_layout(
        polar=dict(bgcolor="#111827",
            radialaxis=dict(visible=True,range=[0,1],color="#8fa3cc",
                            gridcolor="#1e2740",tickfont=dict(color="#8fa3cc")),
            angularaxis=dict(color="#8fa3cc",gridcolor="#1e2740")),
        paper_bgcolor="#111827",
        title=dict(text="HRV Radar",font=dict(color="#e0e6f0",size=15)),
        margin=dict(l=40,r=40,t=50,b=40), showlegend=False)
    return fig

def plot_feature_importance_xgb(model):
    """Bar chart of XGBoost feature importances."""
    try:
        scores = model.get_booster().get_fscore()
        if not scores:
            scores = dict(zip([f"f{i}" for i in range(len(FEATURE_NAMES))],
                              model.feature_importances_))
        # map f0..f23 → feature names
        named = {}
        for k, v in scores.items():
            try:
                idx = int(k.replace("f",""))
                named[FEATURE_NAMES[idx]] = v
            except Exception:
                named[k] = v
        df = pd.DataFrame({"Feature":list(named.keys()),"Importance":list(named.values())})
        df = df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=df["Importance"], y=df["Feature"], orientation="h",
            marker_color="#63b3ed"))
        fig.update_layout(title=dict(text="XGBoost Feature Importance",
            font=dict(color="#e0e6f0",size=14)),
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            xaxis=dict(color="#8fa3cc",gridcolor="#1e2740"),
            yaxis=dict(color="#8fa3cc"),
            margin=dict(l=10,r=20,t=40,b=30), height=380)
        return fig
    except Exception:
        return None

def plot_feature_importance_cb(model):
    """Bar chart of CatBoost feature importances."""
    try:
        imps = model.get_feature_importance()
        df = pd.DataFrame({"Feature": FEATURE_NAMES[:len(imps)], "Importance": imps})
        df = df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=df["Importance"], y=df["Feature"], orientation="h",
            marker_color="#f6ad55"))
        fig.update_layout(title=dict(text="CatBoost Feature Importance",
            font=dict(color="#e0e6f0",size=14)),
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            xaxis=dict(color="#8fa3cc",gridcolor="#1e2740"),
            yaxis=dict(color="#8fa3cc"),
            margin=dict(l=10,r=20,t=40,b=30), height=380)
        return fig
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🫀 AFib Detector")
        st.markdown("---")

        st.markdown("### Input Source")
        input_mode = st.radio("Choose input",
            ["Demo — Normal ECG","Demo — AFib ECG","Upload .npy file","Upload .csv file"],
            label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### Signal Settings")
        fs_input = st.number_input("Sampling rate (Hz)", min_value=64, max_value=2000,
                                   value=128, step=1)
        window_index = st.number_input("Window index (multi-window files)",
                                       min_value=0, value=0, step=1)

        st.markdown("---")
        st.markdown("### Model Selection")

        MODEL_OPTIONS = []
        MODEL_OPTIONS.append("HRV Heuristic (no weights needed)")
        if XGB_AVAILABLE:
            MODEL_OPTIONS.append("XGBoost")
        else:
            st.caption("⚠️ xgboost not installed")
        if CATBOOST_AVAILABLE:
            MODEL_OPTIONS.append("CatBoost")
        else:
            st.caption("⚠️ catboost not installed")
        if TORCH_AVAILABLE:
            MODEL_OPTIONS += ["CNN","LSTM","RNN","CNN+LSTM"]
        else:
            st.caption("⚠️ torch not installed")

        model_choice = st.selectbox("Model", MODEL_OPTIONS)

        # Weights path — shown only when relevant
        weights_path = ""
        if model_choice == "XGBoost":
            weights_path = st.text_input("XGBoost model path (.pkl)",
                                         value="models/xgb.pkl")
        elif model_choice == "CatBoost":
            weights_path = st.text_input("CatBoost model path (.pkl)",
                                         value="models/catboost.pkl")
        elif model_choice in ("CNN","LSTM","RNN","CNN+LSTM"):
            weights_path = st.text_input("PyTorch weights path (.pth)",
                                         value=f"models/{model_choice.lower()}_best.pth")

        st.markdown("---")
        st.markdown(
            "<div class='disclaimer'>⚠️ <b>Not for clinical use.</b> Research prototype only.</div>",
            unsafe_allow_html=True)

    # ── HEADER ───────────────────────────────────────────────────────────
    st.markdown("# 🫀 Atrial Fibrillation Detection")
    st.markdown(
        "Upload an ECG segment or use a synthetic demo to extract HRV features "
        "and run AFib prediction.")

    # availability badges
    badges = []
    badges.append(f'<span class="model-badge" style="border-color:#68d391;color:#68d391">HRV Heuristic ✓</span>')
    col = "#63b3ed" if XGB_AVAILABLE else "#718096"
    badges.append(f'<span class="model-badge" style="border-color:{col};color:{col}">XGBoost {"✓" if XGB_AVAILABLE else "✗"}</span>')
    col = "#f6ad55" if CATBOOST_AVAILABLE else "#718096"
    badges.append(f'<span class="model-badge" style="border-color:{col};color:{col}">CatBoost {"✓" if CATBOOST_AVAILABLE else "✗"}</span>')
    col = "#b794f4" if TORCH_AVAILABLE else "#718096"
    badges.append(f'<span class="model-badge" style="border-color:{col};color:{col}">PyTorch (CNN/LSTM) {"✓" if TORCH_AVAILABLE else "✗"}</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown("")

    # ── LOAD SIGNAL ──────────────────────────────────────────────────────
    signal = None; signal_label = ""

    if input_mode.startswith("Demo — Normal"):
        signal = make_synthetic_ecg(afib=False, seed=7, fs=fs_input)
        signal_label = "Synthetic Normal ECG"
    elif input_mode.startswith("Demo — AFib"):
        signal = make_synthetic_ecg(afib=True, seed=3, fs=fs_input)
        signal_label = "Synthetic AFib ECG"
    elif input_mode == "Upload .npy file":
        uploaded = st.file_uploader("Upload .npy ECG file", type=["npy"])
        if uploaded:
            data = np.load(io.BytesIO(uploaded.read()))
            if data.ndim == 2:
                idx = min(window_index, data.shape[0]-1)
                signal = data[idx].astype(np.float32)
                signal_label = f"{uploaded.name} [window {idx}]"
                st.info(f"Loaded array shape: {data.shape} — showing window {idx}.")
            else:
                signal = data.astype(np.float32); signal_label = uploaded.name
    elif input_mode == "Upload .csv file":
        uploaded = st.file_uploader("Upload .csv ECG file", type=["csv"])
        if uploaded:
            arr = pd.read_csv(uploaded, header=None).values.astype(np.float32)
            if arr.ndim == 2 and arr.shape[0] > 1:
                idx = min(window_index, arr.shape[0]-1)
                signal = arr[idx]; signal_label = f"{uploaded.name} [row {idx}]"
                st.info(f"Loaded {arr.shape[0]} rows — showing row {idx}.")
            else:
                signal = arr.flatten(); signal_label = uploaded.name

    if signal is None:
        st.info("👈 Select an input source from the sidebar to get started.")
        return

    # ── PREPROCESS + FEATURES ────────────────────────────────────────────
    with st.spinner("Processing signal…"):
        proc     = preprocess(signal, fs=fs_input)
        peaks    = detect_rpeaks(signal, fs=fs_input)
        rr_ms    = np.diff(peaks) / fs_input * 1000
        rr_ms    = rr_ms[(rr_ms > 250) & (rr_ms < 2000)]
        features = extract_hrv(signal, fs=fs_input)

    # ── RUN SELECTED MODEL ────────────────────────────────────────────────
    label = prob = method_note = None
    imp_fig = None   # feature importance chart (XGB / CatBoost only)

    if model_choice == "HRV Heuristic (no weights needed)":
        label, prob, reasons = hrv_heuristic(features)
        method_note = "HRV heuristic"

    elif model_choice == "XGBoost":
        mdl = load_xgb_model(weights_path)
        if mdl is None:
            st.warning(f"XGBoost model not found at `{weights_path}`. Falling back to HRV heuristic.")
            label, prob, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (XGBoost weights missing)"
        else:
            label, prob = predict_xgb(mdl, features)
            reasons = None
            method_note = f"XGBoost — {weights_path}"
            imp_fig = plot_feature_importance_xgb(mdl)

    elif model_choice == "CatBoost":
        mdl = load_catboost_model(weights_path)
        if mdl is None:
            st.warning(f"CatBoost model not found at `{weights_path}`. Falling back to HRV heuristic.")
            label, prob, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (CatBoost weights missing)"
        else:
            label, prob = predict_catboost(mdl, features)
            reasons = None
            method_note = f"CatBoost — {weights_path}"
            imp_fig = plot_feature_importance_cb(mdl)

    elif model_choice in ("CNN","LSTM","RNN","CNN+LSTM"):
        mdl = load_deep_model(model_choice, weights_path)
        if mdl is None:
            st.warning(f"Weights not found at `{weights_path}`. Falling back to HRV heuristic.")
            label, prob, reasons = hrv_heuristic(features)
            method_note = f"HRV heuristic ({model_choice} weights missing)"
        else:
            label, prob = predict_deep(mdl, proc)
            reasons = None
            method_note = f"{model_choice} — {weights_path}"

    # ── RESULT CARD ──────────────────────────────────────────────────────
    col_res, col_info = st.columns([1.2, 1])
    with col_res:
        css_cls   = "result-afib" if label == "AFib" else "result-normal"
        icon      = "⚠️" if label == "AFib" else "✅"
        color     = "#fc8181" if label == "AFib" else "#68d391"
        bar_color = "#e53e3e" if label == "AFib" else "#38a169"
        st.markdown(f"""
        <div class="{css_cls}">
          <div style="font-size:0.85rem;color:{color};letter-spacing:.1em;text-transform:uppercase;font-weight:600">Prediction</div>
          <div class="result-title" style="color:{color}">{icon} {label}</div>
          <div style="font-size:.9rem;color:#a0aec0;margin:.3rem 0 .8rem">
            AFib probability: <b style="color:{color}">{prob*100:.1f}%</b>
          </div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
          </div>
          <div style="font-size:.78rem;color:#718096;margin-top:.6rem">Method: {method_note}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        feat = dict(zip(FEATURE_NAMES, features))
        st.markdown(f"**Signal:** {signal_label}")
        st.markdown(f"**Duration:** {len(signal)/fs_input:.1f} s &nbsp;|&nbsp; **Sample rate:** {fs_input} Hz")
        st.markdown(f"**R-peaks detected:** {len(peaks)} &nbsp;|&nbsp; **Valid RR intervals:** {len(rr_ms)}")
        if len(rr_ms):
            st.markdown(f"**Mean HR:** {feat['mean_hr']:.1f} bpm &nbsp;|&nbsp; **SDNN:** {feat['sdnn']:.1f} ms")
        st.markdown("---")
        if reasons:
            st.markdown("**Score breakdown (HRV heuristic)**")
            for k, v in reasons.items():
                bw  = int(v*100)
                clr = "#e53e3e" if v>0.6 else "#f6ad55" if v>0.3 else "#68d391"
                st.markdown(
                    f"<div style='font-size:.8rem;color:#a0aec0;margin:2px 0'>{k}"
                    f"<span style='float:right;color:{clr}'>{v*100:.0f}%</span></div>"
                    f"<div style='background:#1e2740;border-radius:4px;height:5px;margin-bottom:6px'>"
                    f"<div style='width:{bw}%;background:{clr};height:100%;border-radius:4px'></div></div>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────────
    tabs = st.tabs(["📈 ECG Signal","💓 RR Tachogram","🌀 Poincaré","📊 HRV Features","🌲 Feature Importance"])

    with tabs[0]:
        view_s  = min(10, len(signal)/fs_input)
        n_view  = int(view_s * fs_input)
        st.plotly_chart(plot_ecg(proc[:n_view], peaks[peaks<n_view], fs=fs_input,
            title=f"Preprocessed ECG — first {view_s:.0f}s"), use_container_width=True)

    with tabs[1]:
        if len(rr_ms) >= 3:
            st.plotly_chart(plot_rr(rr_ms), use_container_width=True)
        else:
            st.warning("Not enough RR intervals detected.")

    with tabs[2]:
        if len(rr_ms) >= 4:
            st.plotly_chart(plot_poincare(rr_ms), use_container_width=True)
        else:
            st.warning("Not enough RR intervals for Poincaré plot.")

    with tabs[3]:
        left, right = st.columns([1,1])
        with left:
            st.markdown("#### HRV Feature Values")
            rows = [{"Feature":n,"Value":f"{v:.4f}",
                     "Unit":FEATURE_UNITS.get(n,""),
                     "Description":FEATURE_DESCRIPTIONS.get(n,"")}
                    for n,v in zip(FEATURE_NAMES, features)]
            df_feat = pd.DataFrame(rows)
            st.dataframe(df_feat, use_container_width=True, height=420, hide_index=True)
        with right:
            st.plotly_chart(plot_radar(features), use_container_width=True)

    with tabs[4]:
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)
            st.caption("Importance is derived from the loaded model weights. "
                       "Higher = more influential in the AFib/Normal decision.")
        else:
            st.info("Feature importance is available when an **XGBoost** or **CatBoost** "
                    "model is loaded from the sidebar. Deep models (CNN/LSTM) and the "
                    "HRV heuristic do not produce per-feature importance scores here.")

    # ── DOWNLOAD ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button("⬇️ Download HRV features as CSV",
        data=df_feat.to_csv(index=False).encode(),
        file_name="hrv_features.csv", mime="text/csv")


if __name__ == "__main__":
    main()