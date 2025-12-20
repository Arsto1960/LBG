import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

# --- Page Config ---
st.set_page_config(
    page_title="Vector Quantization (LBG)",
    page_icon="💠",
    layout="wide"
)

# --- CSS for clean look ---
st.markdown("""
<style>
    .stButton>button { width: 100%; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💠 Vector Quantization: The LBG Algorithm")
st.markdown("""
**Linde-Buzo-Gray (LBG)** is a clustering algorithm used to derive a **Codebook** for compressing data. 
It starts with 1 vector and recursively **splits** and **optimizes** (using K-Means) until the desired codebook size is reached.
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Data Generation")
    n_samples = st.slider("Number of Data Points", 100, 1000, 300)
    n_blobs = st.slider("Number of Natural Clusters", 1, 10, 3)
    cluster_std = st.slider("Cluster Spread", 0.5, 3.0, 1.0)
    
    if st.button("🔄 Generate New Data"):
        X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
        st.session_state['data'] = X
        # Reset algorithm state
        if 'codebook' in st.session_state:
            del st.session_state['codebook']
            st.session_state['stage'] = 'init'

    st.divider()
    st.header("2. LBG Settings")
    target_size = st.selectbox("Target Codebook Size", [2, 4, 8, 16, 32, 64], index=1)
    epsilon = st.number_input("Splitting Epsilon (ε)", 0.001, 0.1, 0.01, format="%.3f")

# --- Initialize Data in Session State ---
if 'data' not in st.session_state:
    X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
    st.session_state['data'] = X

X = st.session_state['data']

# --- Helper Functions ---

def get_distortion(X, codebook):
    """Calculate Mean Squared Error (Quantization Error)"""
    # Simple nearest neighbor search
    dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
    min_dist_sq = np.min(dist_sq, axis=1)
    return np.mean(min_dist_sq)

def run_kmeans_step(X, codebook):
    """One iteration of Lloyd's Algorithm (Assignment + Update)"""
    # 1. Assignment
    dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    
    # 2. Update (Centroids)
    new_codebook = []
    for i in range(len(codebook)):
        points = X[labels == i]
        if len(points) > 0:
            new_codebook.append(points.mean(axis=0))
        else:
            # Handle empty cell (keep old or re-randomize - keeping old for simplicity)
            new_codebook.append(codebook[i]) 
    return np.array(new_codebook)

def split_codebook(codebook, epsilon):
    """LBG Splitting: y -> y(1+e), y(1-e)"""
    new_cb = []
    for vec in codebook:
        vec_plus = vec * (1 + epsilon)
        vec_minus = vec * (1 - epsilon)
        new_cb.append(vec_plus)
        new_cb.append(vec_minus)
    return np.array(new_cb)

# --- Algorithm State Management ---
if 'codebook' not in st.session_state:
    # Initialize with global centroid (N=1)
    st.session_state['codebook'] = np.array([X.mean(axis=0)])
    st.session_state['stage'] = "Optimized (N=1)"

# --- Main Layout ---
col_plot, col_info = st.columns([2, 1])

with col_info:
    st.subheader("Algorithm Control")
    
    current_N = len(st.session_state['codebook'])
    distortion = get_distortion(X, st.session_state['codebook'])
    
    # Metrics
    st.markdown(f"""
    <div class="metric-box">
        <b>Current Codebook Size:</b> {current_N} <br>
        <b>Distortion (MSE):</b> {distortion:.4f}
    </div>
    """, unsafe_allow_html=True)

    # State Machine Logic
    if current_N < target_size:
        st.info(f"Goal: Reach size {target_size}. Current: {current_N}")
        
        if st.button("✂️ SPLIT (Double Codebook)"):
            st.session_state['codebook'] = split_codebook(st.session_state['codebook'], epsilon)
            st.session_state['stage'] = "Just Split"
            st.rerun()
            
        if st.button("⚙️ Optimize (K-Means Iteration)"):
            st.session_state['codebook'] = run_kmeans_step(X, st.session_state['codebook'])
            st.session_state['stage'] = "Optimizing..."
            st.rerun()
    else:
        st.success("Target Codebook Size Reached!")
        if st.button("⚙️ Continue Optimizing"):
            st.session_state['codebook'] = run_kmeans_step(X, st.session_state['codebook'])
            st.rerun()
            
    if st.button("Reset Algorithm"):
        del st.session_state['codebook']
        st.session_state['stage'] = "Reset"
        st.rerun()

    with st.expander("📝 What is happening?"):
        st.write("""
        1. **Split:** Each red X (codevector) splits into two nearby vectors.
        2. **Optimize:** The vectors move to the center of the data points closest to them (Lloyd's Rule).
        3. **Repeat:** Until we have the desired number of quantization levels.
        """)

with col_plot:
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Plot Data
    ax.scatter(X[:, 0], X[:, 1], s=15, c='gray', alpha=0.5, label="Data Points")
    
    # 2. Plot Voronoi Regions (if N >= 2)
    cb = st.session_state['codebook']
    if len(cb) >= 2:
        try:
            vor = Voronoi(cb)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=0)
        except Exception:
            pass # Voronoi fails if points are collinear or weird geometry, ignore for demo
            
    # 3. Plot Codebook
    ax.scatter(cb[:, 0], cb[:, 1], c='red', s=100, marker='x', linewidths=3, label="Codevectors")
    
    # Labels
    ax.set_title(f"LBG State: {st.session_state.get('stage', 'Init')} (N={len(cb)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Transparent background for embedding
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    st.pyplot(fig)
