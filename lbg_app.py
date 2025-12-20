import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d
import time

# --- Page Config ---
st.set_page_config(
    page_title="LBG Quantization Visualizer",
    page_icon="💠",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .big-stat { font-size: 24px; font-weight: bold; color: #007bff; }
    .stat-label { font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 💠 LBG Vector Quantization")
# --- Theory Expander ---
with st.expander("📝 How to use this App?"):
    st.markdown("""
    1. **Split & Optimize:** Use the **Double N** button to double the red codevectors, then click **Optimize** to move them to the center of their data clusters.
    2. Observe how the **Voronoi regions** carve the space and how **Distortion (MSE)** drops as the codebook grows.
    3. Continue the split/optimize cycle until you reach your target codebook size.
    """)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Data Setup")
    n_samples = st.slider("Samples", 200, 2000, 1000, step=100)
    n_blobs = st.slider("Clusters", 1, 10, 4)
    cluster_std = st.slider("Dispersion", 0.1, 3.0, 0.8)
    
    if st.button("🔄 Generate New Data"):
        X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
        st.session_state['data'] = X
        # Reset everything
        initial_cb = np.array([X.mean(axis=0)])
        st.session_state['codebook'] = initial_cb
        st.session_state['region_codebook'] = initial_cb # NEW: For coloring
        st.session_state['history'] = []
        st.session_state['centroids_history'] = [] 
        st.session_state['stage'] = "Init"

    st.divider()
    st.header("2. Algorithm")
    target_size = st.selectbox("Target Size (N)", [2, 4, 8, 16, 32, 64, 128], index=3)
    epsilon = st.number_input("Splitting Noise (ε)", 0.001, 0.1, 0.02)
    show_traj = st.checkbox("Show Trajectories", value=True)
    
    st.divider()
    # NEW: Color Map Selector
    cmap_choice = st.selectbox("Region Color Map", 
                               ['Pastel1', 'Set3', 'Paired', 'tab20', 'tab20c', 'Accent'])

# --- Initialization ---
if 'data' not in st.session_state:
    X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
    st.session_state['data'] = X

if 'codebook' not in st.session_state:
    initial_cb = np.array([st.session_state['data'].mean(axis=0)])
    st.session_state['codebook'] = initial_cb
    # Initialize region_codebook to match
    st.session_state['region_codebook'] = initial_cb

if 'region_codebook' not in st.session_state:
    # Fallback if key missing during hot-reload
    st.session_state['region_codebook'] = st.session_state['codebook']

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'centroids_history' not in st.session_state:
    st.session_state['centroids_history'] = []

if 'stage' not in st.session_state:
    st.session_state['stage'] = "Init"

X = st.session_state['data']

# --- Core Functions ---
def get_distortion(X, cb):
    dist_sq = np.sum((X[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
    return np.mean(np.min(dist_sq, axis=1))

def step_kmeans(X, cb):
    dist_sq = np.sum((X[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    new_cb = []
    for i in range(len(cb)):
        pts = X[labels == i]
        if len(pts) > 0:
            new_cb.append(pts.mean(axis=0))
        else:
            new_cb.append(X[np.random.choice(len(X))])
    return np.array(new_cb)

def split(cb, eps):
    new = []
    for v in cb:
        new.append(v * (1 + eps))
        new.append(v * (1 - eps))
    return np.array(new)

# --- Layout ---
col_main, col_side = st.columns([3, 1])

with col_side:
    cb = st.session_state['codebook']
    mse = get_distortion(X, cb)
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">Codebook Size</div>
        <div class="big-stat">{len(cb)} / {target_size}</div>
        <hr style="margin: 10px 0; opacity: 0.2;">
        <div class="stat-label">MSE Distortion</div>
        <div class="big-stat" style="color: #dc3545;">{mse:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") 

    if len(cb) < target_size:
        # if st.button("✂️ Double N", type="primary"):
        #     st.session_state['centroids_history'] = [st.session_state['codebook']] 
            
        #     # 1. Split the ACTUAL codebook
        #     st.session_state['codebook'] = split(st.session_state['codebook'], epsilon)
            
        #     # NOTE: We do NOT update 'region_codebook' here. 
        #     # This keeps the background regions frozen (showing the old N state)
        #     # while the red Xs (scatter plot) will show the new split state.
            
        #     st.session_state['stage'] = "Split"
        #     st.rerun()
            
        if st.button("✂️ Double N"):
            st.session_state['centroids_history'] = [st.session_state['codebook']] 
            
            # 1. Split the ACTUAL codebook
            st.session_state['codebook'] = split(st.session_state['codebook'], epsilon)
            
            # NOTE: We do NOT update 'region_codebook' here. 
            # This keeps the background regions frozen (showing the old N state)
            # while the red Xs (scatter plot) will show the new split state.
            
            st.session_state['stage'] = "Split"
            # st.rerun()


            
            st.session_state['centroids_history'] = [st.session_state['codebook']]
            progress = st.progress(0)
            
            # Run loop
            for i in range(10): 
                old_cb = st.session_state['codebook']
                new_cb = step_kmeans(X, old_cb)
                
                st.session_state['centroids_history'].append(new_cb)
                st.session_state['codebook'] = new_cb
                st.session_state['history'].append(get_distortion(X, new_cb))
                
                if np.allclose(old_cb, new_cb, atol=1e-4):
                    break
                    
                time.sleep(0.05)
                progress.progress((i+1)/10)
            
            # NOW we update the region map, because optimization is done.
            st.session_state['region_codebook'] = st.session_state['codebook']
            st.session_state['stage'] = "Optimized"
            st.rerun()
    else:
        st.success("Target Reached!")
        if st.button("Reset"):
            initial_cb = np.array([X.mean(axis=0)])
            st.session_state['codebook'] = initial_cb
            st.session_state['region_codebook'] = initial_cb
            st.session_state['history'] = []
            st.session_state['centroids_history'] = []
            st.session_state['stage'] = "Init"
            st.rerun()

if len(st.session_state['history']) > 0:
    st.markdown("**Distortion Curve**")
    st.line_chart(st.session_state['history'], height=150)

# --- Visualization ---
with col_main:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define bounds for the background map
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # --- 1. Background Coloring (Decision Boundaries) ---
    # We use 'region_codebook' which lags behind the actual 'codebook' during the split phase
    region_cb = st.session_state['region_codebook']
    
    if len(region_cb) >= 1:
        # Create a dense grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        
        # Find nearest centroid from the REGION codebook
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        dist_sq = np.sum((grid_points[:, np.newaxis, :] - region_cb[np.newaxis, :, :]) ** 2, axis=2)
        Z = np.argmin(dist_sq, axis=1)
        Z = Z.reshape(xx.shape)
        
        ax.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=plt.get_cmap(cmap_choice),
                  aspect='auto', origin='lower', alpha=0.5)

    # --- 2. Data Points ---
    # ax.scatter(X[:, 0], X[:, 1], s=15, c='k', alpha=0.3, label="Data")
    ax.scatter(X[:, 0], X[:, 1], c='k', s=2, zorder=5, label="Data")
    
    # --- 3. Trajectories ---
    if show_traj and len(st.session_state['centroids_history']) > 1:
        hist_arr = np.array(st.session_state['centroids_history'])
        # Only plot trajectories if dimensions match (safety check)
        if hist_arr.shape[1] == len(cb):
            for c_idx in range(len(cb)):
                path = hist_arr[:, c_idx, :]
                ax.plot(path[:, 0], path[:, 1], 'k--', linewidth=1, alpha=0.4)
                ax.scatter(path[0, 0], path[0, 1], c='gray', s=20, alpha=0.4)

    # --- 4. Current Centroids (The REAL ones) ---
    # We always plot the actual 'codebook' here, so you see the split happen 
    # even if the background color hasn't updated yet.
    ax.scatter(cb[:, 0], cb[:, 1], c='#ff0000', s=150, marker='X', linewidth=2.0, label="Centroids", zorder=10, edgecolors='white')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"State: {st.session_state['stage']} (N={len(cb)})", fontsize=14, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Clean Grid
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
    
    fig.patch.set_alpha(0) 
    ax.patch.set_alpha(0)
    st.pyplot(fig)
