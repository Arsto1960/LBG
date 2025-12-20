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

# --- CSS  ---
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

# st.title("💠 LBG Vector Quantization")
st.markdown("### 💠 LBG Vector Quantization")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Data Setup")
    n_samples = st.slider("Samples", 200, 2000, 1000, step=100)
    n_blobs = st.slider("Clusters", 1, 10, 4)
    cluster_std = st.slider("Dispersion", 0.1, 3.0, 0.8)
    
    if st.button("🔄 Generate New Data"):
        X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
        st.session_state['data'] = X
        st.session_state['codebook'] = np.array([X.mean(axis=0)]) # Reset to N=1
        st.session_state['history'] = []
        st.session_state['centroids_history'] = [] 
        st.session_state['stage'] = "Init"

    st.divider()
    st.header("2. Algorithm")
    target_size = st.selectbox("Target Size (N)", [2, 4, 8, 16, 32, 64, 128], index=3)
    epsilon = st.number_input("Splitting Noise (ε)", 0.001, 0.1, 0.02)
    show_traj = st.checkbox("Show Trajectories", value=True, help="Draw lines showing where centroids moved")

# --- Safe Initialization (The Fix) ---
# We check each key individually to handle hot-reloading gracefully
if 'data' not in st.session_state:
    X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
    st.session_state['data'] = X

if 'codebook' not in st.session_state:
    st.session_state['codebook'] = np.array([st.session_state['data'].mean(axis=0)])

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
    # 1. Assign
    dist_sq = np.sum((X[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    # 2. Update
    new_cb = []
    for i in range(len(cb)):
        pts = X[labels == i]
        if len(pts) > 0:
            new_cb.append(pts.mean(axis=0))
        else:
            # Re-initialize empty cell to a random data point
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
    
    # Custom HTML Metric
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">Codebook Size</div>
        <div class="big-stat">{len(cb)} / {target_size}</div>
        <hr style="margin: 10px 0; opacity: 0.2;">
        <div class="stat-label">MSE Distortion</div>
        <div class="big-stat" style="color: #dc3545;">{mse:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # Controls
    if len(cb) < target_size:
        if st.button("✂️ SPLIT", type="primary"):
            # Archive history for trajectory
            st.session_state['centroids_history'] = [st.session_state['codebook']] 
            
            st.session_state['codebook'] = split(st.session_state['codebook'], epsilon)
            st.session_state['stage'] = "Split"
            st.rerun()
            
        if st.button("▶️ OPTIMIZE"):
            st.session_state['centroids_history'] = [st.session_state['codebook']]
            
            progress = st.progress(0)
            for i in range(15): # 15 iterations
                old_cb = st.session_state['codebook']
                new_cb = step_kmeans(X, old_cb)
                
                # Save for trajectory
                st.session_state['centroids_history'].append(new_cb)
                st.session_state['codebook'] = new_cb
                
                # Update MSE history
                st.session_state['history'].append(get_distortion(X, new_cb))
                
                # Convergence check
                if np.allclose(old_cb, new_cb, atol=1e-4):
                    break
                    
                time.sleep(0.05) # Animation speed
                progress.progress((i+1)/15)
                
            st.session_state['stage'] = "Optimized"
            st.rerun()
    else:
        st.success("Target Reached!")
        if st.button("Reset"):
            st.session_state['codebook'] = np.array([X.mean(axis=0)])
            st.session_state['history'] = []
            st.session_state['centroids_history'] = []
            st.session_state['stage'] = "Init"
            st.rerun()

# Mini Distortion Plot
if len(st.session_state['history']) > 0:
    st.markdown("**Distortion Curve**")
    st.line_chart(st.session_state['history'], height=150)

colors = ['#d7191c', '#33a02c', '#ffff99', '#cab2d6', '#fdbf6f',
          '#a65628', '#1f78b4', '#a6cee3', '#1f78b4']
# --- Visualization ---
with col_main:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Data Density
    # ax.scatter(X[:, 0], X[:, 1], s=15, c='gray', alpha=0.4, label="Data")
    ax.scatter(X[:, 0], X[:, 1], c='k', s=2, zorder=5, label="Data")
    
    # 2. Trajectories (The path centroids took)
    if show_traj and len(st.session_state['centroids_history']) > 1:
        hist_arr = np.array(st.session_state['centroids_history'])
        # hist_arr shape: (Iterations, N_centroids, 2)
        # We iterate over centroids to draw lines
        for c_idx in range(len(cb)):
            # Get path for centroid 'c_idx' across all time steps
            # Check bounds to avoid crashing if splitting changed dimensions mid-history
            if c_idx < hist_arr.shape[1]: 
                path = hist_arr[:, c_idx, :]
                ax.plot(path[:, 0], path[:, 1], 'k--', linewidth=1, alpha=0.5)
                ax.scatter(path[0, 0], path[0, 1], c='gray', s=20, alpha=0.5)


    # 3. Voronoi Regions
    if len(cb) >= 2:
        try:
            vor = Voronoi(cb)
            voronoi_plot_2d(vor, ax=ax, 
                            show_vertices=False, 
                            line_colors='black',
                            line_width=0.5, 
                            line_alpha=0.6, 
                            point_size=0)

            # Color the regions
            for i, region in enumerate(vor.regions):
                if not -1 in region and len(region) > 0:
                    polygon = [vor.vertices[j] for j in region]
                    ax.fill(*zip(*polygon), color=colors[i % len(colors)], alpha=0.7)
        except:
            pass 
            
    # 4. Current Centroids
    ax.scatter(cb[:, 0], cb[:, 1], c='#ff0000', s=150, marker='+', linewidth=2.5, label="Centroids", zorder=10)
    # ax.scatter(cb[:, 0], centroids[:, 1], c='m', marker='x', s=100, linewidth=3, zorder=10)
    
    # Style
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
    
    # ax.set_title('K-means clustering on 2-Dimension Vector Space\nCentroids are marked with magenta crosses')
    # ax.set_xlim(-1, 1.1)
    # ax.set_ylim(-1, 1.3)

    # plt.show()





# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from scipy.spatial import Voronoi, voronoi_plot_2d
# import time

# # --- Page Config ---
# st.set_page_config(
#     page_title="Vector Quantization (LBG)",
#     page_icon="💠",
#     layout="wide"
# )

# # --- CSS for clean look ---
# st.markdown("""
# <style>
#     .stButton>button { width: 100%; }
#     .metric-box {
#         background-color: #f0f2f6;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 20px;
#         text-align: center;
#         border: 1px solid #e0e0e0;
#     }
#     .highlight { color: #ff4b4b; font-weight: bold; }
# </style>
# """, unsafe_allow_html=True)

# # --- Header ---
# # st.title("💠 The LBG Algorithm")
# st.markdown("### 💠 The LBG Algorithm")
# # --- Theory Expander ---
# with st.expander("📝 How does LBG work? (Click to read)"):
#     st.markdown("""
#     **Vector Quantization (VQ)** compresses data by grouping thousands of data points into a few "representative" points called **Codevectors**. The collection of these points is the **Codebook**.
    
#     **The LBG Recipe:**
#     1. **Start:** Calculate the average of ALL data (1 Codevector).
#     2. **Split:** Duplicate every codevector and nudge them slightly apart (Double the size: 1 → 2 → 4 → ...).
#     3. **Optimize (K-Means):** Move the codevectors to the center of their local neighborhood to reduce error.
#     4. **Repeat:** Keep splitting and optimizing until you reach the **Target Size**.
#     """)

# # --- Sidebar Controls ---
# with st.sidebar:
#     st.header("1. Data Generation")
#     n_samples = st.slider("Number of Data Points", 100, 1000, 300)
#     n_blobs = st.slider("Number of Natural Clusters", 1, 10, 3)
#     cluster_std = st.slider("Cluster Spread", 0.5, 3.0, 1.0)
    
#     if st.button("🔄 Generate New Data"):
#         X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
#         st.session_state['data'] = X
#         # Reset algorithm state
#         if 'codebook' in st.session_state:
#             del st.session_state['codebook']
#             st.session_state['stage'] = 'init'
#             st.session_state['history'] = []

#     st.divider()
#     st.header("2. LBG Settings")
#     target_size = st.selectbox(
#         "Target Compression Level (N)", 
#         [2, 4, 8, 16, 32, 64], 
#         index=2,
#         help="The final number of vectors allowed. Higher N = Less Distortion but Less Compression."
#     )
#     epsilon = st.number_input("Splitting Epsilon (ε)", 0.001, 0.1, 0.01, format="%.3f")

# # --- Initialize Data ---
# if 'data' not in st.session_state:
#     X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
#     st.session_state['data'] = X
# X = st.session_state['data']

# # --- Helper Functions ---
# def get_distortion(X, codebook):
#     dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
#     min_dist_sq = np.min(dist_sq, axis=1)
#     return np.mean(min_dist_sq)

# def run_kmeans_step(X, codebook):
#     dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
#     labels = np.argmin(dist_sq, axis=1)
#     new_codebook = []
#     for i in range(len(codebook)):
#         points = X[labels == i]
#         if len(points) > 0:
#             new_codebook.append(points.mean(axis=0))
#         else:
#             new_codebook.append(codebook[i]) 
#     return np.array(new_codebook)

# def split_codebook(codebook, epsilon):
#     new_cb = []
#     for vec in codebook:
#         vec_plus = vec * (1 + epsilon)
#         vec_minus = vec * (1 - epsilon)
#         new_cb.append(vec_plus)
#         new_cb.append(vec_minus)
#     return np.array(new_cb)

# # --- State Management ---
# if 'codebook' not in st.session_state:
#     st.session_state['codebook'] = np.array([X.mean(axis=0)])
#     st.session_state['stage'] = "Initialized (N=1)"

# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# # --- Main Layout ---
# col_plot, col_controls = st.columns([3, 1])

# # --- Logic & Controls ---
# with col_controls:
#     current_N = len(st.session_state['codebook'])
#     distortion = get_distortion(X, st.session_state['codebook'])
    
#     # Update History if changed
#     if not st.session_state['history'] or abs(st.session_state['history'][-1] - distortion) > 1e-6:
#         st.session_state['history'].append(distortion)

#     # Metrics Display
#     st.markdown(f"""
#     <div class="metric-box">
#         <h2>N = {current_N}</h2>
#         <small>Target: {target_size}</small><br><br>
#         <b>Distortion (MSE):</b><br>
#         <span style="font-size: 1.2em; color: #ff4b4b;">{distortion:.4f}</span>
#     </div>
#     """, unsafe_allow_html=True)

#     # --- ACTION BUTTONS ---
    
#     # 1. SPLIT LOGIC
#     if current_N < target_size:
#         if st.button("✂️ Double N", type="primary"):
#             st.session_state['codebook'] = split_codebook(st.session_state['codebook'], epsilon)
#             st.session_state['stage'] = "Split (Need Optimization)"
#             st.rerun()
            
#         st.write("---") # Visual Separator
        
#         # 2. OPTIMIZE LOGIC (Animation Loop)
#         if st.button("▶️ Run Optimization"):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Run 10 iterations of K-Means automatically
#             for i in range(10):
#                 # Update Logic
#                 st.session_state['codebook'] = run_kmeans_step(X, st.session_state['codebook'])
#                 curr_dist = get_distortion(X, st.session_state['codebook'])
#                 st.session_state['history'].append(curr_dist)
                
#                 # Update UI
#                 status_text.text(f"Optimizing... Iteration {i+1}/10")
#                 progress_bar.progress((i+1)/10)
                
#                 # Small sleep to allow visual "animation" feeling
#                 time.sleep(0.1) 
                
#             status_text.success("Optimization Converged!")
#             st.session_state['stage'] = "Optimized"
#             time.sleep(0.5)
#             st.rerun()

#     else:
#         st.success("✅ Target Reached!")
#         if st.button("Reset All"):
#             del st.session_state['codebook']
#             st.session_state['history'] = []
#             st.rerun()

# # --- Visualization ---
# with col_plot:
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     # 1. Plot Data
#     ax.scatter(X[:, 0], X[:, 1], s=15, c='gray', alpha=0.4, label="Data")
    
#     # 2. Plot Voronoi Regions
#     cb = st.session_state['codebook']
#     if len(cb) >= 2:
#         try:
#             vor = Voronoi(cb)
#             voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=0)
#         except Exception:
#             pass 
            
#     # 3. Plot Centroids
#     ax.scatter(cb[:, 0], cb[:, 1], c='red', s=120, marker='X', edgecolor='black', linewidth=1, label="Codevectors")

#     # 4. Distortion Plot (Small)
#     st.write("### Distortion")
#     st.line_chart(st.session_state['history'], height=150)
#     # st.caption("Lower is better. Jumps indicate Splits.")
    
#     # Visual Polish
#     ax.set_title(f"Visualizing LBG State: {st.session_state.get('stage', 'Init')}", fontsize=14)
#     ax.legend(loc="upper right")
#     ax.grid(True, alpha=0.2, linestyle="--")
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Embedding settings
#     fig.patch.set_alpha(0)
#     ax.patch.set_alpha(0)
#     st.pyplot(fig)



# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.datasets import make_blobs
# # from scipy.spatial import Voronoi, voronoi_plot_2d

# # # --- Page Config ---
# # st.set_page_config(
# #     page_title="Vector Quantization (LBG)",
# #     page_icon="💠",
# #     layout="wide"
# # )

# # # --- CSS for clean look ---
# # st.markdown("""
# # <style>
# #     .stButton>button { width: 100%; }
# #     .metric-box {
# #         background-color: #f0f2f6;
# #         padding: 10px;
# #         border-radius: 5px;
# #         margin-bottom: 10px;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # st.title("💠 The LBG Algorithm")
# # with st.expander("📝 How to use this App?"):
# #     st.markdown("""
# #     ### 🚀 How to use this App
# #     1. **Split & Optimize:** Use the **Split** button to double the red codevectors, then click **Optimize** to move them to the center of their data clusters.
# #     2. **Watch the Map:** Observe how the **Voronoi regions** (orange lines) carve the space and how **Distortion (MSE)** drops as the codebook grows.
# #     3. **Refine:** Continue the split/optimize cycle until you reach your target codebook size.
# #     """)


# # # --- Sidebar Controls ---
# # with st.sidebar:
# #     st.header("1. Data Generation")
# #     n_samples = st.slider("Number of Data Points", 100, 1000, 300)
# #     n_blobs = st.slider("Number of Natural Clusters", 1, 10, 3)
# #     cluster_std = st.slider("Cluster Spread", 0.5, 3.0, 1.0)
    
# #     if st.button("🔄 Generate New Data"):
# #         X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
# #         st.session_state['data'] = X
# #         # Reset algorithm state
# #         if 'codebook' in st.session_state:
# #             del st.session_state['codebook']
# #             st.session_state['stage'] = 'init'

# #     st.divider()
# #     st.header("2. LBG Settings")
# #     target_size = st.selectbox("Target Codebook Size", [2, 4, 8, 16, 32, 64], index=1)
# #     epsilon = st.number_input("Splitting Epsilon (ε)", 0.001, 0.1, 0.01, format="%.3f")

# # # --- Initialize Data in Session State ---
# # if 'data' not in st.session_state:
# #     X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
# #     st.session_state['data'] = X

# # X = st.session_state['data']

# # # --- Helper Functions ---

# # def get_distortion(X, codebook):
# #     """Calculate Mean Squared Error (Quantization Error)"""
# #     # Simple nearest neighbor search
# #     dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
# #     min_dist_sq = np.min(dist_sq, axis=1)
# #     return np.mean(min_dist_sq)

# # def run_kmeans_step(X, codebook):
# #     """One iteration of Lloyd's Algorithm (Assignment + Update)"""
# #     # 1. Assignment
# #     dist_sq = np.sum((X[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)
# #     labels = np.argmin(dist_sq, axis=1)
    
# #     # 2. Update (Centroids)
# #     new_codebook = []
# #     for i in range(len(codebook)):
# #         points = X[labels == i]
# #         if len(points) > 0:
# #             new_codebook.append(points.mean(axis=0))
# #         else:
# #             # Handle empty cell (keep old or re-randomize - keeping old for simplicity)
# #             new_codebook.append(codebook[i]) 
# #     return np.array(new_codebook)

# # def split_codebook(codebook, epsilon):
# #     """LBG Splitting: y -> y(1+e), y(1-e)"""
# #     new_cb = []
# #     for vec in codebook:
# #         vec_plus = vec * (1 + epsilon)
# #         vec_minus = vec * (1 - epsilon)
# #         new_cb.append(vec_plus)
# #         new_cb.append(vec_minus)
# #     return np.array(new_cb)

# # # --- Algorithm State Management ---
# # if 'codebook' not in st.session_state:
# #     # Initialize with global centroid (N=1)
# #     st.session_state['codebook'] = np.array([X.mean(axis=0)])
# #     st.session_state['stage'] = "Optimized (N=1)"

# # if 'history' not in st.session_state:
# #     st.session_state['history'] = []

# # # Update this every time the codebook changes
# # current_dist = get_distortion(X, st.session_state['codebook'])
# # if not st.session_state['history'] or st.session_state['history'][-1] != current_dist:
# #     st.session_state['history'].append(current_dist)

# # # --- Main Layout ---
# # col_plot, col_info = st.columns([2, 1])

# # with col_info:
# #     # st.subheader("Algorithm Control")
    
# #     current_N = len(st.session_state['codebook'])
# #     distortion = get_distortion(X, st.session_state['codebook'])
    
# #     # Metrics
# #     st.markdown(f"""
# #     <div class="metric-box">
# #         <b>Current Codebook Size:</b> {current_N} <br>
# #         <b>Distortion (MSE):</b> {distortion:.4f}
# #     </div>
# #     """, unsafe_allow_html=True)

# #     # State Machine Logic
# #     if current_N < target_size:
# #         st.info(f"Goal: Reach size {target_size}. Current: {current_N}")
        
# #         if st.button("✂️ SPLIT (Double Codebook)"):
# #             st.session_state['codebook'] = split_codebook(st.session_state['codebook'], epsilon)
# #             st.session_state['stage'] = "Just Split"
# #             st.rerun()
            
# #         if st.button("⚙️ Optimize (K-Means Iteration)"):
# #             st.session_state['codebook'] = run_kmeans_step(X, st.session_state['codebook'])
# #             st.session_state['stage'] = "Optimizing..."
# #             st.rerun()
# #     else:
# #         st.success("Target Codebook Size Reached!")
# #         if st.button("⚙️ Continue Optimizing"):
# #             st.session_state['codebook'] = run_kmeans_step(X, st.session_state['codebook'])
# #             st.rerun()
            
# #     if st.button("Reset Algorithm"):
# #         del st.session_state['codebook']
# #         st.session_state['stage'] = "Reset"
# #         st.rerun()

# # st.subheader("📉 Distortion Trend")
# # st.line_chart(st.session_state['history'])

# # with st.expander("📝 What is happening?"):
# #         st.write("""
# #         1. **Split:** Each red X (codevector) splits into two nearby vectors.
# #         2. **Optimize:** The vectors move to the center of the data points closest to them (Lloyd's Rule).
# #         3. **Repeat:** Until we have the desired number of quantization levels.
# #         """)

# # with col_plot:
# #     # --- Visualization ---
# #     fig, ax = plt.subplots(figsize=(8, 6))
    
# #     # 1. Plot Data
# #     ax.scatter(X[:, 0], X[:, 1], s=15, c='gray', alpha=0.5, label="Data Points")
    
# #     # 2. Plot Voronoi Regions (if N >= 2)
# #     cb = st.session_state['codebook']
# #     if len(cb) >= 2:
# #         try:
# #             vor = Voronoi(cb)
# #             voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=0)
# #         except Exception:
# #             pass # Voronoi fails if points are collinear or weird geometry, ignore for demo
            
# #     # 3. Plot Codebook
# #     ax.scatter(cb[:, 0], cb[:, 1], c='red', s=100, marker='x', linewidths=3, label="Codevectors")
    
# #     # Labels
# #     ax.set_title(f"LBG State: {st.session_state.get('stage', 'Init')} (N={len(cb)})")
# #     ax.legend()
# #     ax.grid(True, alpha=0.3)
# #     ax.set_xticks([])
# #     ax.set_yticks([])
    
# #     # Transparent background for embedding
# #     fig.patch.set_alpha(0)
# #     ax.patch.set_alpha(0)
    
# #     st.pyplot(fig)
