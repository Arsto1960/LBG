import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
import time

# --- Page Config ---
st.set_page_config(
    page_title="LBG Vector Quantization // Ops Center",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 BLACK OPS THEME CONFIGURATION
# ==============================================================================
theme = {
    "bg": "#0e1117",           # Deep dark background
    "text": "#e0e0e0",         # Light grey text
    "accent": "#00d4ff",       # Cyan accent (Data)
    "highlight": "#ff00ff",    # Magenta highlight (Centroids)
    "plot_bg": "#161b22",      # Plot background
    "grid": "#303030"          # Grid lines
}

# --- CSS Injection ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {theme['bg']}; 
        color: {theme['text']};
    }}
    /* Sidebar Background */
    section[data-testid="stSidebar"] {{
        background-color: {theme['plot_bg']};
        border-right: 1px solid {theme['grid']};
    }}
    /* Metric Container */
    div[data-testid="metric-container"] {{
        border: 1px solid {theme['grid']};
        padding: 10px;
        border-radius: 5px;
        background-color: {theme['plot_bg']};
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Courier New', Courier, monospace;
        color: {theme['accent']};
        font-weight: bold;
    }}
    h1, h2, h3, p, label, .stMarkdown {{
        font-family: 'Helvetica', sans-serif;
        color: {theme['text']} !important;
    }}
    /* Button Styling */
    .stButton>button {{
        border-radius: 4px;
        border: 1px solid {theme['accent']};
        background-color: {theme['plot_bg']};
        color: {theme['accent']};
        font-family: 'Courier New';
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {theme['accent']};
        color: {theme['bg']};
        box-shadow: 0 0 10px {theme['accent']};
    }}
    /* Primary Button (Split) */
    button[kind="primary"] {{
        background-color: {theme['plot_bg']} !important;
        border: 1px solid {theme['highlight']} !important;
        color: {theme['highlight']} !important;
    }}
    button[kind="primary"]:hover {{
        background-color: {theme['highlight']} !important;
        color: {theme['bg']} !important;
        box-shadow: 0 0 10px {theme['highlight']};
    }}
</style>
""", unsafe_allow_html=True)

# --- Plot Styling Helper ---
def apply_theme_to_plot(fig, ax):
    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['plot_bg'])
    ax.tick_params(colors=theme['text'])
    ax.spines['bottom'].set_color(theme['text'])
    ax.spines['left'].set_color(theme['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False) # Turn off grid for cleaner Voronoi look
    
    # Update title/labels
    ax.xaxis.label.set_color(theme['text'])
    ax.yaxis.label.set_color(theme['text'])
    ax.title.set_color(theme['text'])

# --- State Management ---
if 'data' not in st.session_state:
    st.session_state.update({
        'data': None, 'codebook': None, 'region_codebook': None,
        'history': [], 'centroids_history': [], 'stage': "Init"
    })

# ==============================================================================
# 🎛️ CONTROLS (Sidebar)
# ==============================================================================
with st.sidebar:
    st.title("💠 LBG OPS")
    st.markdown("Vector Quantization Console")
    st.divider()

    st.markdown("### 1. Data Generation")
    n_samples = st.slider("Samples", 200, 2000, 1000, step=100)
    n_blobs = st.slider("Clusters", 1, 10, 4)
    cluster_std = st.slider("Dispersion", 0.1, 3.0, 0.8)
    
    if st.button("🔄 GENERATE DATA"):
        X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
        st.session_state['data'] = X
        # Reset
        initial_cb = np.array([X.mean(axis=0)])
        st.session_state['codebook'] = initial_cb
        st.session_state['region_codebook'] = initial_cb
        st.session_state['history'] = []
        st.session_state['centroids_history'] = [] 
        st.session_state['stage'] = "Init"
        st.rerun()

    st.divider()
    st.markdown("### 2. Algorithm")
    target_size = st.selectbox("Target Size (N)", [2, 4, 8, 16, 32, 64, 128], index=3)
    epsilon = st.number_input("Spl






# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from scipy.spatial import Voronoi, voronoi_plot_2d
# import time

# # --- Page Config ---
# st.set_page_config(
#     page_title="LBG Quantization Visualizer",
#     page_icon="💠",
#     layout="wide"
# )

# # --- CSS ---
# st.markdown("""
# <style>
#     .block-container {
#             padding-top: 1rem;
#             padding-bottom: 0rem;
#             padding-left: 1rem;
#             padding-right: 1rem;
#         }
#     .stButton>button { width: 100%; border-radius: 5px; }
#     .metric-container {
#         background-color: #f8f9fa;
#         border: 1px solid #e9ecef;
#         padding: 15px;
#         border-radius: 10px;
#         text-align: center;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#     }
#     .big-stat { font-size: 24px; font-weight: bold; color: #007bff; }
#     .stat-label { font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
# </style>
# """, unsafe_allow_html=True)

# st.subheader("💠 LBG Vector Quantization")
# # st.markdown("### 💠 LBG Vector Quantization")
# # --- Theory Expander ---
# with st.expander("📝 How to use this App?"):
#     st.markdown("""
#     1. **Split & Optimize:** Use the **Double N** button to double the red codevectors, then click **Optimize** to move them to the center of their data clusters.
#     2. Observe how the **Voronoi regions** carve the space and how **Distortion (MSE)** drops as the codebook grows.
#     3. Continue the split/optimize cycle until you reach your target codebook size.
#     """)

# # --- Sidebar Controls ---
# with st.sidebar:
#     st.header("1. Data Setup")
#     n_samples = st.slider("Samples", 200, 2000, 1000, step=100)
#     n_blobs = st.slider("Clusters", 1, 10, 4)
#     cluster_std = st.slider("Dispersion", 0.1, 3.0, 0.8)
    
#     if st.button("🔄 Generate New Data"):
#         X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=None)
#         st.session_state['data'] = X
#         # Reset everything
#         initial_cb = np.array([X.mean(axis=0)])
#         st.session_state['codebook'] = initial_cb
#         st.session_state['region_codebook'] = initial_cb # NEW: For coloring
#         st.session_state['history'] = []
#         st.session_state['centroids_history'] = [] 
#         st.session_state['stage'] = "Init"

#     st.divider()
#     st.header("2. Algorithm")
#     target_size = st.selectbox("Target Size (N)", [2, 4, 8, 16, 32, 64, 128], index=3)
#     epsilon = st.number_input("Splitting Noise (ε)", 0.001, 0.1, 0.02)
#     show_traj = st.checkbox("Show Trajectories", value=True)
    
#     st.divider()
#     # NEW: Color Map Selector
#     cmap_choice = st.selectbox("Region Color Map", 
#                                ['Accent', 'Pastel1', 'Set3', 'Paired', 'tab20', 'tab20c'])

# # --- Initialization ---
# if 'data' not in st.session_state:
#     X, y = make_blobs(n_samples=n_samples, centers=n_blobs, cluster_std=cluster_std, random_state=42)
#     st.session_state['data'] = X

# if 'codebook' not in st.session_state:
#     initial_cb = np.array([st.session_state['data'].mean(axis=0)])
#     st.session_state['codebook'] = initial_cb
#     # Initialize region_codebook to match
#     st.session_state['region_codebook'] = initial_cb

# if 'region_codebook' not in st.session_state:
#     # Fallback if key missing during hot-reload
#     st.session_state['region_codebook'] = st.session_state['codebook']

# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# if 'centroids_history' not in st.session_state:
#     st.session_state['centroids_history'] = []

# if 'stage' not in st.session_state:
#     st.session_state['stage'] = "Init"

# X = st.session_state['data']

# # --- Core Functions ---
# def get_distortion(X, cb):
#     dist_sq = np.sum((X[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
#     return np.mean(np.min(dist_sq, axis=1))

# def step_kmeans(X, cb):
#     dist_sq = np.sum((X[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
#     labels = np.argmin(dist_sq, axis=1)
#     new_cb = []
#     for i in range(len(cb)):
#         pts = X[labels == i]
#         if len(pts) > 0:
#             new_cb.append(pts.mean(axis=0))
#         else:
#             new_cb.append(X[np.random.choice(len(X))])
#     return np.array(new_cb)

# def split(cb, eps):
#     new = []
#     for v in cb:
#         new.append(v * (1 + eps))
#         new.append(v * (1 - eps))
#     return np.array(new)

# # --- Layout ---
# col_main, col_side = st.columns([3, 1])

# with col_side:
#     cb = st.session_state['codebook']
#     mse = get_distortion(X, cb)
    
#     st.markdown(f"""
#     <div class="metric-container">
#         <div class="stat-label">Codebook Size</div>
#         <div class="big-stat">{len(cb)} / {target_size}</div>
#         <hr style="margin: 10px 0; opacity: 0.2;">
#         <div class="stat-label">MSE Distortion</div>
#         <div class="big-stat" style="color: #dc3545;">{mse:.4f}</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.write("") 

#     if len(cb) < target_size:
#         # if st.button("✂️ Double N", type="primary"):
#         #     st.session_state['centroids_history'] = [st.session_state['codebook']] 
            
#         #     # 1. Split the ACTUAL codebook
#         #     st.session_state['codebook'] = split(st.session_state['codebook'], epsilon)
            
#         #     # NOTE: We do NOT update 'region_codebook' here. 
#         #     # This keeps the background regions frozen (showing the old N state)
#         #     # while the red Xs (scatter plot) will show the new split state.
            
#         #     st.session_state['stage'] = "Split"
#         #     st.rerun()
            
#         if st.button("✂️ Double N"):
#             st.session_state['centroids_history'] = [st.session_state['codebook']] 
            
#             # 1. Split the ACTUAL codebook
#             st.session_state['codebook'] = split(st.session_state['codebook'], epsilon)
            
#             # NOTE: We do NOT update 'region_codebook' here. 
#             # This keeps the background regions frozen (showing the old N state)
#             # while the red Xs (scatter plot) will show the new split state.
            
#             st.session_state['stage'] = "Split"
#             # st.rerun()


            
#             st.session_state['centroids_history'] = [st.session_state['codebook']]
#             progress = st.progress(0)
            
#             # Run loop
#             for i in range(10): 
#                 old_cb = st.session_state['codebook']
#                 new_cb = step_kmeans(X, old_cb)
                
#                 st.session_state['centroids_history'].append(new_cb)
#                 st.session_state['codebook'] = new_cb
#                 st.session_state['history'].append(get_distortion(X, new_cb))
                
#                 if np.allclose(old_cb, new_cb, atol=1e-4):
#                     break
                    
#                 time.sleep(0.05)
#                 progress.progress((i+1)/10)
            
#             # NOW we update the region map, because optimization is done.
#             st.session_state['region_codebook'] = st.session_state['codebook']
#             st.session_state['stage'] = "Optimized"
#             st.rerun()
#     else:
#         st.success("Target Reached!")
#         if st.button("Reset"):
#             initial_cb = np.array([X.mean(axis=0)])
#             st.session_state['codebook'] = initial_cb
#             st.session_state['region_codebook'] = initial_cb
#             st.session_state['history'] = []
#             st.session_state['centroids_history'] = []
#             st.session_state['stage'] = "Init"
#             st.rerun()

# if len(st.session_state['history']) > 0:
#     st.markdown("**Distortion Curve**")
#     st.line_chart(st.session_state['history'], height=150)

# # # --- Visualization ---
# # with col_main:
# #     fig, ax = plt.subplots(figsize=(10, 7))
    
# #     # Define bounds for the background map
# #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
# #     # --- 1. Background Coloring (Decision Boundaries) ---
# #     # We use 'region_codebook' which lags behind the actual 'codebook' during the split phase
# #     region_cb = st.session_state['region_codebook']
    
# #     if len(region_cb) >= 1:
# #         # Create a dense grid
# #         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
# #                              np.arange(y_min, y_max, 0.05))
        
# #         # Find nearest centroid from the REGION codebook
# #         grid_points = np.c_[xx.ravel(), yy.ravel()]
# #         dist_sq = np.sum((grid_points[:, np.newaxis, :] - region_cb[np.newaxis, :, :]) ** 2, axis=2)
# #         Z = np.argmin(dist_sq, axis=1)
# #         Z = Z.reshape(xx.shape)
        
# #         ax.imshow(Z, interpolation='nearest',
# #                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
# #                   cmap=plt.get_cmap(cmap_choice),
# #                   aspect='auto', origin='lower', alpha=0.5)

# #     # --- 2. Data Points ---
# #     # ax.scatter(X[:, 0], X[:, 1], s=15, c='k', alpha=0.3, label="Data")
# #     ax.scatter(X[:, 0], X[:, 1], c='k', s=2, zorder=5, label="Data")
    
# #     # --- 3. Trajectories ---
# #     if show_traj and len(st.session_state['centroids_history']) > 1:
# #         hist_arr = np.array(st.session_state['centroids_history'])
# #         # Only plot trajectories if dimensions match (safety check)
# #         if hist_arr.shape[1] == len(cb):
# #             for c_idx in range(len(cb)):
# #                 path = hist_arr[:, c_idx, :]
# #                 ax.plot(path[:, 0], path[:, 1], 'k--', linewidth=1, alpha=0.4)
# #                 ax.scatter(path[0, 0], path[0, 1], c='gray', s=20, alpha=0.4)

# #     # --- 4. Current Centroids (The REAL ones) ---
# #     # We always plot the actual 'codebook' here, so you see the split happen 
# #     # even if the background color hasn't updated yet.
# #     ax.scatter(cb[:, 0], cb[:, 1], c='#ff0000', s=150, marker='X', linewidth=2.0, label="Centroids", zorder=10, edgecolors='white')
    
# #     ax.set_xlim(x_min, x_max)
# #     ax.set_ylim(y_min, y_max)
# #     ax.set_title(f"State: {st.session_state['stage']} (N={len(cb)})", fontsize=14, pad=10)
# #     ax.legend(loc="upper right", frameon=True, framealpha=0.9)
# #     ax.set_xticks([])
# #     ax.set_yticks([])
    
# #     # Clean Grid
# #     ax.grid(False)
# #     for spine in ax.spines.values():
# #         spine.set_visible(True)
# #         spine.set_color('#dddddd')
    
# #     fig.patch.set_alpha(0) 
# #     ax.patch.set_alpha(0)
# #     st.pyplot(fig)


# with col_main:
#     fig, ax = plt.subplots(figsize=(10, 7))
    
#     # Define bounds for the background map
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
#     # --- 1. Background Coloring (Decision Boundaries) ---
#     region_cb = st.session_state['region_codebook']
#     N_regions = len(region_cb)
    
#     if N_regions >= 1:
#         # Create a dense grid
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
#                              np.arange(y_min, y_max, 0.05))
        
#         # Find nearest centroid from the REGION codebook
#         grid_points = np.c_[xx.ravel(), yy.ravel()]
#         dist_sq = np.sum((grid_points[:, np.newaxis, :] - region_cb[np.newaxis, :, :]) ** 2, axis=2)
#         Z = np.argmin(dist_sq, axis=1)
#         Z = Z.reshape(xx.shape)
        
#         # --- COLOR FIX START ---
#         from matplotlib.colors import ListedColormap
#         import matplotlib.cm as cm
        
#         # 1. Get the base colormap object
#         try:
#             base_cmap = cm.get_cmap(cmap_choice)
#         except:
#             base_cmap = cm.get_cmap('tab20') # Fallback
            
#         # 2. Extract standard colors from the map
#         if hasattr(base_cmap, 'colors'):
#             # For discrete maps like 'Set3', 'Pastel1'
#             base_colors = base_cmap.colors 
#         else:
#             # For continuous maps like 'viridis' or 'nipy_spectral'
#             base_colors = base_cmap(np.linspace(0, 1, 256))
            
#         # 3. Create a new list that cycles through base colors enough times to cover N_regions
#         # This ensures we never run out of colors, even if N=128
#         extended_colors = [base_colors[i % len(base_colors)] for i in range(N_regions)]
        
#         # 4. Shuffle them deterministically so adjacent regions (often close in index) 
#         # don't get similar colors if the cycle is short.
#         # (Optional, but helps contrast)
#         np.random.seed(42) 
#         np.random.shuffle(extended_colors)
        
#         custom_cmap = ListedColormap(extended_colors)
        
#         # 5. Plot with explicit vmin/vmax to map indices 0..N-1 exactly to our N colors
#         ax.imshow(Z, interpolation='nearest',
#                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                   cmap=custom_cmap,
#                   vmin=0, vmax=N_regions-1,
#                   aspect='auto', origin='lower', alpha=0.5)
#         # --- COLOR FIX END ---

#     # --- 2. Data Points ---
#     # ax.scatter(X[:, 0], X[:, 1], s=15, c='k', alpha=0.3, label="Data")
#     ax.scatter(X[:, 0], X[:, 1], c='k', s=2, zorder=5, label="Data")
    
#     # --- 3. Trajectories ---
#     if show_traj and len(st.session_state['centroids_history']) > 1:
#         hist_arr = np.array(st.session_state['centroids_history'])
#         if hist_arr.shape[1] == len(cb):
#             for c_idx in range(len(cb)):
#                 path = hist_arr[:, c_idx, :]
#                 ax.plot(path[:, 0], path[:, 1], 'k--', linewidth=1, alpha=0.4)
#                 ax.scatter(path[0, 0], path[0, 1], c='gray', s=20, alpha=0.4)

#     # --- 4. Current Centroids ---
#     ax.scatter(cb[:, 0], cb[:, 1], c='#ff0000', s=150, marker='X', linewidth=2.0, label="Centroids", zorder=10, edgecolors='white')
    
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_title(f"State: {st.session_state['stage']} (N={len(cb)})", fontsize=14, pad=10)
#     ax.legend(loc="upper right", frameon=True, framealpha=0.9)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     ax.grid(False)
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color('#dddddd')
    
#     fig.patch.set_alpha(0) 
#     ax.patch.set_alpha(0)
    
#     st.pyplot(fig)
