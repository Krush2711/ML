import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="Linear Regression Viz",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title & Intro ---
st.title("Interactive Linear Regression Visualization")
st.markdown("""
This tool helps you visualize how **Linear Regression** finds the best fit line by minimizing the **Mean Squared Error (MSE)**.
Adjust the slope ($m$) and intercept ($b$) to see how the line changes in 2D and how the error changes on the 3D Cost Surface.
""")

# --- Sidebar: Data Generation ---
st.sidebar.header("1. Data Generation")
n_points = st.sidebar.slider("Number of Points", 10, 200, 50)
noise_level = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.0)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Initialize or update data in session state
if 'data' not in st.session_state or st.sidebar.button("Generate New Data"):
    np.random.seed(seed)
    X = np.random.rand(n_points) * 10  # X values 0 to 10
    # True underlying relationship: y = 2x + 1 + noise
    true_m, true_b = 2.0, 1.0
    Y = true_m * X + true_b + np.random.randn(n_points) * noise_level
    st.session_state.data = {'X': X, 'Y': Y}

X = st.session_state.data['X']
Y = st.session_state.data['Y']

# --- Sidebar: Model Parameters ---
st.sidebar.divider()
st.sidebar.header("2. Model Parameters")
m = st.sidebar.slider("Slope (m)", -5.0, 10.0, 1.0, step=0.1)
b = st.sidebar.slider("Intercept (b)", -5.0, 10.0, 0.0, step=0.1)

# --- Core Calculations ---
Y_pred = m * X + b
mse = np.mean((Y - Y_pred) ** 2)

# Col layout
col1, col2 = st.columns([1, 1])

# --- 2D Visualization (Left Column) ---
with col1:
    st.subheader("2D Space: Data & Regression Line")
    
    # Create scatter plot of data
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(
        x=X, y=Y, 
        mode='markers', 
        name='Data Points',
        marker=dict(color='#3498db', opacity=0.7)
    ))
    
    # Create regression line
    # Generate line points for smoother drawing (limits of X)
    x_range = np.linspace(0, 10, 100)
    y_range = m * x_range + b
    
    fig_2d.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode='lines',
        name=f'Fit: y = {m}x + {b}',
        line=dict(color='#e74c3c', width=3)
    ))

    fig_2d.update_layout(
        title=f"Current MSE: {mse:.4f}",
        xaxis_title="X (Input)",
        yaxis_title="Y (Target)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# --- 3D Visualization (Right Column) ---
with col2:
    st.subheader("3D Space: Cost Function Surface")
    
    # Generate Grid for Surface Plot
    m_grid = np.linspace(-5, 10, 40)
    b_grid = np.linspace(-5, 10, 40)
    M_mesh, B_mesh = np.meshgrid(m_grid, b_grid)
    
    # Vectorized MSE calculation for the grid
    # Z (Cost) = mean((Y - (m*X + b))^2)
    # This is a bit heavy, so we do it carefully
    # Z[i,j] is the MSE for M_mesh[i,j] and B_mesh[i,j]
    Z_mesh = np.zeros_like(M_mesh)
    
    for i in range(M_mesh.shape[0]):
        for j in range(M_mesh.shape[1]):
            m_val = M_mesh[i, j]
            b_val = B_mesh[i, j]
            y_p = m_val * X + b_val
            Z_mesh[i, j] = np.mean((Y - y_p) ** 2)
            
    fig_3d = go.Figure(data=[go.Surface(
        z=Z_mesh, 
        x=M_mesh, 
        y=B_mesh, 
        colorscale='Viridis', 
        opacity=0.8,
        name='Cost Surface'
    )])
    
    # Add the current point (m, b, mse)
    fig_3d.add_trace(go.Scatter3d(
        x=[m], y=[b], z=[mse],
        mode='markers',
        name='Current Model',
        marker=dict(size=10, color='red', symbol='circle')
    ))

    fig_3d.update_layout(
        title="Cost Function J(m, b)",
        scene=dict(
            xaxis_title='Slope (m)',
            yaxis_title='Intercept (b)',
            zaxis_title='Cost (MSE)'
        ),
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- Explanation Section ---
st.divider()
st.markdown(f"""
### What is happening?
1.  **The Goal**: We want to find the line $y = mx + b$ that is "closest" to all the blue dots.
2.  **The Cost**: We measure "closeness" using **Mean Squared Error (MSE)**. This is the average of the squared distances (vertical lines) between the blue dots and the red line.
3.  **The 3D Surface**: The surface plot on the right shows the MSE for *possible combination* of $m$ and $b$. 
    - The "valley" or "bowl" bottom is where the error is lowest.
    - Your goal is to move the **Red Dot** into the deepest part of the bowl by adjusting the sliders.
    
**Current Equation**: $y = {m:.2f}x + {b:.2f}$  
**Current Error**: ${mse:.4f}$
""")
