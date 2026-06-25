"""
Streamlit Dashboard for ISRO PS14 Radiation Forecasting.

Interactive visualization of:
- Data exploration
- Model predictions
- Performance metrics
- GRASP comparison

Run with: streamlit run app/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st

# Add project root to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.models.transformer_model import TransformerForecaster
from src.models.lstm_model import LSTMForecaster
from src.features.feature_engineer import FeatureEngineer
from src.evaluation.metrics import ForecastMetrics

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="ISRO PS14 - Radiation Environment Forecaster",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 50%, #0d1117 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Load Data & Models
# =============================================================================

@st.cache_data
def load_processed_data():
    """Load the processed dataset."""
    config = Config()
    cache_path = config.data.get('cache_file', 'data/processed/merged_dataset.parquet')
    
    if Path(cache_path).exists():
        df = pd.read_parquet(cache_path)
        return df
    return None


@st.cache_resource
def load_model(model_type: str = 'transformer'):
    """Load a trained model."""
    config = Config()
    checkpoint_dir = Path(config.training.get('checkpoint_dir', 'models/checkpoints'))
    model_path = checkpoint_dir / f'best_{model_type}.pt'
    
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine num_features from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    if model_type == 'transformer':
        # Get num_features from input projection layer
        input_weight_key = [k for k in state_dict.keys() if 'input_proj.0.weight' in k]
        if input_weight_key:
            num_features = state_dict[input_weight_key[0]].shape[1]
        else:
            num_features = 29  # fallback
        
        model = TransformerForecaster.from_config(config._config, num_features)
    else:
        input_weight_key = [k for k in state_dict.keys() if 'input_proj.weight' in k]
        if input_weight_key:
            num_features = state_dict[input_weight_key[0]].shape[1]
        else:
            num_features = 29
        
        model = LSTMForecaster.from_config(config._config, num_features)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, checkpoint.get('metrics', {})


def load_predictions():
    """Load saved test predictions if available."""
    pred_dir = PROJECT_ROOT / 'outputs' / 'predictions'
    
    results = {}
    for model_type in ['transformer', 'lstm']:
        pred_file = pred_dir / f'{model_type}_test_predictions.npz'
        if pred_file.exists():
            data = np.load(pred_file)
            results[model_type] = {
                'predictions': data['predictions'],
                'targets': data['targets'],
                'timestamps': data.get('timestamps', None),
                'p5': data.get('p5', None),
                'p95': data.get('p95', None),
            }
            
            # Load importance if available
            imp_file = pred_dir / f'{model_type}_importance.json'
            if imp_file.exists():
                import json
                with open(imp_file, 'r') as f:
                    results[model_type]['importance'] = json.load(f)
                    
    return results


# =============================================================================
# Dashboard Pages
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ ISRO PS14 — Radiation Environment Forecaster</h1>
        <p>AI-powered prediction of >2 MeV electron fluxes at geostationary orbit</p>
    </div>
    """, unsafe_allow_html=True)


def page_data_explorer():
    """Data exploration page."""
    st.header("📊 Data Explorer")
    
    df = load_processed_data()
    
    if df is None:
        st.warning("⚠️ No processed data found. Run the preprocessing pipeline first:")
        st.code("python main.py --mode preprocess")
        return
    
    st.success(f"✅ Loaded {len(df):,} records from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df.index[0].date())
    with col2:
        end_date = st.date_input("End Date", df.index[-1].date())
    
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    df_view = df[mask]
    
    if len(df_view) == 0:
        st.warning("No data in selected range")
        return
    
    # Multi-panel plot
    flux_col = 'electron_flux_gt2MeV'
    proton_col = 'grasp_proton_flux'
    
    if flux_col in df_view.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Electron Flux (>2 MeV)', 'Proton Flux (>10 MeV)'),
            row_heights=[0.5, 0.5],
        )
        
        # Downsample for performance
        step = max(1, len(df_view) // 5000)
        df_plot = df_view.iloc[::step]
        
        # Electron flux
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot[flux_col],
                      mode='lines', name='Electron Flux',
                      line=dict(color='#2196F3', width=0.8)),
            row=1, col=1
        )
        fig.update_yaxes(type='log', title='Flux (pfu)', row=1, col=1)
        
        # Proton flux
        if proton_col in df_plot.columns:
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot[proton_col],
                          mode='lines', name='Proton Flux',
                          line=dict(color='#E91E63', width=0.8)),
                row=2, col=1
            )
            fig.update_yaxes(type='log', title='Flux (pfu)', row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data summary
    st.subheader("📋 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df_view):,}")
    with col2:
        valid_pct = df_view.notna().mean().mean() * 100
        st.metric("Data Validity", f"{valid_pct:.1f}%")
    with col3:
        duration = (df_view.index[-1] - df_view.index[0]).days
        st.metric("Duration", f"{duration} days")
    with col4:
        st.metric("Features", len(df_view.columns))
    
    # Column details
    with st.expander("📊 Column Statistics"):
        st.dataframe(df_view.describe().T.style.format("{:.3f}"))


def page_predictions():
    """Model predictions page."""
    st.header("🔮 Model Predictions")
    
    predictions = load_predictions()
    
    if not predictions:
        st.info("No saved predictions found. Showing model loading status...")
        
        for model_type in ['transformer', 'lstm']:
            model, metrics = load_model(model_type)
            if model is not None:
                st.success(f"✅ {model_type.capitalize()} model loaded")
                if metrics:
                    st.json(metrics)
            else:
                st.warning(f"⚠️ {model_type.capitalize()} model not found. Train it first.")
        
        st.code("python main.py --mode train")
        return
    
    # Model selector
    model_name = st.selectbox("Select Model", list(predictions.keys()))
    
    if model_name not in predictions:
        return
    
    pred_data = predictions[model_name]
    preds = pred_data['predictions']
    targets = pred_data['targets']
    
    horizon_labels = ['30min', '6h', '12h']
    
    # Horizon selector
    horizon_idx = st.radio(
        "Forecast Horizon",
        range(min(3, preds.shape[1])),
        format_func=lambda x: horizon_labels[x] if x < len(horizon_labels) else f'H{x}',
        horizontal=True
    )
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    true_h = targets[:, horizon_idx]
    pred_h = preds[:, horizon_idx]
    mask = np.isfinite(true_h) & np.isfinite(pred_h)
    
    pe = ForecastMetrics.prediction_efficiency(true_h[mask], pred_h[mask])
    rmse = ForecastMetrics.rmse_log(true_h[mask], pred_h[mask])
    r = ForecastMetrics.pearson_r(true_h[mask], pred_h[mask])
    mae = ForecastMetrics.mae_log(true_h[mask], pred_h[mask])
    
    col1.metric("Prediction Efficiency", f"{pe:.4f}", delta="Target: 0.80")
    col2.metric("RMSE (log₁₀)", f"{rmse:.4f}", delta="Target: <0.30")
    col3.metric("Pearson R", f"{r:.4f}", delta="Target: >0.90")
    col4.metric("MAE (log₁₀)", f"{mae:.4f}")
    
    # Time series plot
    n_points = min(5000, len(true_h))
    step = max(1, len(true_h) // n_points)
    
    fig = go.Figure()
    
    # Add uncertainty bands if available
    has_uncertainty = pred_data.get('p5') is not None and pred_data.get('p95') is not None
    if has_uncertainty and np.any(pred_data['p5']):
        p5_h = pred_data['p5'][:, horizon_idx]
        p95_h = pred_data['p95'][:, horizon_idx]
        
        # Upper bound
        fig.add_trace(go.Scatter(
            y=p95_h[::step], mode='lines',
            line=dict(width=0), showlegend=False,
            hoverinfo='skip'
        ))
        # Lower bound
        fig.add_trace(go.Scatter(
            y=p5_h[::step], mode='lines',
            fill='tonexty', fillcolor='rgba(255, 87, 34, 0.2)',
            line=dict(width=0), name='90% Confidence Interval'
        ))

    fig.add_trace(go.Scatter(
        y=true_h[::step], mode='lines', name='Observed',
        line=dict(color='#2196F3', width=1)
    ))
    fig.add_trace(go.Scatter(
        y=pred_h[::step], mode='lines', name='Predicted (Mean)',
        line=dict(color='#FF5722', width=1)
    ))
    
    fig.update_layout(
        title=f'{model_name.capitalize()} — {horizon_labels[horizon_idx]} Forecast',
        xaxis_title='Sample Index',
        yaxis_title='log₁₀(Electron Flux)',
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scattergl(
        x=true_h[mask], y=pred_h[mask],
        mode='markers', name='Predictions',
        marker=dict(size=2, color='#FF5722', opacity=0.3)
    ))
    
    # 1:1 line
    lims = [min(true_h[mask].min(), pred_h[mask].min()),
            max(true_h[mask].max(), pred_h[mask].max())]
    fig_scatter.add_trace(go.Scatter(
        x=lims, y=lims, mode='lines', name='1:1',
        line=dict(color='white', dash='dash', width=1)
    ))
    
    fig_scatter.update_layout(
        title='Predicted vs Observed',
        xaxis_title='Observed log₁₀(Flux)',
        yaxis_title='Predicted log₁₀(Flux)',
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)


def page_performance():
    """Model performance comparison page."""
    st.header("📈 Model Performance")
    
    predictions = load_predictions()
    
    if not predictions:
        st.warning("No predictions available. Train models first.")
        return
    
    horizon_labels = ['30min', '6h', '12h']
    
    # Compute metrics for all models
    all_metrics = {}
    for model_name, pred_data in predictions.items():
        metrics = ForecastMetrics.compute_multi_horizon(
            pred_data['targets'],
            pred_data['predictions'],
            horizon_labels[:pred_data['targets'].shape[1]]
        )
        all_metrics[model_name] = metrics
    
    # Comparison table
    st.subheader("📊 Metrics Comparison")
    
    rows = []
    for model, horizons in all_metrics.items():
        for horizon, m in horizons.items():
            rows.append({
                'Model': model.capitalize(),
                'Horizon': horizon,
                'PE': m.get('PE', 0),
                'RMSE(log)': m.get('RMSE_log', 0),
                'Pearson R': m.get('Pearson_R', 0),
                'MAE(log)': m.get('MAE_log', 0),
                'HSS': m.get('HSS', 0),
            })
    
    metrics_df = pd.DataFrame(rows)
    
    st.dataframe(
        metrics_df.style
        .background_gradient(subset=['PE'], cmap='RdYlGn', vmin=0, vmax=1)
        .background_gradient(subset=['Pearson R'], cmap='RdYlGn', vmin=0, vmax=1)
        .background_gradient(subset=['RMSE(log)'], cmap='RdYlGn_r', vmin=0, vmax=1)
        .format("{:.4f}", subset=['PE', 'RMSE(log)', 'Pearson R', 'MAE(log)', 'HSS']),
        use_container_width=True
    )
    
    # Bar chart comparison
    metric_to_plot = st.selectbox("Select Metric", ['PE', 'RMSE(log)', 'Pearson R', 'MAE(log)'])
    
    fig = px.bar(
        metrics_df,
        x='Horizon', y=metric_to_plot,
        color='Model',
        barmode='group',
        template='plotly_dark',
        title=f'{metric_to_plot} by Model and Forecast Horizon',
        color_discrete_map={
            'Transformer': '#FF5722',
            'Lstm': '#4CAF50',
        }
    )
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("🧠 Explainable AI: Feature Importance")
    
    has_importance = False
    for model_name, pred_data in predictions.items():
        if 'importance' in pred_data and pred_data['importance']:
            has_importance = True
            imp = pred_data['importance']
            
            # Sort by value
            imp_df = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importance (%)'])
            imp_df = imp_df.sort_values('Importance (%)', ascending=True)
            
            fig_imp = px.bar(
                imp_df, x='Importance (%)', y='Feature', orientation='h',
                title=f'{model_name.capitalize()} — Permutation Feature Importance',
                template='plotly_dark',
                color='Importance (%)',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(
                height=max(400, len(imp_df)*30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
    if not has_importance:
        st.info("No feature importance data available for the models.")


def page_about():
    """About page with project information."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ### ISRO Problem Statement 14
    **Forecasting Energetic Particle Radiation Environment for Geostationary Satellites**
    
    This system predicts >2 MeV electron fluxes at geostationary orbit using
    deep learning models trained on ISRO GRASP spacecraft data.
    
    #### 🔬 Data Sources
    - **GRASP/GSAT-19**: ISRO electron and proton flux at Indian longitude (PRADAN)
    
    #### 🧠 Models
    - **Transformer** (Primary): Informer-style with 3 encoder layers, 4 attention heads
    - **LSTM** (Baseline): Bidirectional with attention, 2 layers
    
    #### 📊 Features
    Time series forecasting relying on recent history (rolling statistics, derivatives) 
    and temporal features (diurnal and seasonal variations).
    
    #### 🎯 Forecast Horizons
    - **30 minutes**: Short-term alert for immediate satellite protection
    - **6 hours**: Medium-term planning window
    - **12 hours**: Extended forecast for operational scheduling
    
    ---
    *Built for India's national space weather competition*
    """)


def page_live_operations():
    """Render the real-time operational dashboard."""
    st.header("📡 Real-Time Space Weather Operations")
    st.write("Live inference engine monitoring >2 MeV electron fluxes at geostationary orbit using NOAA SWPC data streams.")
    
    # Auto-refresh check
    st.markdown("*(To enable auto-refresh, you can run the dashboard with `streamlit-autorefresh` installed, though this page loads the latest state upon manual refresh)*")
    
    state_file = PROJECT_ROOT / 'outputs' / 'live' / 'live_state.json'
    
    if not state_file.exists():
        st.warning("Live state file not found. Ensure `src/models/live_inference.py` is running in the background.")
        return
        
    import json
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
    except Exception as e:
        st.error(f"Error loading live state: {e}")
        return
        
    col1, col2, col3 = st.columns(3)
    
    status_color = "🟢" if state['status'] == "Normal" else ("🟡" if state['status'] == "WARNING" else "🔴")
    
    with col1:
        st.metric("System Status", f"{status_color} {state['status']}")
    with col2:
        st.metric("Current Flux (>2 MeV)", f"{state['current_flux_gt2MeV']:.1f} pfu")
    with col3:
        st.metric("Last Updated", state['timestamp'].split('.')[0].replace('T', ' '))
        
    st.subheader("🔮 Predictive Horizons")
    
    # Display predictions
    horizons = state['horizons']
    mean_preds = state['predictions_pfu']
    p95_preds = state['p95_pfu']
    
    cols = st.columns(len(horizons))
    
    for i, (hz, mean_v, p95_v) in enumerate(zip(horizons, mean_preds, p95_preds)):
        with cols[i]:
            st.markdown(f"**{hz} Forecast**")
            st.metric("Expected Flux", f"{mean_v:.1f} pfu")
            st.metric("95% Upper Bound", f"{p95_v:.1f} pfu", delta=f"{p95_v - mean_v:.1f}", delta_color="inverse")
            if p95_v > 1000:
                st.error("CRITICAL RISK")
            elif p95_v > 100:
                st.warning("WARNING")
            else:
                st.success("SAFE")




# =============================================================================
# Main App
# =============================================================================

def main():
    """Main Streamlit application."""
    render_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://www.isro.gov.in/media_isro/image/index/ISRO_Logo.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["📡 Live Operations", "📊 Data Explorer", "🔮 Predictions", "📈 Performance", "ℹ️ About"],
            index=0
        )
        
        st.divider()
        
        # System info
        st.caption("System Info")
        st.caption(f"PyTorch: {torch.__version__}")
        st.caption(f"CUDA: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU'}")
    
    # Route to page
    if page == "📡 Live Operations":
        page_live_operations()
    elif page == "📊 Data Explorer":
        page_data_explorer()
    elif page == "🔮 Predictions":
        page_predictions()
    elif page == "📈 Performance":
        page_performance()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
