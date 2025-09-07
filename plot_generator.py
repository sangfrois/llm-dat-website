import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from multiple_test import analyze_results
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import io
import base64

def create_ridge_data(results_df):
    """Create ridge plot data for D3.js visualization"""
    # Filter data exactly as in original code
    df = results_df.loc[(results_df['Strategy']=='Original instructions') &
                        (results_df['Temperature']=='Mid') | (results_df['Temperature'].isnull())]
    
    # Remove outliers and sample data
    df = df.groupby('Model').apply(lambda x: x[np.abs(x['Score'] - x['Score'].mean()) <= 3 * x['Score'].std()]).reset_index(drop=True)
    df = df.groupby('Model').apply(lambda x: x.sample(min(len(x), 500), random_state=32)).reset_index(drop=True)
    
    # Get order exactly as in static version (ascending = low to high scores)
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=False).index
    
    # Get rocket_r colors exactly as in original
    rocket_r_colors = ['#03051A', '#1B0C42', '#4B0C6B', '#781C6D', '#A52C60', '#CF4446', '#ED6925', '#FB9A06', '#F7D03C', '#FCFFA4']
    
    # Calculate statistics using existing analyze_results function
    mean_conf, _, _, _ = analyze_results(df, 'Model', order)
    
    ridge_data = []
    
    for i, model in enumerate(order):
        model_data = df[df['Model'] == model]['Score'].dropna()
        
        if len(model_data) > 0:
            # Create KDE for smooth density curve
            kde = gaussian_kde(model_data)
            # Use data-driven x-axis range with some padding
            data_min = model_data.min()
            data_max = model_data.max()
            
    # Calculate global range across all models for consistent x-axis
    global_min = df['Score'].min() - 2
    global_max = df['Score'].max() + 2
    x_range = np.linspace(global_min, global_max, 200)
    density = kde(x_range)
            
            # Get color from rocket_r palette
            color_idx = int(i * (len(rocket_r_colors)-1) / (len(order)-1))
            color = rocket_r_colors[color_idx]
            
            # Special color for Human (100k) as in original
            if model == 'Human (100k)':
                color = "#F4FF5E"
            
            # Get stats from mean_conf
            model_stats = mean_conf[mean_conf['Model'] == model]
            mean_val = model_stats['mean'].values[0] if len(model_stats) > 0 else model_data.mean()
            sem_val = model_stats['sem'].values[0] if len(model_stats) > 0 else model_data.sem()
            std_val = model_stats['std'].values[0] if len(model_stats) > 0 else model_data.std()
            
            # Create density points for D3
            density_points = []
            for j, x_val in enumerate(x_range):
                density_points.append({
                    'x': float(x_val),
                    'y': float(density[j])
                })
            
            ridge_data.append({
                'category': model,
                'color': color,
                'values': density_points,
                'mean': round(float(mean_val), 2),
                'std': round(float(std_val), 2),
                'sem': round(float(sem_val), 2),
                'samples': int(len(model_data))
            })
    
    return ridge_data

def create_horizontal_bar_plot(results_df):
    """Create horizontal bar plot exactly as in original code"""
    # Filter data exactly as in original code
    df = results_df.loc[(results_df['Strategy']=='Original instructions') &
                        ((results_df['Temperature']=='Mid') | (results_df['Temperature'].isnull()))]
    
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
    
    df = df.groupby('Model').apply(lambda x: x[np.abs(x['Score'] - x['Score'].mean()) <= 3 * x['Score'].std()]).reset_index(drop=True)
    df = df.groupby('Model').apply(lambda x: x.sample(min(len(x), 500), random_state=32)).reset_index(drop=True)
    
    # ASCENDING = LOW TO HIGH SCORES - maintain exact same order logic  
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=False).index
    mean_conf, _, _, _ = analyze_results(df, 'Model', order)
    
    # Create palette exactly as in original (rocket_r) - maintain color consistency
    rocket_r_colors = ['#03051A', '#1B0C42', '#4B0C6B', '#781C6D', '#A52C60', '#CF4446', '#ED6925', '#FB9A06', '#F7D03C', '#FCFFA4']
    colors = []
    for i, model in enumerate(order):
        color_idx = int(i * (len(rocket_r_colors)-1) / (len(order)-1))
        colors.append(rocket_r_colors[color_idx])
    
    # Special color for Human (100k) as in original - EXACT SAME LOGIC
    final_colors = []
    for i, model in enumerate(order):
        if model == 'Human (100k)':
            final_colors.append("#F4FF5E")
        else:
            final_colors.append(colors[i])
    
    fig = go.Figure()
    
    # Add horizontal bars
    for i, model in enumerate(order):
        model_data = mean_conf[mean_conf['Model'] == model]
        mean_val = model_data['mean'].values[0]
        sem_val = model_data['sem'].values[0]
        
        fig.add_trace(go.Bar(
            x=[mean_val],
            y=[model],
            orientation='h',
            marker_color=final_colors[i],
            error_x=dict(
                type='data',
                array=[sem_val],
                color='black',
                thickness=2,
                width=5
            ),
            name=model,
            showlegend=False,
            hovertemplate=f'<b>{model}</b><br>Mean: %{{x:.1f}}<br>SEM: ±{sem_val:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Mean creativity scores by model</b>',
            font=dict(size=18, color='white'),
            x=0.5,
            pad=dict(t=15)
        ),
        xaxis=dict(
            title=dict(text='<b>Creativity score</b>', font=dict(size=16, color='white')),
            tickfont=dict(size=14, color='white'),
            range=[60, 90],
            gridcolor='rgba(255,255,255,0.3)',
            showgrid=True,
            gridwidth=1
        ),
        yaxis=dict(
            title=dict(text='<b>Model</b>', font=dict(size=16, color='white')),
            tickfont=dict(size=14, color='white'),
            categoryorder='array',
            categoryarray=list(reversed(order))
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=len(order) * 50 + 150,
        margin=dict(l=150, r=50, t=100, b=80)
    )
    
    return fig

def create_heatmap_plot(results_df):
    """Create heatmap plot using the existing create_heatmap function logic"""
    # Filter data exactly as in original code
    df = results_df.loc[(results_df['Strategy']=='Original instructions') &
                        ((results_df['Temperature']=='Mid') | (results_df['Temperature'].isnull()))]
    
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
    
    df = df.groupby('Model').apply(lambda x: x[np.abs(x['Score'] - x['Score'].mean()) <= 3 * x['Score'].std()]).reset_index(drop=True)
    df = df.groupby('Model').apply(lambda x: x.sample(min(len(x), 500), random_state=32)).reset_index(drop=True)
    
    # ASCENDING = LOW TO HIGH SCORES - maintain exact same order logic
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=False).index
    mean_conf, pvals_table, tvals_table, cohen_d_table = analyze_results(df, 'Model', order)
    
    # Create subplot with bar plot and heatmap
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Mean DAT score</b>', '<b>Pairwise contrasts (t-values)</b>'),
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.1
    )
    
    # Bar plot (left subplot) - maintain color consistency
    rocket_r_colors = ['#03051A', '#1B0C42', '#4B0C6B', '#781C6D', '#A52C60', '#CF4446', '#ED6925', '#FB9A06', '#F7D03C', '#FCFFA4']
    colors = []
    for i, model in enumerate(order):
        color_idx = int(i * (len(rocket_r_colors)-1) / (len(order)-1))
        colors.append(rocket_r_colors[color_idx])
    
    for i, model in enumerate(order):
        model_data = mean_conf[mean_conf['Model'] == model]
        mean_val = model_data['mean'].values[0]
        sem_val = model_data['sem'].values[0]
        
        fig.add_trace(go.Bar(
            x=[model],
            y=[mean_val],
            marker_color=colors[i],
            error_y=dict(
                type='data',
                array=[sem_val],
                color='black',
                thickness=2,
                width=5
            ),
            name=model,
            showlegend=False,
            hovertemplate=f'<b>{model}</b><br>Mean: %{{y:.1f}}<br>SEM: ±{sem_val:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # Heatmap (right subplot) - remove first row and last column as in original
    heatmap_data = tvals_table.iloc[1:, :-1]
    pval_stars = pvals_table.iloc[1:, :-1].applymap(
        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
    )
    
    # Create masked heatmap data (upper triangle)
    masked_data = heatmap_data.values.copy()
    for i in range(len(masked_data)):
        for j in range(len(masked_data[0])):
            if i <= j:
                masked_data[i, j] = np.nan
    
    vmax = np.nanmax(np.abs(masked_data))
    
    # Create text annotations for heatmap
    text_annotations = []
    for i in range(len(heatmap_data.index)):
        row_text = []
        for j in range(len(heatmap_data.columns)):
            if i <= j:  # Upper triangle mask
                row_text.append('')
            else:
                tval = heatmap_data.iloc[i, j]
                stars = pval_stars.iloc[i, j]
                row_text.append(f"{stars}<br>{tval:.2f}")
        text_annotations.append(row_text)
    
    fig.add_trace(go.Heatmap(
        z=masked_data,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-vmax,
        zmax=vmax,
        colorbar=dict(
            title=dict(text='<b>t-values</b>', font=dict(size=14, color='white')),
            tickfont=dict(size=12, color='white'),
            x=1.02
        ),
        showscale=True,
        text=text_annotations,
        texttemplate='%{text}',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{y} vs %{x}</b><br>t-value: %{z:.2f}<extra></extra>'
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        margin=dict(l=50, r=100, t=100, b=100)
    )
    
    # Update bar plot axes
    fig.update_xaxes(
        tickangle=90,
        tickfont=dict(size=12, color='white'),
        title='',
        row=1, col=1
    )
    fig.update_yaxes(
        title=dict(text='<b>Mean</b>', font=dict(size=14, color='white')),
        tickfont=dict(size=12, color='white'),
        range=[60, 90],
        row=1, col=1
    )
    
    # Update heatmap axes
    fig.update_xaxes(
        tickangle=90,
        tickfont=dict(size=12, color='white'),
        title='',
        row=1, col=2
    )
    fig.update_yaxes(
        tickangle=0,
        tickfont=dict(size=12, color='white'),
        title='',
        row=1, col=2
    )
    
    # Update subplot titles
    fig.update_annotations(font=dict(size=16, color='white'))
    
    return fig
