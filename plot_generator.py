import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from multiple_test import analyze_results
import seaborn as sns
from scipy import stats

def create_ridge_plot(results_df):
    """Create a ridge plot using Plotly equivalent to the seaborn version"""
    # Filter data exactly as in original code
    df = results_df.loc[(results_df['Strategy']=='Original instructions') &
                        (results_df['Temperature']=='Mid') | (results_df['Temperature'].isnull())]
    
    # Remove outliers and sample data
    df = df.groupby('Model').apply(lambda x: x[np.abs(x['Score'] - x['Score'].mean()) <= 3 * x['Score'].std()]).reset_index(drop=True)
    df = df.groupby('Model').apply(lambda x: x.sample(min(len(x), 500), random_state=32)).reset_index(drop=True)
    
    # Get order and colors (rocket_r palette equivalent) - ASCENDING = LOW TO HIGH SCORES
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=True).index
    rocket_r_colors = ['#03051A', '#1B0C42', '#4B0C6B', '#781C6D', '#A52C60', '#CF4446', '#ED6925', '#FB9A06', '#F7D03C', '#FCFFA4']
    # Map colors to order index to maintain consistency
    colors = []
    for i, model in enumerate(order):
        color_idx = int(i * (len(rocket_r_colors)-1) / (len(order)-1))
        colors.append(rocket_r_colors[color_idx])
    
    # Calculate means for vertical lines
    mean_conf, _, _, _ = analyze_results(df, 'Model', order)
    
    fig = go.Figure()
    
    # Create ridge plot - reverse order for display (highest scores at top)
    reversed_order = list(reversed(order))
    for i, model in enumerate(reversed_order):
        model_data = df[df['Model'] == model]['Score'].values
        
        if len(model_data) == 0:
            continue
            
        # Create KDE using scipy with fewer points for performance
        kde = stats.gaussian_kde(model_data, bw_method=1.0)
        x_range = np.linspace(20, 100, 100)
        density = kde(x_range)
        
        # Normalize density for ridge effect
        density_normalized = density / density.max() * 0.8
        y_offset = i * 1.2
        
        # Get color for this model from original order
        original_idx = list(order).index(model)
        model_color = colors[original_idx]
        
        # Add filled area using tonexty with baseline
        fig.add_trace(go.Scatter(
            x=[20, 100],
            y=[y_offset, y_offset],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density_normalized + y_offset,
            fill='tonexty',
            fillcolor=model_color,
            line=dict(color='white', width=2),
            name=model,
            showlegend=False,
            hovertemplate=f'<b>{model}</b><br>Score: %{{x:.1f}}<extra></extra>'
        ))
        
        # Add baseline line
        fig.add_trace(go.Scatter(
            x=[20, 100],
            y=[y_offset, y_offset],
            mode='lines',
            line=dict(color=model_color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add mean line (vertical dashed line)
        model_mean = mean_conf[mean_conf['Model'] == model]['mean'].values[0]
        fig.add_trace(go.Scatter(
            x=[model_mean, model_mean],
            y=[y_offset, y_offset + 0.4],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False,
            hovertemplate=f'<b>{model}</b><br>Mean: {model_mean:.1f}<extra></extra>'
        ))
        
        # Add model label
        fig.add_annotation(
            x=22,
            y=y_offset + 0.2,
            text=f'<b>{model}</b>',
            showarrow=False,
            font=dict(size=16, color=model_color),
            xanchor='left'
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Creativity Score Distribution by Model</b>',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='<b>Creativity score</b>', font=dict(size=16, color='white')),
            tickfont=dict(size=14, color='white'),
            range=[20, 100],
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=len(reversed_order) * 80 + 100,
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    return fig

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
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=True).index
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
    order = df.groupby('Model')['Score'].mean().dropna().sort_values(ascending=True).index
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
