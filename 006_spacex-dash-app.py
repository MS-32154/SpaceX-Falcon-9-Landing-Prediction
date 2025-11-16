"""
SpaceX Falcon 9 Launch Analytics Dashboard
==========================================
Interactive dashboard for analyzing SpaceX launch data and landing success patterns.

Date: November 2025
"""

import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    """Load and preprocess SpaceX launch data"""
    try:
        df = pd.read_csv("spacex_launch_dash_wiki.csv")
        
        # Data validation
        required_columns = ['Launch Site', 'class', 'Payload Mass (kg)', 'Booster Version Category']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in dataset")
        
        # Clean data
        df['class'] = df['class'].astype(int)
        df['Success'] = df['class'].map({1: 'Success', 0: 'Failure'})
        
        # Add date if available
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
        
        return df
    except FileNotFoundError:
        print("Error: spacex_launch_dash.csv not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

spacex_df = load_data()

if spacex_df is None:
    # Create dummy data for development
    spacex_df = pd.DataFrame({
        'Launch Site': ['CCAFS SLC 40'] * 10,
        'class': [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        'Payload Mass (kg)': [5000, 6000, 4500, 7000, 5500, 6500, 4800, 7200, 5200, 6800],
        'Booster Version Category': ['v1.0'] * 10,
        'Success': ['Success', 'Failure', 'Success', 'Success', 'Success', 
                    'Failure', 'Success', 'Success', 'Success', 'Success']
    })

# Calculate metrics
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
total_launches = len(spacex_df)
total_success = spacex_df['class'].sum()
success_rate = (total_success / total_launches * 100) if total_launches > 0 else 0

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

app.title = "SpaceX Launch Analytics"
server = app.server

# ============================================================================
# STYLING
# ============================================================================

# Color scheme
colors = {
    'background': '#FFFFFF',
    'text': '#2C3E50',
    'primary': '#2E86AB',
    'success': '#27AE60',
    'failure': '#E74C3C',
    'secondary': '#95A5A6',
    'card': '#F8F9FA',
    'border': '#E1E8ED'
}

# Custom CSS styles
styles = {
    'container': {
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    'header': {
        'backgroundColor': colors['primary'],
        'color': 'white',
        'padding': '30px',
        'borderRadius': '8px',
        'marginBottom': '30px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'card': {
        'backgroundColor': colors['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
        'border': f'1px solid {colors["border"]}'
    },
    'metric_card': {
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'textAlign': 'center',
        'border': f'2px solid {colors["border"]}'
    },
    'metric_value': {
        'fontSize': '36px',
        'fontWeight': 'bold',
        'color': colors['primary'],
        'marginBottom': '5px'
    },
    'metric_label': {
        'fontSize': '14px',
        'color': colors['secondary'],
        'textTransform': 'uppercase',
        'letterSpacing': '1px'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_site_options():
    """Generate dropdown options for launch sites"""
    options = [{'label': 'All Sites', 'value': 'ALL'}]
    options.extend([
        {'label': site, 'value': site} 
        for site in sorted(spacex_df["Launch Site"].unique())
    ])
    return options

def get_site_stats(df, site):
    """Calculate statistics for a specific site or all sites"""
    if site != 'ALL':
        df = df[df['Launch Site'] == site]
    
    total = len(df)
    successes = df['class'].sum()
    failures = total - successes
    rate = (successes / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'successes': successes,
        'failures': failures,
        'rate': rate
    }

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_header():
    """Create dashboard header"""
    return html.Div([
        html.H1('SpaceX Falcon 9 Launch Analytics Dashboard', 
                style={'margin': '0', 'fontSize': '32px', 'fontWeight': '600'}),
        html.P('Interactive analysis of launch success patterns and performance metrics',
               style={'margin': '10px 0 0 0', 'fontSize': '16px', 'opacity': '0.9'})
    ], style=styles['header'])

def create_metrics_row():
    """Create key metrics summary cards"""
    return html.Div([
        html.Div([
            html.Div([
                html.Div(f"{total_launches}", style=styles['metric_value']),
                html.Div("Total Launches", style=styles['metric_label'])
            ], style=styles['metric_card'], className='metric-card')
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
        
        html.Div([
            html.Div([
                html.Div(f"{total_success}", style={**styles['metric_value'], 'color': colors['success']}),
                html.Div("Successful", style=styles['metric_label'])
            ], style=styles['metric_card'], className='metric-card')
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
        
        html.Div([
            html.Div([
                html.Div(f"{total_launches - total_success}", style={**styles['metric_value'], 'color': colors['failure']}),
                html.Div("Failed", style=styles['metric_label'])
            ], style=styles['metric_card'], className='metric-card')
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
        
        html.Div([
            html.Div([
                html.Div(f"{success_rate:.1f}%", style={**styles['metric_value'], 'color': colors['primary']}),
                html.Div("Success Rate", style=styles['metric_label'])
            ], style=styles['metric_card'], className='metric-card')
        ], style={'width': '24%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px'})

def create_controls():
    """Create filter controls section"""
    return html.Div([
        html.Div([
            html.Label('Launch Site Selection', 
                      style={'fontWeight': '600', 'marginBottom': '10px', 'display': 'block',
                             'color': colors['text'], 'fontSize': '14px'}),
            dcc.Dropdown(
                id='site-dropdown',
                options=create_site_options(),
                value='ALL',
                placeholder="Select a Launch Site",
                searchable=True,
                style={'borderRadius': '4px'}
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Payload Mass Range (kg)', 
                      style={'fontWeight': '600', 'marginBottom': '10px', 'display': 'block',
                             'color': colors['text'], 'fontSize': '14px'}),
            dcc.RangeSlider(
                id='payload-slider',
                min=0,
                max=int(max_payload) + 1000,
                step=500,
                value=[int(min_payload), int(max_payload)],
                marks={
                    int(min_payload): f'{int(min_payload/1000)}k',
                    int(max_payload/2): f'{int(max_payload/2000)}k',
                    int(max_payload): f'{int(max_payload/1000)}k'
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(id='payload-output', style={'textAlign': 'center', 'marginTop': '10px',
                                                  'color': colors['secondary'], 'fontSize': '13px'})
        ])
    ], style=styles['card'])

def create_charts_section():
    """Create main charts section"""
    return html.Div([
        # Row 1: Pie charts
        html.Div([
            html.Div([
                html.H3('Launch Success Distribution', 
                       style={'color': colors['text'], 'fontSize': '18px', 'marginBottom': '15px'}),
                dcc.Graph(id='success-pie-chart', config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.H3('Booster Version Distribution', 
                       style={'color': colors['text'], 'fontSize': '18px', 'marginBottom': '15px'}),
                dcc.Graph(id='booster-pie-chart', config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
        ], style=styles['card']),
        
        # Row 2: Scatter plot
        html.Div([
            html.H3('Payload vs Success Correlation', 
                   style={'color': colors['text'], 'fontSize': '18px', 'marginBottom': '15px'}),
            dcc.Graph(id='success-payload-scatter-chart', config={'displayModeBar': True})
        ], style=styles['card']),
        
        # Row 3: Bar chart
        html.Div([
            html.H3('Success Rate by Launch Site', 
                   style={'color': colors['text'], 'fontSize': '18px', 'marginBottom': '15px'}),
            dcc.Graph(id='success-rate-bar-chart', config={'displayModeBar': False})
        ], style=styles['card']),
        
        # Row 4: Summary statistics table
        html.Div([
            html.H3('Launch Statistics Summary', 
                   style={'color': colors['text'], 'fontSize': '18px', 'marginBottom': '15px'}),
            html.Div(id='summary-table')
        ], style=styles['card'])
    ])

# ============================================================================
# APP LAYOUT
# ============================================================================

app.layout = html.Div([
    create_header(),
    html.Div([
        create_metrics_row(),
        create_controls(),
        create_charts_section()
    ], style=styles['container'])
])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('payload-output', 'children'),
    Input('payload-slider', 'value')
)
def update_payload_output(value):
    """Update payload range display"""
    return f'Selected range: {value[0]:,} kg - {value[1]:,} kg'

@app.callback(
    Output('success-pie-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_success_pie(site, payload_range):
    """Update success distribution pie chart"""
    # Filter data
    filtered_df = spacex_df[
        spacex_df['Payload Mass (kg)'].between(payload_range[0], payload_range[1])
    ]
    
    if site == 'ALL':
        # Show success distribution across all sites
        fig = px.pie(
            filtered_df,
            names='Launch Site',
            values='class',
            title='Successful Launches by Site',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        # Show success vs failure for selected site
        filtered_df = filtered_df[filtered_df['Launch Site'] == site]
        success_counts = filtered_df['Success'].value_counts()
        
        fig = px.pie(
            values=success_counts.values,
            names=success_counts.index,
            title=f'Launch Outcomes: {site}',
            color=success_counts.index,
            color_discrete_map={'Success': colors['success'], 'Failure': colors['failure']}
        )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=350,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    
    return fig

@app.callback(
    Output('booster-pie-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_booster_pie(site, payload_range):
    """Update booster version distribution pie chart"""
    # Filter data
    filtered_df = spacex_df[
        spacex_df['Payload Mass (kg)'].between(payload_range[0], payload_range[1])
    ]
    
    if site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == site]
    
    booster_counts = filtered_df['Booster Version Category'].value_counts()
    
    fig = px.pie(
        values=booster_counts.values,
        names=booster_counts.index,
        title='Launches by Booster Version',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=350,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    
    return fig

@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter(site, payload_range):
    """Update payload vs success scatter plot"""
    # Filter data
    filtered_df = spacex_df[
        spacex_df['Payload Mass (kg)'].between(payload_range[0], payload_range[1])
    ]
    
    if site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == site]
        title = f'Payload vs Success: {site}'
    else:
        title = 'Payload vs Success: All Sites'
    
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title=title,
        labels={'class': 'Success (1) / Failure (0)'},
        hover_data=['Launch Site'] if site == 'ALL' else None,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(
        height=400,
        xaxis_title='Payload Mass (kg)',
        yaxis_title='Launch Outcome',
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Failure', 'Success']),
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@app.callback(
    Output('success-rate-bar-chart', 'figure'),
    Input('payload-slider', 'value')
)
def update_bar_chart(payload_range):
    """Update success rate bar chart"""
    # Filter data
    filtered_df = spacex_df[
        spacex_df['Payload Mass (kg)'].between(payload_range[0], payload_range[1])
    ]
    
    # Calculate success rate by site
    site_stats = filtered_df.groupby('Launch Site').agg({
        'class': ['sum', 'count', 'mean']
    }).reset_index()
    
    site_stats.columns = ['Launch Site', 'Successes', 'Total', 'Success Rate']
    site_stats['Success Rate'] = site_stats['Success Rate'] * 100
    site_stats = site_stats.sort_values('Success Rate', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=site_stats['Launch Site'],
        x=site_stats['Success Rate'],
        orientation='h',
        text=site_stats['Success Rate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        marker=dict(
            color=site_stats['Success Rate'],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>%{y}</b><br>Success Rate: %{x:.1f}%<br>Total: %{customdata[0]}<extra></extra>',
        customdata=site_stats[['Total']]
    ))
    
    fig.update_layout(
        title='Success Rate Comparison',
        xaxis_title='Success Rate (%)',
        yaxis_title='',
        height=300,
        margin=dict(t=40, b=40, l=150, r=80),
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('summary-table', 'children'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_summary_table(site, payload_range):
    """Update summary statistics table"""
    # Filter data
    filtered_df = spacex_df[
        spacex_df['Payload Mass (kg)'].between(payload_range[0], payload_range[1])
    ]
    
    if site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == site]
    
    # Calculate statistics by site
    summary = filtered_df.groupby('Launch Site').agg({
        'class': ['count', 'sum', 'mean'],
        'Payload Mass (kg)': ['mean', 'min', 'max']
    }).reset_index()
    
    summary.columns = ['Launch Site', 'Total Launches', 'Successful', 'Success Rate',
                       'Avg Payload (kg)', 'Min Payload (kg)', 'Max Payload (kg)']
    
    summary['Success Rate'] = (summary['Success Rate'] * 100).round(1)
    summary['Avg Payload (kg)'] = summary['Avg Payload (kg)'].round(0).astype(int)
    summary['Min Payload (kg)'] = summary['Min Payload (kg)'].round(0).astype(int)
    summary['Max Payload (kg)'] = summary['Max Payload (kg)'].round(0).astype(int)
    
    # Add percentage formatting
    summary['Success Rate'] = summary['Success Rate'].apply(lambda x: f'{x}%')
    
    return dash_table.DataTable(
        data=summary.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in summary.columns],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'inherit',
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['card']
            }
        ],
        style_table={'overflowX': 'auto'}
    )

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)