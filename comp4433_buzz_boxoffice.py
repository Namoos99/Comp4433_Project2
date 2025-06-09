import pandas as pd
import numpy as np
from datetime import timedelta
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------
#  Configuration & Data Loading
# ------------------------------

# Paths to your two CSV files:
MOVIES_CSV  = '20_24_movies.csv'       # Updated to correct filename
REVIEWS_CSV = 'movie_reviews.csv'      # must contain: movie_id, review_date, rating

print("="*60)
print("MOVIE BUZZ vs BOX OFFICE DASHBOARD")
print("="*60)

# 1) Load movies + revenue
try:
    movies_df_all = pd.read_csv(MOVIES_CSV)
    print(f"✓ Loaded {len(movies_df_all)} movies from {MOVIES_CSV}")
    print(f"  Columns: {list(movies_df_all.columns)}")
    
    # Check for required columns
    required_cols = ['movie_id', 'title', 'release_date', 'revenue']
    missing_cols = [col for col in required_cols if col not in movies_df_all.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
        
except FileNotFoundError:
    print(f"✗ Error: File '{MOVIES_CSV}' not found!")
    print(f"  Make sure the file exists in: {os.getcwd()}")
    exit(1)
except Exception as e:
    print(f"✗ Error loading movies CSV: {e}")
    exit(1)

# Standardize movie_id column name
if 'movieId' in movies_df_all.columns:
    movies_df_all = movies_df_all.rename(columns={'movieId': 'movie_id'})
elif 'MovieID' in movies_df_all.columns:
    movies_df_all = movies_df_all.rename(columns={'MovieID': 'movie_id'})
elif 'movie_Id' in movies_df_all.columns:
    movies_df_all = movies_df_all.rename(columns={'movie_Id': 'movie_id'})

# Parse release_date → datetime, drop invalid, strip tz
movies_df_all['release_date'] = pd.to_datetime(
    movies_df_all['release_date'], errors='coerce'
).dt.tz_localize(None)
movies_df_all = movies_df_all.dropna(subset=['release_date']).copy()

# Ensure revenue numeric
movies_df_all['revenue'] = pd.to_numeric(
    movies_df_all.get('revenue', 0), errors='coerce'
).fillna(0).astype(int)

# Get available years
available_years = sorted(movies_df_all['release_date'].dt.year.unique())
print(f"✓ Years available: {', '.join(map(str, available_years))}")

# Show top movies by revenue
print(f"\n📊 Top 5 Movies by Revenue:")
top_movies = movies_df_all.nlargest(5, 'revenue')[['title', 'revenue', 'release_date']]
for _, movie in top_movies.iterrows():
    print(f"  • {movie['title']} ({movie['release_date'].year}): ${movie['revenue']:,}")

# 2) Load all reviews
try:
    reviews_df_global = pd.read_csv(REVIEWS_CSV, parse_dates=['review_date'])
    print(f"\n✓ Loaded {len(reviews_df_global)} reviews from {REVIEWS_CSV}")
except Exception as e:
    print(f"✗ Error loading reviews CSV: {e}")
    exit(1)

# Standardize column names
if 'movieId' in reviews_df_global.columns:
    reviews_df_global = reviews_df_global.rename(columns={'movieId': 'movie_id'})
elif 'MovieID' in reviews_df_global.columns:
    reviews_df_global = reviews_df_global.rename(columns={'MovieID': 'movie_id'})

# Strip tzinfo
reviews_df_global['review_date'] = reviews_df_global['review_date'].dt.tz_localize(None)

# Quick stats
unique_movies_reviewed = reviews_df_global['movie_id'].nunique()
avg_rating = reviews_df_global['rating'].mean()
print(f"✓ Reviews cover {unique_movies_reviewed} unique movies")
print(f"✓ Average rating: {avg_rating:.2f}")

print("="*60)

# ------------------
#  Dash App Setup
# ------------------

app = Dash(__name__)
app.title = "Movie Buzz vs. Box Office Dashboard"

# ----------------
#  App Layout
# ----------------

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎬 Early Buzz vs. Box Office Performance",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
        html.H3("Movie Releases Analysis (2020–2024)",
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
    ]),

    # Instructions Panel
    html.Div([
        html.H4("📊 How to Use This Dashboard:", 
                style={'color': '#34495e', 'marginBottom': '15px'}),
        
        # Two column layout for instructions
        html.Div([
            # Left column - Basic Usage
            html.Div([
                html.H5("🎯 Basic Controls:", style={'color': '#3498db', 'marginBottom': '10px'}),
                html.Ul([
                    html.Li([
                        html.Strong("Select Year: "),
                        "Choose from 2020-2024 to analyze different release periods"
                    ]),
                    html.Li([
                        html.Strong("Buzz Metric: "),
                        "Toggle between review count (volume) or average rating (quality)"
                    ]),
                    html.Li([
                        html.Strong("Buzz Window: "),
                        "Adjust days after release (7-60) to define 'early' reviews"
                    ]),
                    html.Li([
                        html.Strong("Min Reviews: "),
                        "Filter out movies with too few reviews for reliable analysis"
                    ]),
                    html.Li([
                        html.Strong("Revenue Scale: "),
                        "Switch to log scale for better visibility of lower-revenue films"
                    ]),
                ], style={'fontSize': '14px'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Right column - Understanding the Data
            html.Div([
                html.H5("📈 Understanding the Visualizations:", style={'color': '#27ae60', 'marginBottom': '10px'}),
                html.Ul([
                    html.Li([
                        html.Strong("Scatter Plot: "),
                        "Shows relationship between early buzz and final revenue"
                    ]),
                    html.Li([
                        html.Strong("Color Coding: "),
                        "Green = Hit (top 25%), Red = Low (bottom 25%), Orange = Mixed"
                    ]),
                    html.Li([
                        html.Strong("Correlation Score: "),
                        "Measures strength of relationship (-1 to 1, closer to 1 = stronger)"
                    ]),
                    html.Li([
                        html.Strong("Box Plots: "),
                        "Show rating distributions across performance categories"
                    ]),
                    html.Li([
                        html.Strong("Top Movies: "),
                        "Ranked by selected metric to identify outliers"
                    ]),
                ], style={'fontSize': '14px'}),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        
        # Key Questions Section
        html.Div([
            html.H5("❓ Understanding The Dashboard:", 
                    style={'color': '#e74c3c', 'marginBottom': '10px', 'marginTop': '20px'}),
            html.P("This dashboard explores the relationship between early movie buzz in the form of reviews and box office performance. " \
            "It looks at the quality and quantity of reviews from a period before a movies release to a user defined end point to answer the questions below.", 
                   style = {'fontSize': '14px'}),
            html.Ol([
                html.Li("Does early audience engagement predict box office success?"),
                html.Li("Is review volume or quality a better predictor?"),
                html.Li("What's the minimum buzz threshold for commercial viability?"),
                html.Li("How consistent are these patterns across different years?"),
            ], style={'fontSize': '14px'}),
        ]),
    ], style={
        'backgroundColor': '#ecf0f1', 
        'padding': '20px', 
        'borderRadius': '10px', 
        'marginBottom': '25px',
        'border': '1px solid #bdc3c7',
        'maxWidth': '900px',
        'margin': '0 auto 25px auto'
    }),

    # Controls Section
    html.Div([
        # Row 1: Year and Scale
        html.Div([
            html.Div([
                html.Label("📅 Select Year:", 
                          style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(y), 'value': y} for y in available_years],
                    value=available_years[-1] if available_years else 2024,
                    clearable=False,
                    style={'marginBottom': '15px'}
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Label("📈 Revenue Scale:", 
                          style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.RadioItems(
                    id='scale-radio',
                    options=[
                        {'label': 'Linear Scale', 'value': 'linear'},
                        {'label': 'Logarithmic Scale', 'value': 'log'}
                    ],
                    value='linear',
                    inline=True,
                    style={'marginTop': '5px'}
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        # Row 2: Buzz Metric
        html.Div([
            html.Label("🎯 Buzz Metric:", 
                      style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': '📊 Review Count (Buzz Volume)', 'value': 'early_count'},
                    {'label': '⭐ Average Early Rating (Buzz Quality)', 'value': 'early_avg'},
                ],
                value='early_count',
                clearable=False,
                style={'marginBottom': '20px'}
            ),
        ]),

        # Row 3: Sliders
        html.Div([
            html.Label("⏰ Buzz Window (days after release):", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Slider(
                id='window-slider',
                min=7, max=60, step=7, value=30,
                marks={i: f'{i}d' for i in range(7, 61, 7)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'marginBottom': '25px'}),

        html.Div([
            html.Label("📊 Minimum Early Reviews:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Slider(
                id='min-review-slider',
                min=0, max=20, step=1, value=0,
                marks={0: '0', 5: '5', 10: '10', 15: '15', 20: '20'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'marginBottom': '25px'}),

        # Row 4: Title Filter
        html.Div([
            html.Label("🔍 Filter by Movie Title:", 
                      style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Input(
                id='title-input',
                type='text',
                placeholder='Search for movies... (e.g., Dune, Spider-Man)',
                
                style={
                    'width': '100%', 
                    'padding': '10px', 
                    'borderRadius': '5px',
                    'border': '1px solid #bdc3c7'
                }
            )
        ], style={'marginBottom': '30px'}),
        
    ], style={
        'maxWidth': '900px', 
        'margin': '0 auto', 
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px',
        'marginBottom': '30px'
    }),

    # Data Summary
    html.Div(id='data-summary', style={'marginBottom': '25px'}),

    # Visualizations
    html.Div([
        # Main scatter plot
        dcc.Graph(id='scatter-plot', style={'marginBottom': '20px'}),
        
        # Histograms
        html.Div([
            html.Div([
                dcc.Graph(id='metric-hist')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                dcc.Graph(id='revenue-hist')
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),
        
        # Additional visualizations
        dcc.Graph(id='rating-distribution', style={'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                dcc.Graph(id='top-movies-chart')
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ])
    ]),

    # Footer with Analysis Section
    html.Div([
        html.Hr(style={'margin': '50px 0', 'border': '2px solid #e9ecef'}),
        
        # Analysis Section
        html.Div([
            html.H2("🔍 Key Insights & Analysis", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            
            # Insight Cards
            html.Div([
                # Insight 1
                html.Div([
                    html.H4("📊 Weak Correlation Between Buzz and Revenue", 
                            style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.P("With a correlation of only 0.413, early review count explains just 17% of box office variance.", 
                           style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("This suggests that while early buzz matters, many other factors (marketing, competition, release timing) play crucial roles in determining box office success."),
                ], style={
                    'backgroundColor': '#fadbd8', 
                    'padding': '25px', 
                    'borderRadius': '12px', 
                    'marginBottom': '20px',
                    'border': '1px solid #e74c3c'
                }),

                # Insight 2
                html.Div([
                    html.H4("🎯 The 'Death Zone' Effect", 
                            style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("Movies with 0-2 early reviews almost always underperform.", 
                           style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("This threshold effect is more reliable than positive correlations. While high buzz doesn't guarantee success, extremely low buzz (< 3 reviews) is a strong predictor of failure."),
                ], style={
                    'backgroundColor': '#fdeaa7', 
                    'padding': '25px', 
                    'borderRadius': '12px', 
                    'marginBottom': '20px',
                    'border': '1px solid #f39c12'
                }),

                # Insight 3
                html.Div([
                    html.H4("⭐ Quality vs. Quantity", 
                            style={'color': '#27ae60', 'marginBottom': '15px'}),
                    html.P("Average ratings show even weaker correlation with revenue.", 
                           style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("This counterintuitive finding suggests that getting people talking (volume) may be more important than what they're saying (ratings). However, this could also indicate that blockbusters attract more diverse opinions."),
                ], style={
                    'backgroundColor': '#d5f4e6', 
                    'padding': '25px', 
                    'borderRadius': '12px', 
                    'marginBottom': '20px',
                    'border': '1px solid #27ae60'
                }),

                # Insight 4
                html.Div([
                    html.H4("📈 Non-Linear Relationships", 
                            style={'color': '#3498db', 'marginBottom': '15px'}),
                    html.P("The scatter plot reveals clusters rather than linear patterns.", 
                           style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("Many movies with 4-6 early reviews achieve vastly different revenues, from near-zero to over $1B. This suggests that early buzz might have threshold effects or interact with other variables like genre, franchise status, or star power."),
                ], style={
                    'backgroundColor': '#dbeafe', 
                    'padding': '25px', 
                    'borderRadius': '12px', 
                    'marginBottom': '30px',
                    'border': '1px solid #3498db'
                }),
            ]),

            # Methodology Section
            html.Div([
                html.H3("📐 Methodology & Approach", 
                        style={'color': '#2c3e50', 'marginBottom': '20px'}),
                
                html.Div([
                    html.H5("Data Sources:", style={'color': '#34495e', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("Movie data: 426 films released between 2020-2024"),
                        html.Li("Review data: 2,589 user reviews with ratings"),
                        html.Li("Box office revenue from reliable industry sources"),
                    ], style={'marginBottom': '15px'}),
                    
                    html.H5("Key Metrics:", style={'color': '#34495e', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("Early Buzz Window: Reviews within X days of release"),
                        html.Li("Review Count: Total number of early reviews"),
                        html.Li("Average Rating: Mean score of early reviews (1-10 scale)"),
                        html.Li("Revenue Categories: Based on 25th/75th percentiles"),
                    ]),
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '30px'
                }),
            ]),

            # Recommendations
            html.Div([
                html.H3("💡 Industry Recommendations", 
                        style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div([
                        html.H5("For Studios:", style={'color': '#8e44ad', 'marginBottom': '15px'}),
                        html.Ol([
                            html.Li("Ensure minimum engagement (3+ early reviews) through screening strategies"),
                            html.Li("Focus on generating discussion volume over perfect ratings"),
                            html.Li("Use early buzz as a risk indicator, not success predictor"),
                            html.Li("Combine buzz metrics with other factors for forecasting"),
                        ])
                    ], style={
                        'width': '48%', 
                        'display': 'inline-block', 
                        'verticalAlign': 'top',
                        'backgroundColor': '#f4f1fb',
                        'padding': '20px',
                        'borderRadius': '10px'
                    }),
                    
                    html.Div([
                        html.H5("For Analysts:", style={'color': '#e67e22', 'marginBottom': '15px'}),
                        html.Ol([
                            html.Li("Monitor the 0-3 review 'death zone' as a red flag"),
                            html.Li("Consider non-linear models for prediction"),
                            html.Li("Account for genre, franchise, and star effects"),
                            html.Li("Track sentiment beyond simple rating averages"),
                        ])
                    ], style={
                        'width': '48%', 
                        'display': 'inline-block',
                        'backgroundColor': '#fdf2e9',
                        'padding': '20px',
                        'borderRadius': '10px'
                    }),
                ], style={'display':'flex', 'justifyContent': 'space-between','marginBottom': '30px'}),
            ]),

            # Future Work
            html.Div([
                html.H3("🚀 Future Enhancements", 
                        style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.P("To improve predictive power, future versions could incorporate:", 
                       style={'marginBottom': '15px'}),
                html.Ul([
                    html.Li("Sentiment analysis of review text (beyond numeric ratings)"),
                    html.Li("Social media buzz metrics (Twitter mentions, YouTube trailer views)"),
                    html.Li("Genre-specific models (action films vs. dramas may behave differently)"),
                    html.Li("Seasonal and competition effects"),
                    html.Li("Marketing spend data"),
                    html.Li("Real-time tracking during release window"),
                ]),
            ], style={
                'backgroundColor': '#e8f5e9',
                'padding': '25px',
                'borderRadius': '10px',
                'marginBottom': '40px'
            }),
        ], style={'maxWidth': '1000px', 'margin': '0 auto', 'padding': '20px'}),
        
        html.Hr(style={'margin': '40px 0', 'border': '1px solid #e9ecef'}),
        
        html.P("Analysis conducted using Python, Dash, and Plotly • COMP 4433 Final Project", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginBottom': '20px'})
    ])
    
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# -----------------------
#  Callback & Processing
# -----------------------

@app.callback(
    [
        Output('scatter-plot', 'figure'),
        Output('metric-hist', 'figure'),
        Output('revenue-hist', 'figure'),
        Output('data-summary', 'children'),
        Output('rating-distribution', 'figure'),
        Output('correlation-plot', 'figure'),
        Output('top-movies-chart', 'figure'),
    ],
    [
        Input('year-dropdown', 'value'),
        Input('metric-dropdown', 'value'),
        Input('window-slider', 'value'),
        Input('min-review-slider', 'value'),
        Input('title-input', 'value'),
        Input('scale-radio', 'value'),
    ]
)
def update_charts(year, metric, window_days, min_reviews, title_filter, scale):
    try:
        # 1) Filter movies by year
        df_movies = movies_df_all[
            movies_df_all['release_date'].dt.year == year
        ].copy()
        
        if df_movies.empty:
            empty_fig = create_empty_figure(f"No movies found for {year}")
            summary = create_error_summary(f"No data available for {year}")
            return [empty_fig] * 5 + [summary, empty_fig]

        # 2) Filter reviews to those movies
        df_revs = reviews_df_global[
            reviews_df_global['movie_id'].isin(df_movies['movie_id'])
        ].copy()

        # 3) Compute early window per movie
        buzz = []
        for mid, grp in df_revs.groupby('movie_id', sort=False):
            movie_info = df_movies[df_movies['movie_id'] == mid]
            if movie_info.empty:
                continue
                
            rd = movie_info['release_date'].iloc[0]
            window_end = rd + timedelta(days=window_days)
            
            early = grp[
                (grp['review_date'] >= rd) &
                (grp['review_date'] <= window_end)
            ]
            
            buzz.append({
                'movie_id': mid,
                'early_count': len(early),
                'early_avg': early['rating'].mean() if not early.empty else np.nan,
                'total_reviews': len(grp),
                'overall_avg': grp['rating'].mean()
            })
            
        buzz_df = pd.DataFrame(buzz)

        # 4) Merge back to movies
        df = df_movies.merge(buzz_df, on='movie_id', how='left')
        df['early_count'] = df['early_count'].fillna(0).astype(int)
        df['total_reviews'] = df['total_reviews'].fillna(0).astype(int)

        # 5) Apply filters
        df = df[df['early_count'] >= min_reviews]
        
        if title_filter and title_filter.strip():
            df = df[df['title'].str.contains(title_filter.strip(), case=False, na=False)]
            
        if metric == 'early_avg':
            df = df.dropna(subset=['early_avg'])

        # 6) Add categories BEFORE creating summary
        df = add_movie_categories(df)
        
        # 7) Create summary (now it can access Category column)
        total_early_reviews = df['early_count'].sum()
        summary = create_data_summary(df, window_days, min_reviews, total_early_reviews, year)

        if df.empty:
            empty_fig = create_empty_figure("No movies match current filters", 
                                          "Try reducing minimum reviews or adjusting filters")
            return [empty_fig] * 5 + [summary, empty_fig]

        # 8) Create all visualizations
        scatter = create_scatter_plot(df, metric, scale)
        metric_hist = create_metric_histogram(df, metric)
        revenue_hist = create_revenue_histogram(df)
        rating_dist = create_rating_distribution(df, df_revs)
        correlation = create_correlation_analysis(df)
        top_movies = create_top_movies_chart(df, metric)

        return scatter, metric_hist, revenue_hist, summary, rating_dist, correlation, top_movies
        
    except Exception as e:
        print(f"Callback error: {e}")
        import traceback
        traceback.print_exc()
        
        error_fig = create_error_figure(f"Error: {str(e)}")
        error_summary = create_error_summary(f"An error occurred: {str(e)}")
        return [error_fig] * 5 + [error_summary, error_fig]

# -----------------------------
#  Helper Functions (Visuals)
# -----------------------------

def add_movie_categories(df):
    """Categorize movies based on revenue performance"""
    if df.empty or len(df) == 0:
        return df
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Only calculate percentiles if we have enough data
    if len(df) < 3:
        # If too few movies, just assign all as "Mixed"
        df['Category'] = '📊 Mixed Performance'
        return df
        
    q75 = df['revenue'].quantile(0.75)
    q25 = df['revenue'].quantile(0.25)
    
    categories = []
    for _, row in df.iterrows():
        rev = row['revenue']
        if rev >= q75:
            categories.append('🎯 Box Office Hit')
        elif rev <= q25:
            categories.append('💥 Low Performer')
        else:
            categories.append('📊 Mixed Performance')
    
    df['Category'] = categories
    return df

def create_scatter_plot(df, metric, scale='linear'):
    """Create the main scatter plot"""
    color_map = {
        '🎯 Box Office Hit': '#27ae60',
        '💥 Low Performer': '#e74c3c', 
        '📊 Mixed Performance': '#f39c12'
    }
    
    fig = px.scatter(
        df, 
        x=metric, 
        y='revenue',
        color='Category',
        color_discrete_map=color_map,
        hover_data=['title', 'early_count', 'early_avg', 'total_reviews'],
        title=f"🎬 {metric.replace('_', ' ').title()} vs. Box Office Revenue",
        log_y=(scale == 'log'),
        labels={
            'early_count': 'Early Review Count',
            'early_avg': 'Average Early Rating',
            'revenue': 'Box Office Revenue ($)',
            'total_reviews': 'Total Reviews'
        }
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=16,
        height=550,
        xaxis_title=f"📊 {metric.replace('_', ' ').title()}",
        yaxis_title=f"💰 Box Office Revenue ({'Log Scale' if scale == 'log' else 'Linear Scale'})",
        font=dict(size=12),
        plot_bgcolor='rgba(248,249,250,0.8)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add correlation if we have enough data
    if len(df) >= 3 and not df[metric].isna().all() and df['revenue'].std() > 0:
        try:
            correlation = df[metric].corr(df['revenue'])
            fig.add_annotation(
                text=f"Correlation: {correlation:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        except:
            pass
    
    return fig

def create_metric_histogram(df, metric):
    """Create histogram for the selected metric"""
    color_map = {
        '🎯 Box Office Hit': '#27ae60',
        '💥 Low Performer': '#e74c3c', 
        '📊 Mixed Performance': '#f39c12'
    }
    
    fig = px.histogram(
        df, 
        x=metric,
        color='Category',
        color_discrete_map=color_map,
        title=f"📊 Distribution of {metric.replace('_', ' ').title()}",
        nbins=min(15, len(df)),
        labels={metric: metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=14,
        height=400,
        xaxis_title=f"📊 {metric.replace('_', ' ').title()}",
        yaxis_title="Number of Movies",
        plot_bgcolor='rgba(248,249,250,0.8)',
        showlegend=False
    )
    
    return fig

def create_revenue_histogram(df):
    """Create histogram for revenue distribution"""
    color_map = {
        '🎯 Box Office Hit': '#27ae60',
        '💥 Low Performer': '#e74c3c', 
        '📊 Mixed Performance': '#f39c12'
    }
    
    fig = px.histogram(
        df, 
        x='revenue',
        color='Category',
        color_discrete_map=color_map,
        title="💰 Box Office Revenue Distribution",
        nbins=min(15, len(df)),
        labels={'revenue': 'Box Office Revenue ($)'}
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=14,
        height=400,
        xaxis_title="💰 Box Office Revenue ($)",
        yaxis_title="Number of Movies",
        plot_bgcolor='rgba(248,249,250,0.8)',
        showlegend=False
    )
    
    return fig

def create_rating_distribution(df, reviews):
    """Create box plot showing rating distribution by category"""
    movie_ids = df['movie_id'].unique()
    filtered_reviews = reviews[reviews['movie_id'].isin(movie_ids)].copy()
    
    if filtered_reviews.empty:
        return create_empty_figure("No review ratings available")
    
    # Merge with category information
    filtered_reviews = filtered_reviews.merge(
        df[['movie_id', 'Category']], 
        on='movie_id', 
        how='left'
    )
    
    color_map = {
        '🎯 Box Office Hit': '#27ae60',
        '💥 Low Performer': '#e74c3c', 
        '📊 Mixed Performance': '#f39c12'
    }
    
    fig = px.box(
        filtered_reviews,
        x='Category',
        y='rating',
        color='Category',
        color_discrete_map=color_map,
        title="⭐ Rating Distribution by Performance Category",
        labels={'rating': 'User Rating (1-10)'}
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=14,
        height=400,
        plot_bgcolor='rgba(248,249,250,0.8)',
        showlegend=False,
        xaxis_title=""
    )
    
    return fig

def create_correlation_analysis(df):
    """Create a correlation heatmap for different metrics"""
    # Select numeric columns for correlation
    corr_columns = ['early_count', 'early_avg', 'revenue', 'total_reviews']
    
    # Filter columns that exist and have variation
    available_cols = []
    for col in corr_columns:
        if col in df.columns and df[col].notna().sum() > 1:
            available_cols.append(col)
    
    if len(available_cols) < 2:
        return create_empty_figure("Not enough data for correlation analysis")
    
    # Calculate correlation matrix
    corr_data = df[available_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_data,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=available_cols,
        y=available_cols,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="📊 Correlation Matrix"
    )
    
    # Add text annotations
    fig.update_traces(text=corr_data.round(2), texttemplate='%{text}')
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=14,
        height=400
    )
    
    return fig

def create_top_movies_chart(df, metric):
    """Create a bar chart of top movies by the selected metric"""
    # Sort by the selected metric and take top 15
    top_n = 15
    df_sorted = df.nlargest(min(top_n, len(df)), metric)
    
    if df_sorted.empty:
        return create_empty_figure("No data to display")
    
    color_map = {
        '🎯 Box Office Hit': '#27ae60',
        '💥 Low Performer': '#e74c3c', 
        '📊 Mixed Performance': '#f39c12'
    }
    
    fig = px.bar(
        df_sorted,
        x=metric,
        y='title',
        orientation='h',
        color='Category',
        color_discrete_map=color_map,
        title=f"🏆 Top {len(df_sorted)} Movies by {metric.replace('_', ' ').title()}",
        labels={
            'early_count': 'Early Review Count',
            'early_avg': 'Average Early Rating',
            'title': 'Movie Title'
        }
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=14,
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(248,249,250,0.8)',
        showlegend=False
    )
    
    return fig

def create_data_summary(df, window_days, min_reviews, total_early_reviews, year):
    """Create the data summary component"""
    n_movies = len(df)
    avg_revenue = df['revenue'].mean() if n_movies > 0 else 0
    total_revenue = df['revenue'].sum()
    avg_early_rating = df['early_avg'].mean() if 'early_avg' in df.columns and not df['early_avg'].isna().all() else 0
    
    # Count hits vs flops - check if Category column exists
    if 'Category' in df.columns:
        n_hits = len(df[df['Category'] == '🎯 Box Office Hit'])
        n_low = len(df[df['Category'] == '💥 Low Performer'])
    else:
        n_hits = 0
        n_low = 0
    
    return html.Div([
        html.H4(f"📈 Analysis Summary for {year}", 
                style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H5(f"{n_movies}", 
                        style={'fontSize': '24px', 'color': '#3498db', 'margin': '0'}),
                html.P("Movies Analyzed", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            html.Div([
                html.H5(f"{window_days}", 
                        style={'fontSize': '24px', 'color': '#e74c3c', 'margin': '0'}),
                html.P("Day Buzz Window", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            html.Div([
                html.H5(f"{min_reviews}", 
                        style={'fontSize': '24px', 'color': '#f39c12', 'margin': '0'}),
                html.P("Min Reviews", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            html.Div([
                html.H5(f"{total_early_reviews:,}", 
                        style={'fontSize': '24px', 'color': '#27ae60', 'margin': '0'}),
                html.P("Total Early Reviews", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            html.Div([
                html.H5(f"${avg_revenue/1e6:.1f}M" if avg_revenue > 0 else "$0", 
                        style={'fontSize': '24px', 'color': '#9b59b6', 'margin': '0'}),
                html.P("Avg Revenue", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
            
            html.Div([
                html.H5(f"{avg_early_rating:.1f}/10" if avg_early_rating > 0 else "N/A", 
                        style={'fontSize': '24px', 'color': '#3498db', 'margin': '0'}),
                html.P("Avg Early Rating", 
                       style={'margin': '0', 'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
        
        html.P(
            f"📊 {n_hits} Box Office Hits | {n_low} Low Performers | Total Revenue: ${total_revenue/1e9:.2f}B" 
            if 'Category' in df.columns else 
            f"📊 {n_movies} Movies | Total Revenue: ${total_revenue/1e9:.2f}B", 
            style={'textAlign': 'center', 'marginTop': '10px', 'color': '#7f8c8d'}
        )
    ], style={
        'backgroundColor': '#f8f9fa', 
        'padding': '20px', 
        'borderRadius': '10px',
        'border': '1px solid #e9ecef',
        'maxWidth': '900px',
        'margin': '0 auto'
    })

def create_empty_figure(title, subtitle=""):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"{title}<br><i>{subtitle}</i>",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        height=400, 
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    return fig

def create_error_figure(error_msg):
    """Create an error figure"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ {error_msg}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        height=400, 
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    return fig

def create_error_summary(error_msg):
    """Create an error summary component"""
    return html.Div([
        html.H4("⚠️ Error", style={'color': '#e74c3c'}),
        html.P(error_msg, style={'color': '#e74c3c'})
    ], style={
        'backgroundColor': '#fadbd8', 
        'padding': '15px', 
        'borderRadius': '5px',
        'maxWidth': '600px',
        'margin': '0 auto'
    })

# -------------------
#  Run the App
# -------------------

if __name__ == '__main__':
    print("\n🚀 Dashboard running at: http://127.0.0.1:8050/")
    print("💡 Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)
