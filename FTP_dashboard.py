# Importing necessary libraries for Dash
import dash
from dash import no_update, dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from dash import callback_context

# External stylesheet for consistent global styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap', 
                        'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Arial&display=swap']

app = dash.Dash('FTP', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Set up cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',  # Simple cache type for local use (in-memory cache)
    'CACHE_DEFAULT_TIMEOUT': 300  # Cache timeout in seconds
})

# Global variables for cached data
cached_df = None
cached_df_teams = None

# Cache the loading of the first dataset (NBA player data)
def load_large_data1():
    global cached_df
    if cached_df is None:  # Check if data is already cached
        print("Loading and caching player data...")
        url = 'https://drive.google.com/uc?export=download&id=1--R70gS8K-ye1LW3kht6_8FM9JmSZXuw'
        cached_df = pd.read_csv(url)
    return cached_df

# Cache the loading of the second dataset (NBA team data)
def load_large_data2():
    global cached_df_teams
    if cached_df_teams is None:  # Check if data is already cached
        print("Loading and caching team data...")
        url2 = 'https://drive.google.com/uc?export=download&id=1o00DWxJAYTU8djknN04qS0eWAXUxH8OX'
        cached_df_teams = pd.read_csv(url2)
    return cached_df_teams

# Loading data using the cached functions
df = load_large_data1()
df_teams = load_large_data2()


# url2 = 'https://drive.google.com/uc?export=download&id=1o00DWxJAYTU8djknN04qS0eWAXUxH8OX'
# df_teams = pd.read_csv(url2)

# Preparing data for the app
seasonal_stats = df.groupby('season').agg({'PTS': 'mean', 'AST': 'mean', 'REB': 'mean', 'STL': 'mean', 'PF': 'mean', 'MIN': 'mean'}).reset_index()

player_stats = df.groupby('player')[['PTS', 'AST', 'REB', 'STL', 'FG%', 'PF', 'MIN']].mean().reset_index()

season_options = [{'label': 'All-Time', 'value': 'All-Time'}] + [{'label': str(season), 'value': season} for season in sorted(df['season'].unique(), reverse=True)]

team_ppg = df_teams.groupby('team')['PTS'].sum() / df_teams.groupby('team')['gameid'].nunique()
team_wins = df_teams[df_teams['win'] == 1].groupby('team').size()
team_playoff_years = df_teams[df_teams['type'] == 'playoff'].groupby('team')['season'].nunique()
team_games = df_teams.groupby('team')['gameid'].nunique()

team_metrics = pd.DataFrame({
    'PPG': team_ppg,
    'Total_Wins': team_wins,
    'Playoff_Qualifications': team_playoff_years,
    'Total_Games': team_games
}).reset_index()

team_metrics['Total_Wins'].fillna(0, inplace=True)
team_metrics['Playoff_Qualifications'].fillna(0, inplace=True)
# Define the landing page layout

# Define the landing page layout with NBA logo and improved styling
landing_page_layout = html.Div([
    html.H1("Welcome to NBA Boxscore Analysis", style={'textAlign': 'center', 'font-size': '70px', 'font-family': 'Montserrat, sans-serif'}),
    html.P("An interactive dashboard for visualizing NBA player and team statistics from 1997 to 2023", 
           style={'font-size': '28px', 'textAlign': 'center', 'margin-bottom': '40px', 'font-family': 'Arial, sans-serif'}),
    
    # NBA Logo (Adjust the path based on your environment)
    html.Div([
        html.Img(src='/assets/nba-logo.png', style={'width': '250px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-bottom': '40px'})
    ]),

    html.Button('START NOW', id='start-button', style={
        'font-size': '30px', 
        # 'padding': '5px 30px', 
        'margin': '0 auto', 
        'display': 'block', 
        'border-radius': '20px',
        'background-color': '#007bff', 
        'color': 'white', 
        'border': 'none', 
        'cursor': 'pointer'
    }),
    html.P("Developed by: Kumar Saurav Jha", style={'textAlign': 'center', 'font-size': '20px', 'margin-top': '30px', 'font-family': 'Arial, sans-serif'}),
], style={
    'backgroundColor': '#E3FDFD', 
    'padding': '50px', 
    'border-radius': '10px', 
    'width': '60%', 
    'margin': '50px auto', 
    'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.1)'
})
# Define the dashboard layout (the tabs section)
dashboard_layout = html.Div([
    # Heading with proper padding and alignment
    html.H1('NBA BOXSCORERS 1997-2023', style={
        'textAlign': 'center', 
        'font-family': 'Montserrat, sans-serif', 
        'font-size': '45px', 
        'font-weight': '700',
        'background-color': '#E3FDFD',
        'padding': '20px',  # Reduced padding for balance
        'margin-bottom': '20px',  # Space between the heading and tabs
        'color': '#333'
    }),

    # Tab container with proper spacing and alignment
    html.Div([
        dcc.Tabs(id='tabs', value='tab-overview', children=[
            dcc.Tab(label='🏀 Player Analysis', value='tab-player', style={
                'font-size': '20px',  # Proper font size
                'background-color': '#E3FDFD',
                'padding': '15px 20px',  # Proper padding for spacing
                'margin-right': '10px',  # Space between tabs
                'vertical-align': 'middle',
                'border-radius': '10px'  # Rounded corners for tabs
            }),
            dcc.Tab(label='📊 Team Analysis', value='tab-team',  style={
                'font-size': '20px',  # Proper font size
                'background-color': '#E3FDFD',
                'padding': '15px 20px',  # Proper padding for spacing
                'margin-right': '10px',  # Space between tabs
                'vertical-align': 'middle',
                'border-radius': '10px'
            }),
            dcc.Tab(label='📈 Overall Trend Analysis', value='tab-overview',  style={
                'font-size': '20px',  # Proper font size
                'background-color': '#E3FDFD',
                'padding': '15px 20px',  # Proper padding for spacing
                'vertical-align': 'middle',
                'border-radius': '10px'  # Rounded corners
            }),
        ], style={
            'display': 'flex',  # Flexbox to align items
            'justify-content': 'center',  # Horizontally align tabs in center
            'align-items': 'center',  # Vertically align tabs
            'height': 'auto',  # Automatic height
            'border-radius': '10px',  # Rounded corners for tab container
            'background-color': '#ffffff',  # Background color for container
            'padding': '10px',  # Padding around the tabs container
            'max-width': '1800px',  # Limit the width of the tabs container
            'margin': '0 auto'  # Center the tabs container
        }),
    ]),

    # Content area with proper padding and spacing
    html.Div(id='tabs-content', style={
        'padding': '40px',  # Padding for breathing space
        'margin': '20px auto',  # Margin around content
        'max-width': '1800px',  # Limit content area width
        'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',  # Subtle shadow
        'border-radius': '10px',  # Rounded corners for content area
        'background-color': '#FFFFFF'  # White background
    }),

    # Button to download the report
    html.Button('Download Report', id='download-btn', n_clicks=0, style={
        'font-size': '20px', 
        'background-color': '#007bff', 
        'color': 'white', 
        'border': 'none', 
        'cursor': 'pointer', 
        'padding': '10px 20px',  # Proper padding for the button
        'border-radius': '10px',  # Rounded corners for the button
        'margin-top': '20px',  # Space above the button
        'display': 'block',  # Block-level button for centering
        'margin-left': 'auto',  # Center the button horizontally
        'margin-right': 'auto'  # Center the button horizontally
    }),

    dcc.Download(id="download-dataframe-csv"),  # Download component
    # Add this to your app layout at the end
    html.Button("Feedback", id="open-feedback-modal", n_clicks=0, style={
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'font-size': '16px',
        'padding': '10px 20px',
        'background-color': '#007bff',
        'color': 'white',
        'border': 'none',
        'border-radius': '5px',
        'cursor': 'pointer',
        'z-index': '1000'
    }),
    # Include this in your layout
    html.Div([
        html.Div([
            html.Div([
                html.H2("Send Feedback", style={'margin-bottom': '20px'}),
                dcc.Textarea(
                    id='feedback-textarea',
                    placeholder='Enter your feedback here...',
                    style={'width': '100%', 'height': '150px', 'font-size': '16px'},
                    maxLength=200
                ),
                html.Div("0/200 characters", id='char-count', style={'textAlign': 'right', 'margin-top': '5px'}),
                html.Button('Send', id='send-feedback', n_clicks=0, style={
                    'margin-top': '20px',
                    'background-color': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'border-radius': '5px',
                    'cursor': 'pointer'
                }),
                html.Button('Cancel', id='close-feedback-modal', n_clicks=0, style={
                    'margin-left': '10px',
                    'margin-top': '20px',
                    'background-color': '#dc3545',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'border-radius': '5px',
                    'cursor': 'pointer'
                }),
            ], style={'padding': '20px'}),
        ], style={
            'position': 'fixed',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'background-color': 'white',
            'width': '400px',
            'box-shadow': '0 5px 15px rgba(0,0,0,.5)',
            'border-radius': '10px',
            'z-index': '1001'
        }),
        html.Div('', style={
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'background-color': 'rgba(0,0,0,0.5)',
            'z-index': '1000'
        })
    ], id='feedback-modal', style={'display': 'none'}),


], style={
    'max-width': '1800px',  # Limit the overall width of the page
    'margin': '0 auto',  # Center the entire layout
    'background-color': '#F0F8FF',  # Light background color
    'padding': '20px 20px',  # Balanced padding around the layout
})


# App layout with a placeholder for dynamic content
app.layout = html.Div([
    html.Div(id='page-content', children=landing_page_layout),
])
# Callback to navigate from landing page to dashboard
@app.callback(
    Output('page-content', 'children'),
    Input('start-button', 'n_clicks')
)
def navigate_to_dashboard(n_clicks):
    if n_clicks is None:
        return landing_page_layout  # Show landing page when no button has been clicked
    else:
        return dashboard_layout  # Show dashboard once the button is clicked


# Create a callback to trigger the download
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks"),
    Input("tabs", "value")],
    prevent_initial_call=True
)
def download_csv(n_clicks, active_tab):
    # Get the context of what triggered the callback
    ctx = callback_context

    # Check if the button was the trigger of the callback
    if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'download-btn.n_clicks':
        raise PreventUpdate  # Prevent download if the button wasn't clicked

    # Generate reports based on the active tab
    if active_tab == 'tab-player':
        # Create a report for Player Analysis
        player_report = df[['player', 'PTS', 'AST', 'REB']]  # Example columns for player analysis
        return dcc.send_data_frame(player_report.to_csv, "nba_player_report.csv")

    elif active_tab == 'tab-team':
        # Use team_metrics instead of df_teams for the report
        team_report = team_metrics[['team', 'PPG', 'Total_Wins', 'Playoff_Qualifications', 'Total_Games']]  # Example columns
        return dcc.send_data_frame(team_report.to_csv, "nba_team_report.csv")

    elif active_tab == 'tab-overview':
        # Create a report for Overall Trend Analysis
        trend_report = seasonal_stats[['season', 'PTS', 'AST', 'REB']]  # Example columns for overall trend analysis
        return dcc.send_data_frame(trend_report.to_csv, "nba_trend_report.csv")

    return None  # Return None if no button click has happened
# Callback to update the tab content
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-overview':
        return html.Div([
            html.H3('Select the metric', style={'padding': '10px 0'}),
            dcc.Dropdown(
                id='metric-select-dropdown',
                options=[
                    {'label': 'Points', 'value': 'PTS'},
                    {'label': 'Assists', 'value': 'AST'},
                    {'label': 'Rebounds', 'value': 'REB'},
                    {'label': 'Steals', 'value': 'STL'},
                    {'label': 'Fouls', 'value': 'PF'},
                    {'label': 'Minutes', 'value': 'MIN'},
                ],
                value=['PTS'],
                clearable=False,
                multi=True,
                style={'width': '50%', 'padding': '10px'}
            ),
            dcc.Graph(id='seasonal-trend-chart', style={'padding': '20px 0'})
        ])
    elif tab == 'tab-player':
        return html.Div([
            dcc.Graph(id='player-stats-bar-chart', style={'padding': '20px 0'}),
            html.Label("Select Season:", style={'font-size': '30px', 'padding': '10px 0'}),
            dcc.Slider(
                id='season-select',
                min=0,
                max=len(season_options) - 1,
                value=0,
                marks={i: {'label': season_options[i]['label'], 'style': {'font-size': '18px'}} for i in range(len(season_options))},
                step=None,
                # tooltip={'always_visible': True}
            ),
            html.Br(),
            html.Label("Filter by Match Type:", style={'font-size': '30px', 'padding': '10px 0'}),
            dcc.RadioItems(
                id='match-type-select',
                options=[
                    {'label': 'All Matches', 'value': 'all'},
                    {'label': 'Regular Season', 'value': 'regular'},
                    {'label': 'Playoffs', 'value': 'playoff'},
                ],
                value='all',
                labelStyle={'display': 'inline-block', 'margin-right': '20px', 'font-size':'18px'},
            ),
            html.Br(),
            html.Label("Select Metric:", style={'font-size': '30px', 'padding': '10px 0'}),
            dcc.Dropdown(
                id='player-stat-select',
                options=[
                    {'label': 'Points per Game', 'value': 'PTS'},
                    {'label': 'Assists per Game', 'value': 'AST'},
                    {'label': 'Rebounds per Game', 'value': 'REB'},
                    {'label': 'Steals per Game', 'value': 'STL'},
                    {'label': 'Accuracy per Game', 'value': 'FG%'},
                    {'label': 'Fouls per Game', 'value': 'PF'},
                    {'label': 'Minutes per Game', 'value': 'MIN'},
                ],
                value='PTS',
                clearable=False,
                style={'width': '50%'}
            )
        ])
    elif tab == 'tab-team':
        return html.Div([
            html.H3('Select 2 teams for comparison', style={'padding': '10px 0'}),
            html.Div([
                dcc.Dropdown(
                    id='team1-select',
                    options=[{'label': team, 'value': team} for team in df_teams['team'].unique()],
                    value='Lakers',
                    style={'width': '48%', 'display': 'inline-block'}
                ),
                dcc.Dropdown(
                    id='team2-select',
                    options=[{'label': team, 'value': team} for team in df_teams['team'].unique()],
                    value='Celtics',
                    style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
                )
            ], style={'padding': '10px 0'}),
            html.H3('Select the metric', style={'padding': '10px 0'}),
            dcc.Dropdown(
                id='metrics-select',
                options=[
                    {'label': 'Points Per Game', 'value': 'PPG'},
                    {'label': 'Total Number of Wins', 'value': 'Total_Wins'},
                    {'label': 'Number of Playoff Qualifications', 'value': 'Playoff_Qualifications'},
                    {'label': 'Number of Games Played', 'value': 'Total_Games'},
                ],
                value=['PPG'],
                multi=True,
                style={'width': '50%', 'padding': '10px 0'}
            ),
            dcc.Graph(id='comparison-chart', style={'padding': '20px 0'})
        ])

# Callback for updating the trend chart
@app.callback(
    Output('seasonal-trend-chart', 'figure'),
    Input('metric-select-dropdown', 'value'))
def update_chart(selected_metrics):
    return create_multi_metric_trend(seasonal_stats, selected_metrics)

# Callback for updating the player stats chart
@app.callback(
    Output('player-stats-bar-chart', 'figure'),
    [Input('player-stat-select', 'value'),
     Input('season-select', 'value'),
     Input('match-type-select', 'value')]
)
def update_player_stats_chart(selected_stat, slider_value, match_type):
    selected_season = season_options[slider_value]['value']
    if selected_season == 'All-Time':
        # Filter dataframe based on match type
        if match_type != 'all':
            filtered_df = df[(df['type'] == match_type)].groupby(['player', 'team'])[selected_stat].mean().reset_index().sort_values(selected_stat, ascending=False)
        else:
            filtered_df = df.groupby(['player', 'team'])[selected_stat].mean().reset_index().sort_values(selected_stat, ascending=False)
    else:
        # Filter dataframe for selected season and match type
        if match_type != 'all':
            filtered_df = df[(df['season'] == selected_season) & (df['type'] == match_type)].groupby(['player', 'team'])[selected_stat].mean().reset_index().sort_values(selected_stat, ascending=False)
        else:
            filtered_df = df[(df['season'] == selected_season)].groupby(['player', 'team'])[selected_stat].mean().reset_index().sort_values(selected_stat, ascending=False)

    fig = px.bar(
        filtered_df.head(10),  # Show top 10 for clarity
        y='player',
        x=selected_stat,
        color='team',
        title=f'Top 10 Players by {selected_stat} for {selected_season}',
        labels={'player': 'Player', selected_stat: f'{selected_stat} per Game'}
    )
    
    # Sort y-axis in descending order
    fig.update_layout(
        title_font=dict(size=30),  # Larger title font
        font=dict(size=22),  # Larger global font
        xaxis=dict(
            tickfont=dict(size=20)  # Larger x-axis tick labels
        ),
        yaxis=dict(
            tickfont=dict(size=20),  # Larger y-axis tick labels
            categoryorder='total ascending'  # Sorting categories in ascending order based on the total
        ),
        legend=dict(
            font=dict(size=18)  # Larger legend font
        ),
        margin=dict(l=40, r=40, t=80, b=40)  # Adjust margins for better readability
    )


    return fig

# Callback for updating team comparison chart
@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('team1-select', 'value'),
     Input('team2-select', 'value'),
     Input('metrics-select', 'value')]
)
def update_comparison_chart(team1, team2, metrics):
    if not metrics:
        raise PreventUpdate("No metrics selected")
    team1_data = team_metrics[team_metrics['team'] == team1]
    team2_data = team_metrics[team_metrics['team'] == team2]
    traces = []
    for metric in metrics:
        traces.append(go.Bar(name=f'{team1} - {metric}', x=[metric], y=team1_data[metric], marker={'color': '#1f77b4'}))
        traces.append(go.Bar(name=f'{team2} - {metric}', x=[metric], y=team2_data[metric], marker={'color': '#ff7f0e'}))
    fig = go.Figure(data=traces)
    fig.update_layout(barmode='group', title=f'Comparison of {team1} and {team2}', yaxis_title='Value', xaxis_title='Metrics')
    return fig

# Helper function to create multi-metric trend graphs
def create_multi_metric_trend(df, metrics):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=df['season'],
            y=df[metric],
            name=metric,
            mode='lines',
            line=dict(color=colors[i % len(colors)])
        ))

    # Update the layout with larger fonts
    fig.update_layout(
        title='Seasonal Trends for Multiple Metrics',
        title_font=dict(size=30, family='Arial, sans-serif', color='#333'),  # Larger title font
        font=dict(size=22, family='Arial, sans-serif', color='#333'),  # Larger global font size
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='lightgrey',
            tickfont=dict(size=20),  # Larger x-axis tick font
            title_font=dict(size=24)  # Larger x-axis title font
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='lightgrey',
            tickfont=dict(size=20),  # Larger y-axis tick font
            title_font=dict(size=24)  # Larger y-axis title font
        ),
        legend_title_font=dict(size=20),  # Larger legend title
        legend_font=dict(size=18),  # Larger legend items font
        margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins
        hovermode='closest'
    )
    return fig

# from dash.dependencies import Input, Output, State

# @app.callback(
#     Output('feedback-modal', 'style'),
#     [Input('open-feedback-modal', 'n_clicks'),
#      Input('close-feedback-modal', 'n_clicks'),
#      Input('send-feedback', 'n_clicks')],
#     [State('feedback-modal', 'style')]
# )
# def toggle_modal(open_clicks, close_clicks, send_clicks, current_style):
#     ctx = dash.callback_context

#     if not ctx.triggered:
#         return {'display': 'none'}
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#         if button_id == 'open-feedback-modal':
#             return {'display': 'block'}
#         else:
#             return {'display': 'none'}

# @app.callback(
#     Output('char-count', 'children'),
#     Input('feedback-textarea', 'value')
# )
# def update_char_count(text):
#     if text:
#         return f"{len(text)}/200 characters"
#     else:
#         return "0/200 characters"


# import os

# Set these environment variables in your system or a .env file
# SENDER_EMAIL = os.environ.get('kumarsaurav.jha@gwu.edu')
# EMAIL_PASSWORD = os.environ.get('University2k23')
# RECEIVER_EMAIL = os.environ.get('kumarsaurav.jha@gwu.edu')  # Your email address

SENDER_EMAIL = 'jha.saurav919@gmail.com'
EMAIL_PASSWORD = 'cskixlyryxfudgfs'
RECEIVER_EMAIL = 'kumarsaurav.jha@gwu.edu'

import smtplib
from email.mime.text import MIMEText

# @app.callback(
#     [Output('feedback-textarea', 'value'),     # Clear the feedback textarea
#      Output('feedback-modal', 'style'),        # Close the modal
#      Output('char-count', 'children')],        # Show confirmation message
#     Input('send-feedback', 'n_clicks'),
#     State('feedback-textarea', 'value')
# )
# def send_feedback(n_clicks, feedback_text):
#     if n_clicks and feedback_text:
#         # Email configuration
#         sender_email = SENDER_EMAIL
#         receiver_email = RECEIVER_EMAIL
#         password = EMAIL_PASSWORD

#         # Create the email content
#         msg = MIMEText(feedback_text)
#         msg['Subject'] = 'Feedback from NBA Dashboard'
#         msg['From'] = sender_email
#         msg['To'] = receiver_email

#         # Send the email
#         try:
#             with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#                 server.login(sender_email, password)
#                 server.sendmail(sender_email, receiver_email, msg.as_string())
#             print('Feedback sent successfully!')
#             # After successfully sending the email, return a message and close the modal
#             return '', {'display': 'none'}, "Thank you for your feedback!" 
#         except Exception as e:
#             print(f"Error sending email: {e}")
#             return dash.no_update, dash.no_update

#         # Clear the textarea after sending
#         # return ''
#     else:
#         return dash.no_update, dash.no_update

from dash.dependencies import Input, Output, State
import dash

@app.callback(
    [Output('feedback-textarea', 'value'),     # Clear the feedback textarea
     Output('feedback-modal', 'style'),        # Close the modal
     Output('char-count', 'children')],        # Show confirmation message
    [Input('send-feedback', 'n_clicks'),       # Send button clicked
     Input('open-feedback-modal', 'n_clicks'), # Open modal button clicked
     Input('close-feedback-modal', 'n_clicks')],  # Close modal button clicked
    [State('feedback-textarea', 'value'),      # The feedback text
     State('feedback-modal', 'style')]         # Current modal style (open/close state)
)
def handle_feedback(send_clicks, open_clicks, close_clicks, feedback_text, current_style):
    ctx = dash.callback_context  # Get the context to see which button triggered the callback

    if not ctx.triggered:
        return dash.no_update, {'display': 'none'}, "0/200 characters"  # Default state when nothing is clicked

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'open-feedback-modal':
        # Open the feedback modal
        return dash.no_update, {'display': 'block'}, dash.no_update

    elif button_id == 'close-feedback-modal':
        # Close the feedback modal
        return dash.no_update, {'display': 'none'}, dash.no_update

    elif button_id == 'send-feedback' and feedback_text:
        # Handle sending feedback via email
        sender_email = SENDER_EMAIL
        receiver_email = RECEIVER_EMAIL
        password = EMAIL_PASSWORD

        # Create the email content
        msg = MIMEText(feedback_text)
        msg['Subject'] = 'Feedback from NBA Dashboard'
        msg['From'] = sender_email
        msg['To'] = receiver_email

        try:
            # Send the email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
            print('Feedback sent successfully!')
            return '', {'display': 'none'}, "Thank you for your feedback!"
        except Exception as e:
            print(f"Error sending email: {e}")
            return dash.no_update, dash.no_update, "Error sending feedback. Please try again."

    return dash.no_update, dash.no_update, "0/200 characters"




@app.callback(
    Output('send-feedback', 'disabled'),
    [Input('send-feedback', 'n_clicks')]
)
def disable_button_after_click(n_clicks):
    if n_clicks:
        return True
    return False


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
