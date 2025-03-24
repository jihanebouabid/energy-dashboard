import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load datasets

def load_predictions(model_name):
    file_mapping = {
        "Gradient Boosting": "predictions_gradient_boosting.csv",
        "Random Forest": "predictions_random_forest.csv",
        "Neural Network": "predictions_neural_network.csv",
        "LSTM": "predictions_lstm.csv"
    }
    return pd.read_csv(file_mapping[model_name])

def load_2019_predictions():
    try:
        df = pd.read_csv("predictions_2019_neural_network.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Actual", "Predicted"])
        return df
    except Exception as e:
        print("⚠️ Error loading 2019 predictions:", e)
        return pd.DataFrame(columns=["Date", "Actual", "Predicted"])

def load_metrics():
    with open("model_metrics.json", "r") as f:
        return json.load(f)

metrics = load_metrics()

def load_outlier_data():
    df = pd.read_csv("merged_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def load_final_merged_data():
    df = pd.read_csv("final_merged_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"])
app.title = "Energy Dashboard"

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='main-dashboard', children=[
        dcc.Tab(label='Prediction Dashboard', value='main-dashboard'),
        dcc.Tab(label='Data and Feature Analysis', value='outlier-analysis')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'main-dashboard':
        df_2019 = load_2019_predictions()
        if df_2019.empty:
            fig_2019 = go.Figure()
            fig_2019.add_annotation(text="No 2019 predictions available.",
                                    xref="paper", yref="paper", showarrow=False,
                                    font=dict(size=20))
            fig_2019_scatter = go.Figure()
            fig_2019_metrics = go.Figure()
        else:
            fig_2019 = px.line(df_2019, x='Date', y=['Actual', 'Predicted'],
                               title="Neural Network Predictions - 2019",
                               labels={"value": "Power Consumption (kW)", "Date": "Date"})

            fig_2019_scatter = px.scatter(df_2019, x='Actual', y='Predicted',
                                          title="2019 Scatter Plot: Actual vs Predicted",
                                          labels={"Actual": "Actual Power (kW)", "Predicted": "Predicted Power (kW)"},
                                          trendline="ols")

            # Compute metrics
            mae = np.mean(np.abs(df_2019['Actual'] - df_2019['Predicted']))
            mse = np.mean((df_2019['Actual'] - df_2019['Predicted']) ** 2)
            rmse = np.sqrt(mse)
            cv_rmse = rmse / np.mean(df_2019['Actual']) * 100
            r2 = 1 - (np.sum((df_2019['Actual'] - df_2019['Predicted']) ** 2) / np.sum((df_2019['Actual'] - np.mean(df_2019['Actual'])) ** 2))

            metrics_2019 = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "CV(RMSE) (%)": cv_rmse,
                "R²": r2
            }
            metrics_df_2019 = pd.DataFrame({"Metric": list(metrics_2019.keys()), "Value": list(metrics_2019.values())})

            fig_2019_metrics = px.bar(metrics_df_2019, x='Metric', y='Value',
                                      title="Neural Network Performance Metrics - 2019",
                                      labels={"Value": "Metric Value", "Metric": "Metrics"},
                                      color='Metric')

        return html.Div([
            html.H1("Energy Consumption Dashboard", className="text-center mt-4 mb-4"),

            html.Div([
                dcc.Dropdown(
                    id='model-selector',
                    options=[
                        {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
                        {'label': 'Random Forest', 'value': 'Random Forest'},
                        {'label': 'Neural Network', 'value': 'Neural Network'},
                        {'label': 'LSTM', 'value': 'LSTM'}
                    ],
                    value='Gradient Boosting',
                    clearable=False,
                    className="form-select"
                )
            ], className="container mb-4"),

            html.Div([
                html.Div(dcc.Graph(id='prediction-graph'), className="card p-3 mb-4 shadow"),
                html.Div(dcc.Graph(id='scatter-plot'), className="card p-3 mb-4 shadow"),
                html.Div(dcc.Graph(id='metrics-histogram'), className="card p-3 mb-4 shadow"),
            ], className="container"),

            html.Div([
                dash_table.DataTable(
                    id='data-table',
                    columns=[
                        {'name': 'Date', 'id': 'Date'},
                        {'name': 'Actual', 'id': 'Actual'},
                        {'name': 'Predicted', 'id': 'Predicted'}
                    ],
                    page_size=10,
                    style_table={'overflowX': 'auto'}
                )
            ], className="container card p-3 shadow mb-5"),

            html.Div([
                html.H3("Neural Network Prediction for 2019", className="text-center mb-3 mt-5"),
                dcc.Graph(figure=fig_2019),
                dcc.Graph(figure=fig_2019_scatter),
                dcc.Graph(figure=fig_2019_metrics)
            ], className="container card p-4 shadow mb-5")
        ])

    elif tab == 'outlier-analysis':
        df = load_outlier_data()
        df_final = load_final_merged_data()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        final_line_fig = px.line(df_final.set_index("timestamp"),
                                 title="Final Cleaned Data After Resampling & Filling Missing Values",
                                 labels={"value": "Power (kW) and Other Features", "timestamp": "Timestamp"})

        corr_matrix = df_final.corr()
        heatmap_fig = px.imshow(corr_matrix,
                                labels=dict(color="Correlation"),
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                color_continuous_scale="RdBu",
                                title="Feature Correlation Heatmap")

        return html.Div([
            html.H2("Outlier Detection using Z-Score", className="text-center mt-4 mb-4"),
            html.Div([
                dcc.Dropdown(
                    id='column-selector',
                    options=[{'label': col, 'value': col} for col in numeric_cols],
                    value=numeric_cols[0],
                    clearable=False,
                    className="form-select mb-4"
                ),
                dcc.Graph(id='outlier-graph')
            ], className="container"),

            html.H2("Feature Exploration", className="text-center mt-5 mb-4"),
            html.Div([
                dcc.Graph(figure=final_line_fig, className="mb-5")
            ], className="container"),

            html.H2("Feature Correlation", className="text-center mt-5 mb-4"),
            html.Div([
                dcc.Graph(figure=heatmap_fig, className="mb-5")
            ], className="container")
        ])

@app.callback(
    [Output('prediction-graph', 'figure'), Output('data-table', 'data'), Output('scatter-plot', 'figure'), Output('metrics-histogram', 'figure')],
    [Input('model-selector', 'value')]
)
def update_dashboard(selected_model):
    df = load_predictions(selected_model)

    fig_line = px.line(df, x='Date', y=['Actual', 'Predicted'],
                       title=f"{selected_model} Predictions",
                       labels={"value": "Power Consumption (kW)", "variable": "Legend"})

    fig_scatter = px.scatter(df, x='Actual', y='Predicted',
                             title=f"{selected_model} Scatter Plot: Actual vs Predicted",
                             labels={"Actual": "Actual Power (kW)", "Predicted": "Predicted Power (kW)"},
                             trendline="ols")

    model_metrics = metrics[selected_model]
    if "MAPE (%)" in model_metrics:
        del model_metrics["MAPE (%)"]

    metrics_df = pd.DataFrame({"Metric": list(model_metrics.keys()), "Value": list(model_metrics.values())})

    fig_histogram = px.bar(metrics_df, x='Metric', y='Value',
                           title=f"{selected_model} Performance Metrics",
                           labels={"Value": "Metric Value", "Metric": "Metrics"},
                           color='Metric')

    return fig_line, df.to_dict('records'), fig_scatter, fig_histogram

@app.callback(
    Output('outlier-graph', 'figure'),
    Input('column-selector', 'value')
)
def update_outlier_plot(selected_col):
    df = load_outlier_data()
    z_scores = (df[selected_col] - df[selected_col].mean()) / df[selected_col].std()
    outliers = df[np.abs(z_scores) > 3]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Boxplot of {selected_col}", f"Z-score Outliers in {selected_col}"))

    fig.add_trace(go.Box(y=df[selected_col], name=selected_col, boxpoints='outliers', marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[selected_col], mode='lines', name='Data', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=outliers["timestamp"], y=outliers[selected_col], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=2)

    fig.update_layout(title_text=f"Outlier Detection - {selected_col}", showlegend=True, height=500)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

server = app.server

