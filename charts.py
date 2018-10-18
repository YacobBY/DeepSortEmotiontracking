import webbrowser
# Import Supporting Libraries
import pandas as pd

# Import Dash Visualization Libraries
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash.dependencies
from dash.dependencies import Input, Output, State
import plotly

# Load datasets
US_STATES_URL = 'templates/test1.csv'
US_AG_URL = 'templates/test2.csv'

df_ag = pd.read_csv(US_AG_URL)

print(df_ag.head())

# Create our app layout
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H2('My Dash App'),
    dt.DataTable(
        id='my-datatable',
        rows=df_ag.to_dict('records'),
        editable=False,
        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[]
    ),
    html.Div(id='selected-indexes'),
    dcc.Graph(
        id='datatable-subplots'
    )
], style={'width': '90%'})


# filename = 'Fig1.html'
@app.callback(Output('datatable-subplots', 'figure'),
              [Input('my-datatable', 'rows'),
               Input('my-datatable', 'selected_row_indices')])
def update_figure(rows, selected_row_indices):
    dff = pd.DataFrame(rows)
    fig = plotly.tools.make_subplots(
        rows=3, cols=1,
        subplot_titles=('Matties', 'Aantal', 'IDB'),
        shared_xaxes=True)
    marker = {'color': ['#0074D9'] * len(dff)}
    for i in (selected_row_indices or []):
        marker['color'][i] = '#FF851B'
    fig.append_trace({
        'x': dff['Aantal'],
        'y': dff['Matties'],
        'type': 'bar',
        'marker': marker
    }, 1, 1)
    fig.append_trace({
        'x': dff['Aantal'],
        'y': dff['Indebuurt'],
        'type': 'bar',
        'marker': marker
    }, 2, 1)
    fig.append_trace({
        'x': dff['Aantal'],
        'y': dff['Indebuurt'],
        'type': 'bar',
        'marker': marker
    }, 3, 1)
    fig['layout']['showlegend'] = False
    fig['layout']['height'] = 1000
    fig['layout']['width'] = 1200
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
    }
    return fig


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
