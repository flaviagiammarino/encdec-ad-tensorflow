import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(x, y, s, quantile):
    
    '''
    Plot the actual and reconstructed values of the time series and annotate the anomalies.

    Parameters:
    __________________________________
    x: np.array.
        Actual time series, array with shape (samples, features) where samples is the length
        of the time series and features is the number of time series.

    y: np.array.
        Reconstructed time series, array with shape (samples, features) where samples is the
        length of the time series and features is the number of time series.

    s: np.array.
        Anomaly scores, array with shape (samples,) where samples is the length of the time
        series.

    quantile: float.
        Quantile of anomaly score used for identifying the anomalies.

    Returns:
    __________________________________
    fig: go.Figure.
        Line charts of actual and reconstructed values with annotated anomalies,
        two subplots for each time series.
    '''

    if x.shape[1] == y.shape[1]:
        features = x.shape[1]
    else:
        raise ValueError(f'Expected {x.shape[1]} features, found {y.shape[1]}.')

    fig = make_subplots(
        subplot_titles=['Feature ' + str(i + 1) + ' ' + s for i in range(features) for s in ['(Actual)', '(Reconstructed)']],
        specs=[[{'secondary_y': True}], [{'secondary_y': False}]] * features,
        vertical_spacing=0.1,
        rows=2 * features,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
        legend=dict(
            font=dict(
                color='#000000',
            ),
        ),
    )

    fig.update_annotations(
        font=dict(
            size=13
        )
    )

    rows = [1, 2]
    
    for i in range(features):
        
        fig.add_trace(
            go.Scatter(
                y=np.where(s > np.quantile(s, quantile), 1, 0),
                showlegend=True if i ==0 else False,
                name='Anomaly',
                legendgroup='Anomaly',
                mode='lines',
                fillcolor='#e6edf6',
                fill='tozeroy',
                line=dict(
                    width=0
                )
            ),
            secondary_y=False,
            row=rows[0],
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=x[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='#b3b3b3',
                    width=2
                )
            ),
            secondary_y=True,
            row=rows[0],
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=y[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='#0550ae',
                    width=2
                )
            ),
            row=rows[1],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=rows[0],
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            secondary_y=True,
            side='left',
            row=rows[0],
            col=1
        )

        fig.update_yaxes(
            range=[0, 1],
            showticklabels=False,
            mirror=False,
            showgrid=False,
            zeroline=False,
            secondary_y=False,
            side='right',
            row=rows[0],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=rows[1],
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            side='left',
            row=rows[1],
            col=1
        )

        rows[0] += 2
        rows[1] += 2
    
    return fig