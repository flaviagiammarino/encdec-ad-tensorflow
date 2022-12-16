import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(x, x_hat, scores, quantile):
    
    '''
    Plot the actual and reconstructed values of the time series and annotate the anomalies.

    Parameters:
    __________________________________
    x: np.array.
        Actual time series, array with shape (samples, features) where samples is the length
        of the time series and features is the number of time series.

    x_hat: np.array.
        Reconstructed time series, array with shape (samples, features) where samples is the
        length of the time series and features is the number of time series.

    scores: np.array.
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

    if x.shape[1] == x_hat.shape[1]:
        features = x.shape[1]
    else:
        raise ValueError(f'Expected {x.shape[1]} features, found {y.shape[1]}.')

    fig = make_subplots(
        subplot_titles=['Feature ' + str(i + 1) + ' ' + s for i in range(features) for s in ['(Actual)', '(Reconstructed)']],
        specs=[[{'secondary_y': True}], [{'secondary_y': False}]] * features,
        vertical_spacing=0.125,
        rows=2 * features,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )

    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )

    rows = [1, 2]
    
    for i in range(features):
        
        fig.add_trace(
            go.Scatter(
                y=x[:, i],
                showlegend=True if i == 0 else False,
                name='Actual',
                mode='lines',
                line=dict(
                    color='#afb8c1',
                    width=1
                )
            ),
            row=rows[0],
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=x_hat[:, i],
                showlegend=True if i == 0 else False,
                name='Reconstructed',
                mode='lines',
                line=dict(
                    color='#0969da',
                    width=1
                )
            ),
            row=rows[1],
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=np.where(scores > np.quantile(scores, quantile), x[:, i], np.nan),
                showlegend=True if i == 0 else False,
                name='Anomaly',
                legendgroup='Anomaly',
                mode='markers',
                marker=dict(
                    color='#cf222e',
                    size=3,
                    line=dict(
                        width=0
                    )
                )
            ),
            row=rows[0],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=rows[0],
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[0],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=rows[1],
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[1],
            col=1
        )

        rows[0] += 2
        rows[1] += 2
    
    return fig