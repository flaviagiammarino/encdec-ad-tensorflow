import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(xn, xa, rn, ra, an, aa, tau):
    '''
    Plot the results.
    '''

    if xn.shape[1] == xa.shape[1] == rn.shape[1] == ra.shape[1]:
        m = xn.shape[1]
    else:
        raise ValueError('Found inconsistent number of time series.')
    
    fig = make_subplots(
        subplot_titles=[f'Time Series {i} - Actual (Normal)' for i in range(1, m + 1)] +
                       [f'Time Series {i} - Anomaly Score (Normal)' for i in range(1, m + 1)] +
                       [f'Time Series {i} - Reconstructed (Normal)' for i in range(1, m + 1)] +
                       [f'Time Series {i} - Actual (Anomalous)' for i in range(1, m + 1)] +
                       [f'Time Series {i} - Anomaly Score (Anomalous)' for i in range(1, m + 1)] +
                       [f'Time Series {i} - Reconstructed (Anomalous)' for i in range(1, m + 1)],
        vertical_spacing=0.075,
        rows=6,
        cols=m
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
            size=10,
        )
    )
    
    for i in range(m):
        
        fig.add_trace(
            go.Scatter(
                y=xn[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(175,184,193,1)',
                    width=0.5
                )
            ),
            row=1,
            col=1 + i
        )

        fig.add_trace(
            go.Scatter(
                y=an,
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(207, 34, 46,1)',
                    width=0.5
                )
            ),
            row=2,
            col=1 + i
        )
        
        fig.add_hline(
            y=tau,
            line=dict(
                dash='dot',
                color='#000000',
                width=0.5,
            ),
            row=2,
            col=1 + i
        )
        
        fig.add_trace(
            go.Scatter(
                y=rn[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(9, 105, 218,1)',
                    width=0.5
                )
            ),
            row=3,
            col=1 + i
        )
        
        fig.add_trace(
            go.Scatter(
                y=xa[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(175,184,193,1)',
                    width=0.5
                )
            ),
            row=4,
            col=1 + i
        )
    
        fig.add_trace(
            go.Scatter(
                y=aa,
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(207, 34, 46,1)',
                    width=0.5
                )
            ),
            row=5,
            col=1 + i
        )
        
        fig.add_hline(
            y=tau,
            line=dict(
                dash='dot',
                color='#000000',
                width=0.5,
            ),
            row=5,
            col=1 + i
        )
        
        fig.add_trace(
            go.Scatter(
                y=ra[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(9, 105, 218,1)',
                    width=0.5
                )
            ),
            row=6,
            col=1 + i
        )
        
        for row in range(1, 7):
            
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
                row=row,
                col=1 + i
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
                row=row,
                col=1 + i
            )
    
    return fig
