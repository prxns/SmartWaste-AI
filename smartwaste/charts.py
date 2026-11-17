"""Interactive charts driven exclusively by calculated or recorded data."""
import plotly.graph_objects as go
from config import MEMBERSHIPS
from smartwaste.fuzzy_engine import membership

COLORS = {"Low": "#829889", "Medium": "#B5A443", "High": "#22694D"}


def style(figure, title, height=330):
    figure.update_layout(title=dict(text=title, font_size=15, x=0.04, y=0.96), height=height,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#18392D", family="Segoe UI, Arial, sans-serif", size=12),
                         margin=dict(l=55, r=25, t=95, b=65),
                         legend=dict(orientation="h", y=1.06, yanchor="bottom", x=0, font_size=11),
                         xaxis=dict(showgrid=False, automargin=True),
                         yaxis=dict(gridcolor="#E4EBDF", automargin=True))
    return figure


def membership_chart(result, title, value):
    figure = go.Figure()
    for name, definition in MEMBERSHIPS.items():
        figure.add_scatter(x=result.universe, y=membership(result.universe, definition),
                           name=name, line=dict(color=COLORS[name], width=2))
    figure.add_vline(x=value, line_dash="dot", line_color="#213B2D")
    figure.update_xaxes(title="Normalized scale", range=[0, 100])
    figure.update_yaxes(title="Membership degree", range=[0, 1.05])
    return style(figure, f"{title} · {value:.1f}")


def aggregate_chart(result):
    figure = go.Figure()
    for name, values in result.outputs.items():
        figure.add_scatter(x=result.universe, y=values, name=name,
                           line=dict(color=COLORS[name], width=1, dash="dot"))
    figure.add_scatter(x=result.universe, y=result.aggregate, fill="tozeroy", name="Aggregated output",
                       line=dict(color="#22694D", width=3), fillcolor="rgba(63,126,80,0.2)")
    figure.add_vline(x=result.score, line_dash="dash", annotation_text=f"Centroid {result.score:.2f}")
    figure.update_xaxes(title="Recycling suitability", range=[0,100])
    figure.update_yaxes(title="Membership degree", range=[0,1.05])
    return style(figure, "Mamdani output and centroid", 340)
