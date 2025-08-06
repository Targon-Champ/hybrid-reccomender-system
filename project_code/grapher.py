import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class Grapher:
    def __init__(self):
        pass
    
    def bar_graph(self, x, y, title="Bar Graph", xlabel="X-axis", ylabel="Y-axis"):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x, y=y, palette='coolwarm')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def pie_chart(self, labels, sizes, title="Pie Chart"):
        total = sum(sizes)
        percentages = [(s / total) * 100 for s in sizes]
        cmap = cm.get_cmap('coolwarm', len(sizes))
        colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b, _ in cmap(np.linspace(0, 1, len(sizes)))]
        hover_texts = [f'{label}: {size:.1f}% (Index {i})' for i, (label, size) in enumerate(zip(labels, percentages))]
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            textinfo='percent',
            hoverinfo='label+percent+value',
            hovertext=hover_texts,
            marker=dict(colors=colors),
            showlegend=True
        )])
        base_size = 500
        extra_per_label = 25
        dimensions = base_size + extra_per_label * len(labels)
        fig.update_layout(
            width=dimensions,
            height=dimensions,
            title=title,
            legend_title="Indexes",
            margin=dict(l=40, r=40, t=80, b=40),
        )


        fig.show()

    def heatmap(self, data, title="Heatmap", cmap="viridis"):
        plt.figure(figsize=(10, 8), facecolor='white')
        sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap)
        plt.title(title)
        plt.show()

    def line_graph(self, x, y, title="Line Graph", xlabel="X-axis", ylabel="Y-axis"):
        plt.figure(figsize=(10, 6), facecolor='white')
        plt.plot(x, y, marker='o', color='b')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()