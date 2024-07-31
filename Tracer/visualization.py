# pytrace/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd  # Ensure you have pandas installed if using interactive scatter plot

# -----------------------------------------------------------------------------
# Memory Related Graphs
# -----------------------------------------------------------------------------

def plot_memory_usage(memory_data):
    """
    Plots memory usage data as a bar chart.

    Parameters:
    - memory_data (dict): A dictionary where keys are memory regions and values are their sizes.
    """
    regions = list(memory_data.keys())
    sizes = list(memory_data.values())
    
    plt.bar(regions, sizes, color='skyblue')
    plt.xlabel('Memory Region')
    plt.ylabel('Size (bytes)')
    plt.title('Memory Usage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_memory_pie_chart(memory_data):
    """
    Plots a pie chart of memory usage.

    Parameters:
    - memory_data (dict): A dictionary where keys are memory regions and values are their sizes.
    """
    regions = list(memory_data.keys())
    sizes = list(memory_data.values())

    plt.pie(sizes, labels=regions, autopct='%1.1f%%', colors=plt.get_cmap('tab20').colors)
    plt.title('Memory Usage Distribution')
    plt.show()

def plot_object_size_heatmap(object_sizes):
    """
    Plots a heatmap of object sizes.

    Parameters:
    - object_sizes (dict): A dictionary where keys are object labels and values are their sizes.
    """
    labels = list(object_sizes.keys())
    sizes = list(object_sizes.values())

    size_matrix = np.array(sizes).reshape(-1, 1)  # Reshape for a 1-column heatmap

    plt.imshow(size_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Size (bytes)')
    plt.xticks([])
    plt.yticks(range(len(labels)), labels)
    plt.title('Object Size Heatmap')
    plt.show()

def plot_object_size_distribution(object_sizes):
    """
    Plots a histogram of object sizes.

    Parameters:
    - object_sizes (dict): A dictionary where keys are object labels and values are their sizes.
    """
    sizes = list(object_sizes.values())

    plt.hist(sizes, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Size (bytes)')
    plt.ylabel('Frequency')
    plt.title('Object Size Distribution')
    plt.grid(True)
    plt.show()

def plot_box_plot(object_sizes):
    """
    Plots a box plot of object sizes.

    Parameters:
    - object_sizes (dict): A dictionary where keys are object labels and values are their sizes.
    """
    sizes = list(object_sizes.values())

    plt.boxplot(sizes)
    plt.ylabel('Size (bytes)')
    plt.title('Object Size Distribution (Box Plot)')
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
# Reference Related Graphs
# -----------------------------------------------------------------------------

def plot_reference_graph(reference_data):
    """
    Plots a reference graph using networkx.

    Parameters:
    - reference_data (dict): A dictionary where keys are object labels and values are lists of referenced object labels.
    """
    G = nx.DiGraph()
    
    for obj, refs in reference_data.items():
        for ref in refs:
            G.add_edge(obj, ref)
    
    pos = nx.spring_layout(G, seed=42)  # Layout for the nodes
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title("Reference Graph")
    plt.show()

def plot_reference_count_histogram(reference_data):
    """
    Plots a histogram of reference counts.

    Parameters:
    - reference_data (dict): A dictionary where keys are object labels and values are lists of referenced object labels.
    """
    reference_counts = [len(refs) for refs in reference_data.values()]

    plt.hist(reference_counts, bins=range(max(reference_counts) + 2), edgecolor='black', alpha=0.7)
    plt.xlabel('Number of References')
    plt.ylabel('Frequency')
    plt.title('Reference Count Distribution')
    plt.grid(True)
    plt.show()

def plot_object_reference_matrix(reference_data):
    """
    Plots a matrix indicating object references.

    Parameters:
    - reference_data (dict): A dictionary where keys are object labels and values are lists of referenced object labels.
    """
    objects = list(reference_data.keys())
    matrix = np.zeros((len(objects), len(objects)), dtype=int)
    obj_index = {obj: idx for idx, obj in enumerate(objects)}

    for obj, refs in reference_data.items():
        row = obj_index[obj]
        for ref in refs:
            col = obj_index.get(ref, -1)
            if col != -1:
                matrix[row, col] = 1

    plt.imshow(matrix, cmap='Greys', aspect='auto')
    plt.colorbar(label='Reference Presence')
    plt.xticks(range(len(objects)), objects, rotation=90)
    plt.yticks(range(len(objects)), objects)
    plt.title('Object Reference Matrix')
    plt.show()

# -----------------------------------------------------------------------------
# Tree Related Graphs
# -----------------------------------------------------------------------------

def plot_tree(tree_data, root_label="Root"):
    """
    Plots a tree structure using networkx and matplotlib.

    Parameters:
    - tree_data (dict): A dictionary where keys are node labels and values are lists of child node labels.
    - root_label (str): The label of the root node.
    """
    G = nx.DiGraph()
    
    def add_edges(node, parent=None):
        if parent:
            G.add_edge(parent, node)
        for child in tree_data.get(node, []):
            add_edges(child, node)
    
    add_edges(root_label)
    
    pos = nx.spring_layout(G, seed=42)  # Layout for the nodes
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title("Tree Visualization")
    plt.show()

def plot_interactive_tree(tree_data, root_label="Root"):
    """
    Plots an interactive tree structure using Plotly.

    Parameters:
    - tree_data (dict): A dictionary where keys are node labels and values are lists of child node labels.
    - root_label (str): The label of the root node.
    """
    edges = []
    labels = [root_label]
    
    def add_edges(node, parent=None):
        if parent:
            edges.append((parent, node))
        for child in tree_data.get(node, []):
            labels.append(child)
            add_edges(child, node)
    
    add_edges(root_label)
    
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[0, 0],
            mode='lines+markers',
            line=dict(width=1, color='black'),
            marker=dict(size=10),
            text=[edge[0], edge[1]],
            textposition='bottom center'
        ))
    
    # Add nodes
    for label in labels:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers+text',
            text=[label],
            textposition='top center',
            marker=dict(size=20)
        ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    fig.show()

# -----------------------------------------------------------------------------
# Utils Related Graphs
# -----------------------------------------------------------------------------

def plot_circular_graph(network_data):
    """
    Plots a circular network graph using networkx.

    Parameters:
    - network_data (dict): A dictionary where keys are node labels and values are lists of connected node labels.
    """
    G = nx.Graph()
    
    for node, neighbors in network_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    plt.title('Circular Network Graph')
    plt.show()

def plot_graph_with_layout(network_data, layout='spring'):
    """
    Plots a network graph with various layout options using networkx.

    Parameters:
    - network_data (dict): A dictionary where keys are node labels and values are lists of connected node labels.
    - layout (str): Layout algorithm to use ('spring', 'kamada_kaway', 'spectral', etc.).
    """
    G = nx.Graph()
    
    for node, neighbors in network_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    if layout == 'kamada_kaway':
        pos = nx.kamada_kaway_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    plt.title(f'Network Graph ({layout.capitalize()} Layout)')
    plt.show()

def plot_graph_with_node_attributes(network_data, node_attributes, attribute='size'):
    """
    Plots a network graph with node attributes using networkx.

    Parameters:
    - network_data (dict): A dictionary where keys are node labels and values are lists of connected node labels.
    - node_attributes (dict): A dictionary where keys are node labels and values are attributes (e.g., sizes, centralities).
    - attribute (str): The attribute to visualize ('size', 'color', etc.).
    """
    G = nx.Graph()
    
    for node, neighbors in network_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    if attribute == 'size':
        sizes = [node_attributes.get(node, 100) for node in G.nodes()]
        nx.draw(G, with_labels=True, node_size=sizes, node_color='lightblue', edge_color='gray', font_size=10, font_weight='bold')
    elif attribute == 'color':
        colors = [node_attributes.get(node, 'lightblue') for node in G.nodes()]
        nx.draw(G, with_labels=True, node_color=colors, edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    
    plt.title(f'Network Graph with Node {attribute.capitalize()}')
    plt.show()

def plot_stacked_bar_chart(stacked_data):
    """
    Plots a stacked bar chart using matplotlib.

    Parameters:
    - stacked_data (dict): A dictionary where keys are categories and values are lists of values for each stack.
    """
    categories = list(stacked_data.keys())
    data = np.array(list(stacked_data.values()))

    plt.bar(categories, data[:, 0], label='Stack 1')
    for i in range(1, data.shape[1]):
        plt.bar(categories, data[:, i], bottom=np.sum(data[:, :i], axis=1), label=f'Stack {i+1}')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Stacked Bar Chart')
    plt.legend()
    plt.show()

def plot_heatmap(data, color_map='viridis'):
    """
    Plots a heatmap of data with customizable color maps.

    Parameters:
    - data (np.ndarray): The data to plot.
    - color_map (str): The color map to use (e.g., 'viridis', 'plasma', 'inferno').
    """
    plt.imshow(data, cmap=color_map, aspect='auto')
    plt.colorbar(label='Value')
    plt.title(f'Heatmap ({color_map.capitalize()} Color Map)')
    plt.show()

def plot_3d_graph(network_data):
    """
    Plots a 3D network graph using matplotlib.

    Parameters:
    - network_data (dict): A dictionary where keys are node labels and values are lists of connected node labels.
    """
    G = nx.Graph()
    
    for node, neighbors in network_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='gray')
    
    for node in G.nodes():
        x, y, z = pos[node]
        ax.scatter(x, y, z, label=node)
    
    plt.title('3D Network Graph')
    plt.legend()
    plt.show()

def animate_graph_changes(network_data_list):
    """
    Creates an animation of changing network graphs over time using matplotlib.

    Parameters:
    - network_data_list (list of dicts): A list of network data dictionaries representing graph changes over time.
    """
    fig, ax = plt.subplots()
    
    def update(num):
        ax.clear()
        G = nx.Graph()
        for node, neighbors in network_data_list[num].items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
        ax.set_title(f'Network Graph - Frame {num}')
    
    ani = animation.FuncAnimation(fig, update, frames=len(network_data_list), repeat=False)
    plt.show()

def plot_interactive_scatter(data):
    """
    Plots an interactive scatter plot using Plotly Express.

    Parameters:
    - data (pd.DataFrame): A DataFrame with columns 'x', 'y', and optionally 'category'.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        marker=dict(size=10),
        text=data.get('category', [])
    ))
    fig.update_layout(
        title='Interactive Scatter Plot',
        xaxis_title='X Axis',
        yaxis_title='Y Axis'
    )
    fig.show()
