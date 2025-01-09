import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

st.set_page_config(layout="wide")

st.title("Conflict Analysis")

def conflict_analysis():
    if 'data' not in st.session_state or 'selected_attributes' not in st.session_state:
        st.warning("Please upload a dataset and select attributes first!")
        return

    st.header("Conflict Analysis")
    st.write("### Conflict Flow Visualization")

    # Filter data based on selected attributes
    filtered_data = st.session_state['data'][st.session_state['selected_attributes']]
    st.write("Filtered Data for Analysis:")
    st.write(filtered_data.head())

    # Define mapping rules
    mapping_rules = {
        'Attrition': {'low': 'No', 'high': 'Yes'},
        'RelationshipSatisfaction': 'scale',
        'WorkLifeBalance': 'scale',
        'EnvironmentSatisfaction': 'scale',
        'JobSatisfaction': 'scale',
        'Education': 'scale',
        'PerformanceRating': "scale",
        'Age': 'balanced',
        'MonthlyIncome': 'balanced',
        'YearsAtCompany': 'balanced',
        'OverTime': {'low': 'No', 'high': 'Yes'}
    }

    # Filter mapping rules to include only available attributes
    available_mapping_rules = {attr: rule for attr, rule in mapping_rules.items() if attr in filtered_data.columns}
    st.write(f"Available Mapping Rules: {list(available_mapping_rules.keys())}")

    # Compute necessary statistics for 'balanced' rules
    stat_values = {}
    for attr, rule in available_mapping_rules.items():
        if rule == 'balanced':
            stat_values[attr] = {
                'low_threshold': filtered_data[attr].quantile(0.4),
                'high_threshold': filtered_data[attr].quantile(0.6)
            }

    def map_attribute(value, rule, stat_value=None):
        if rule == 'scale':
            return 1 if value >= 3 else 0 if value == 2 else -1
        elif isinstance(rule, dict):
            return 1 if value == rule['high'] else -1 if value == rule['low'] else 0
        elif rule == 'balanced':
            return 1 if value > stat_value['high_threshold'] else -1 if value < stat_value['low_threshold'] else 0

    def map_values(row):
        mapped_row = []
        for attr, rule in available_mapping_rules.items():
            if attr in row:
                value = row[attr]
                stat_value = stat_values.get(attr)
                mapped_value = map_attribute(value, rule, stat_value)
                mapped_row.append(mapped_value)
        return tuple(mapped_row)

    # Create the mapped DataFrame
    mapped_data = filtered_data.apply(map_values, axis=1)
    mapped_array = np.array(mapped_data.tolist())
    multi_soft_set_df = pd.DataFrame(mapped_array, columns=available_mapping_rules.keys())

    # st.write("Multi-Soft Set Representation:")
    # st.write(multi_soft_set_df.head())

    # Generate conflict summary
    melted_df = multi_soft_set_df.melt(var_name='Attribute', value_name='Opinion')
    conflict_summary = melted_df.groupby(['Attribute', 'Opinion']).size().reset_index(name='Support')
    conflict_summary['Opinion'] = conflict_summary['Opinion'].replace({1: '+', 0: '0', -1: '-'})
    
    # Compute measures
    total_support_opinion = conflict_summary.groupby('Opinion')['Support'].sum().rename("TotalOpinionSupport")
    total_support_attribute = conflict_summary.groupby('Attribute')['Support'].sum().rename("TotalAttributeSupport")
    conflict_summary = conflict_summary.merge(total_support_opinion, on='Opinion')
    conflict_summary = conflict_summary.merge(total_support_attribute, on='Attribute')
    conflict_summary['Coverage'] = conflict_summary['Support'] / conflict_summary['TotalOpinionSupport']
    conflict_summary['Certainty'] = conflict_summary['Support'] / conflict_summary['TotalAttributeSupport']
    conflict_summary['Strength'] = conflict_summary['Support'] / conflict_summary['Support'].sum()
    
    conflict_summary_cleaned = conflict_summary[['Attribute', 'Opinion', 'Support', 'Coverage', 'Certainty', 'Strength']]
    st.write("Conflict Summary:")
    st.write(conflict_summary_cleaned)

    # Visualization
    def draw_conflict_graph(data, title):
        # Filter the data based on the certainty threshold
        with st.spinner("Generating report..."):
            filtered_data = data[data['Certainty'] >= 0.39]

            # Group attributes by their opinions
            opinions = filtered_data.groupby('Attribute')['Opinion'].agg(lambda x: x.value_counts().idxmax())
            positive_attributes = opinions[opinions == "+"].index.tolist()
            negative_attributes = opinions[opinions == "-"].index.tolist()
            neutral_attributes = opinions[opinions == "0"].index.tolist()

            # Initialize the graph
            G = nx.Graph()

            # Add nodes for all attributes
            for attribute in opinions.index:
                G.add_node(attribute, size=2000, color='skyblue')

            # Add alliance edges (dotted lines) for attributes with the same opinion (++ or --)
            all_attributes = positive_attributes + negative_attributes
            for i, attr1 in enumerate(all_attributes):
                for attr2 in all_attributes[i + 1:]:
                    if attr1 in positive_attributes and attr2 in positive_attributes:
                        G.add_edge(attr1, attr2, style="dotted")
                    elif attr1 in negative_attributes and attr2 in negative_attributes:
                        G.add_edge(attr1, attr2, style="dotted")

            # Add conflict edges (solid lines) for attributes with different opinions (+- or -+)
            for pos_attr in positive_attributes:
                for neg_attr in negative_attributes:
                    G.add_edge(pos_attr, neg_attr, style="solid")

            # Layout for nodes
            pos = nx.circular_layout(G)

            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=[2000 for _ in G.nodes()],  # Uniform node size
                node_color=['skyblue' for _ in G.nodes()],  # Uniform node color
                ax=ax
            )

            # Separate edges by type
            solid_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr["style"] == "solid"]
            dotted_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr["style"] == "dotted"]

            # Draw edges
            nx.draw_networkx_edges(G, pos, edgelist=solid_edges, style="solid", edge_color="red", width=2, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, style="dotted", edge_color="green", width=2, ax=ax)

            # Add labels to nodes
            nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", ax=ax)

            # Set the title and hide axes
            ax.set_title(title, fontsize=16)
            ax.axis("off")

            fig.savefig("images/draw_conflict_graph.png")

            # Display the figure in Streamlit
            image = Image.open('images/draw_conflict_graph.png')
            st.image(image)

    st.write("### How to Interpret the Conflict Graph")
    st.markdown(
    """
    - **Nodes**: Each node represents an attribute from the dataset.
      - The size and color of the nodes are uniform for visual clarity.
    - **Edges**:
      - <span style="color:red; font-weight:bold;">Solid Red Lines</span>: Represent <span style="color:red; font-weight:bold;">"Conflict"</span> between attributes with opposing opinions.
      - <span style="color:green; font-weight:bold;">Dotted Green Lines</span>: Represent <span style="color:green; font-weight:bold;">"Alliance"</span> between attributes with similar opinions.
    - **Threshold**: Only attributes with a certainty above the threshold (e.g., >= 0.39) are displayed.
    - Use the graph to explore relationships between attributes, identifying areas of tension or alignment.
    """,
    unsafe_allow_html=True
)

    draw_conflict_graph(conflict_summary_cleaned, "Conflict Graph")

conflict_analysis()
