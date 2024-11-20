import ast
import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from ast import literal_eval



# Complete code for comparing shelves for two JSONs, calculating shelf scores, and comparing multiple JSONs

import json

# Helper function to extract SKU data from a given image JSON
def extract_shelf_skus(image_data):
    shelf_skus = {}
    for detection in image_data.get('detections', []):
        for shelf in detection.get('shelf_list', []):
            shelf_id = shelf['shelf_id']
            sku_names = [sku['Name'] for sku in shelf['SKU_list']]
            sku_count = {sku: sku_names.count(sku) for sku in set(sku_names)}
            shelf_skus[shelf_id] = sku_count
    return shelf_skus

# Function to compare SKUs between two shelves
def compare_shelves(ideal_shelf, other_shelf):
    missing = {}
    added = {}

    # Compare missing SKUs and include how many were in the ideal shelf
    for sku, count in ideal_shelf.items():
        if sku in other_shelf:
            if other_shelf[sku] < count:
                missing[sku] = {'missing': count - other_shelf[sku], 'in_ideal': count}
        else:
            missing[sku] = {'missing': count, 'in_ideal': count}

    # Compare added SKUs
    for sku, count in other_shelf.items():
        if sku in ideal_shelf:
            if count > ideal_shelf[sku]:
                added[sku] = count - ideal_shelf[sku]
        else:
            added[sku] = count

    return {"missing": missing, "added": added}

# Function to compare two JSON datasets (ideal and another image)
def compare_shelf_jsons_with_counts(ideal_json, other_json):
    
    
    # Parse the JSON strings if needed (handling double-escaped JSON issue)
    ideal_data = json.loads(json.loads(ideal_json)) if isinstance(ideal_json, str) else ideal_json
    other_data = json.loads(json.loads(other_json)) if isinstance(other_json, str) else other_json

    # Extract SKU data for the ideal and other images
    ideal_shelf_skus = extract_shelf_skus(ideal_data)
    other_shelf_skus = extract_shelf_skus(other_data)

    # Compare shelves and store the results
    comparison_result = {'shelf_comparisons': {}}

    # Compare each shelf
    for shelf_id, ideal_skus in ideal_shelf_skus.items():
        other_skus = other_shelf_skus.get(shelf_id, {})
        comparison = compare_shelves(ideal_skus, other_skus)
        comparison_result['shelf_comparisons'][shelf_id] = comparison

    return comparison_result

# Function to calculate the shelf-level scores based on the comparison results
def calculate_activity_score_by_shelf(comparison_result):
    shelf_scores = {}

    # Iterate over each shelf comparison
    for shelf_id, comparison in comparison_result['shelf_comparisons'].items():
        total_missing = 0
        total_added = 0

        # Sum missing SKUs
        for sku, details in comparison['missing'].items():
            if isinstance(details, dict):  # When the missing is stored with 'in_ideal'
                total_missing += details['missing']
            else:  # If it's directly the count of missing
                total_missing += details

        # Sum added SKUs
        for sku, count in comparison['added'].items():
            total_added += count

        # Calculate the score for this shelf
        score = total_missing ** 2 + total_added ** 2
        shelf_scores[shelf_id] = score

    return shelf_scores

# Function to compare multiple JSONs at once and return shelf scores for each
def compare_multiple_jsons_with_ideal(ideal_json, other_json_list):
    all_comparisons = {}

    # Loop through all other JSONs and calculate shelf-level scores for each comparison
    for idx, other_json in enumerate(other_json_list):
        # Perform the comparison for this image
        comparison_result = compare_shelf_jsons_with_counts(ideal_json, other_json)
        # Calculate shelf-level scores for this comparison
        shelf_scores = calculate_activity_score_by_shelf(comparison_result)
        # Use index or another identifier as key
        all_comparisons[f'json_{idx}'] = shelf_scores

    return all_comparisons

# Function to aggregate shelf scores across multiple comparisons
def aggregate_shelf_scores(ideal_json, other_json_list):
    aggregated_scores = {}

    # Loop through all other JSONs and calculate shelf-level scores for each comparison
    for other_json in other_json_list:
        # Perform the comparison for this image
        comparison_result = compare_shelf_jsons_with_counts(ideal_json, other_json)
        # Calculate shelf-level scores for this comparison
        shelf_scores = calculate_activity_score_by_shelf(comparison_result)

        # Aggregate scores for each shelf
        for shelf_id, score in shelf_scores.items():
            if shelf_id in aggregated_scores:
                aggregated_scores[shelf_id] += score
            else:
                aggregated_scores[shelf_id] = score

    return aggregated_scores


import pandas as pd
from collections import Counter

def calculate_kpis(data, start_interval, end_interval):
    # Filter data based on the interval range
    filtered_data = data[(data['Interval'] >= start_interval) & (data['Interval'] <= end_interval)]
    
    # Aggregating KPIs
    top_sold_count = filtered_data['total_sold'].sum()
    top_misplaced_count = filtered_data['total_misplaced'].sum()
    
    # Finding the most active shelf
    most_active_shelf = Counter(filtered_data['most_active_shelf'].dropna()).most_common(1)
    most_active_shelf = most_active_shelf[0][0] if most_active_shelf else None
    
    sold_sku_counts = Counter(filtered_data['top_sold_sku'].dropna())
    misplaced_sku_counts = Counter(filtered_data['top_misplaced_sku'].dropna())
    
    # Concatenating unique sold and misplaced SKUs
    most_sold_sku = sold_sku_counts.most_common(1)[0][0] if sold_sku_counts else None
    top_misplaced_sku = misplaced_sku_counts.most_common(1)[0][0] if misplaced_sku_counts else None
    
    
    # Return the requested KPIs
    return {
        "top_sold": int(top_sold_count),
        "total_misplaced": int(top_misplaced_count),
        "most_active_shelf": most_active_shelf,
        "most_sold_skus": most_sold_sku,
        "top_misplaced_skus": top_misplaced_sku
    }







def calculate_final_kpis(data, start_interval, end_interval):
    # Filter data based on the interval range
    filtered_data = data[(data['Interval'] >= start_interval) & (data['Interval'] <= end_interval)]
    
    # Counting sold SKUs
    sold_sku_counts = Counter(filtered_data['top_sold_sku'].dropna())
    top_sold_items = [(sku, count) for sku, count in sold_sku_counts.most_common(3)]
    
    # Counting misplaced SKUs
    misplaced_sku_counts = Counter(filtered_data['top_misplaced_sku'].dropna())
    top_misplaced_items = [(sku, count) for sku, count in misplaced_sku_counts.most_common(3)]
    
    # Parse misplaced_skus to add rack-based misplacement counts
    misplaced_rack_counts = defaultdict(int)
    for misplaced_skus in filtered_data['misplaced_skus'].dropna():
        skus = literal_eval(misplaced_skus)
        for sku, rack_info_list in skus.items():
            for rack_info in rack_info_list:
                rack_id = rack_info['rack_id']
                misplaced_rack_counts[(sku, rack_id)] += abs(rack_info['count_difference'])

    # Getting the top 3 misplaced racks by count
    top_misplaced_racks = sorted(misplaced_rack_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_misplaced_rack_details = [f"{sku} at rack {rack} ({count})" for (sku, rack), count in top_misplaced_racks]
    
    # Finding the most active shelves and racks
    shelf_counts = Counter(filtered_data['most_active_shelf'].dropna())
    most_active_shelves = [f"Shelf {shelf} ({count})" for shelf, count in shelf_counts.most_common(3)] 
    least_active_shelves = [f"Shelf {shelf} ({count})" for shelf, count in shelf_counts.most_common()[:-4:-1] if count > 0]  # Bottom 3 with counts greater than zero
 
    non_compliance_time = filtered_data['non_compliance_time'].sum() if 'non_compliance_time' in filtered_data.columns else None


    
    return {
        "top_sold_items": top_sold_items,
        "top_misplaced_items": top_misplaced_items,
        "most_active_shelves": most_active_shelves,
        "least_active_shelves": least_active_shelves,
        "top_misplaced_rack_details": top_misplaced_rack_details,
        "top_sold_sku": top_sold_items[0][0] if top_sold_items else None,  # Top sold SKU by name
        "top_misplaced_sku": top_misplaced_items[0][0] if top_misplaced_items else None,  # Top misplaced SKU by name
        "non_compliance_time": non_compliance_time
 
    }
    
    

# def calculate_final_kpis(data, start_interval, end_interval):
#     # Filter data based on the interval range
#     filtered_data = data[(data['Interval'] >= start_interval) & (data['Interval'] <= end_interval)]
    
#     # Counting sold SKUs
#     sold_sku_counts = Counter(filtered_data['top_sold_sku'].dropna())
#     top_sold_items = [(sku, count) for sku, count in sold_sku_counts.most_common(3)]
    
#     # Counting misplaced SKUs
#     misplaced_sku_counts = Counter(filtered_data['top_misplaced_sku'].dropna())
#     top_misplaced_items = [(sku, count) for sku, count in misplaced_sku_counts.most_common(3)]
    
#     # Parse misplaced_skus to add rack-based misplacement counts
#     misplaced_rack_counts = defaultdict(int)
#     for misplaced_skus in filtered_data['misplaced_skus'].dropna():
#         skus = literal_eval(misplaced_skus)
#         for sku, rack_info_list in skus.items():
#             for rack_info in rack_info_list:
#                 rack_id = rack_info['rack_id']
#                 misplaced_rack_counts[(sku, rack_id)] += abs(rack_info['count_difference'])

#     # Getting the top 3 misplaced racks by count
#     top_misplaced_racks = sorted(misplaced_rack_counts.items(), key=lambda x: x[1], reverse=True)[:3]
#     top_misplaced_rack_details = [f"{sku} at rack {rack} ({count})" for (sku, rack), count in top_misplaced_racks]
    
#     # Calculate activity of each rack on each shelf based on misplaced and missing SKUs
#     shelf_rack_misplaced_counts = defaultdict(lambda: defaultdict(int))
#     shelf_rack_missing_counts = defaultdict(lambda: defaultdict(int))

#     for idx, row in filtered_data.iterrows():
#         # Parse misplaced SKUs for rack-level misplacements
#         misplaced_skus = row['misplaced_skus']
#         if pd.notna(misplaced_skus):
#             skus = literal_eval(misplaced_skus)
#             for sku, rack_info_list in skus.items():
#                 for rack_info in rack_info_list:
#                     rack_id = rack_info['rack_id']
#                     count_diff = abs(rack_info['count_difference'])
#                     shelf_id = row['most_active_shelf']
#                     shelf_rack_misplaced_counts[shelf_id][rack_id] += count_diff

#         # Parse missing SKUs for rack-level missing counts
#         missing_skus = row['missing_skus']
#         if pd.notna(missing_skus):
#             skus = literal_eval(missing_skus)
#             for sku, count in skus.items():
#                 shelf_id = row['most_active_shelf']
#                 shelf_rack_missing_counts[shelf_id][shelf_id] += count

#     # Summing misplaced and missing counts for each rack to calculate total activity
#     shelf_rack_activity = defaultdict(dict)
#     for shelf, racks in shelf_rack_misplaced_counts.items():
#         for rack, misplaced_count in racks.items():
#             missing_count = shelf_rack_missing_counts[shelf].get(rack, 0)
#             shelf_rack_activity[shelf][rack] = misplaced_count + missing_count

#     # Determine most and least active racks within each shelf
#     activity_list = [(shelf, rack, activity) for shelf, racks in shelf_rack_activity.items() for rack, activity in racks.items()]
#     most_active = sorted(activity_list, key=lambda x: x[2], reverse=True)[:3]
#     least_active = sorted(activity_list, key=lambda x: x[2])[:3]
    
#     most_active_racks = [f"Shelf {shelf} (Rack {rack})" for shelf, rack, _ in most_active]
#     least_active_racks = [f"Shelf {shelf} (Rack {rack})" for shelf, rack, _ in least_active]

#     # Calculating most and least active shelves based on occurrences in 'most_active_shelf' column
#     shelf_counts = Counter(filtered_data['most_active_shelf'].dropna())
#     most_active_shelves = [f"Shelf {shelf} ({count})" for shelf, count in shelf_counts.most_common(3)]
#     least_active_shelves = [f"Shelf {shelf} ({count})" for shelf, count in shelf_counts.most_common()[:-4:-1] if count > 0]

#     non_compliance_time = filtered_data['non_compliance_time'].sum() if 'non_compliance_time' in filtered_data.columns else None

#     return {
#         "top_sold_items": top_sold_items,
#         "top_misplaced_items": top_misplaced_items,
#         "most_active_shelves": most_active_shelves,
#         "least_active_shelves": least_active_shelves,
#         "top_misplaced_rack_details": top_misplaced_rack_details,
#         "most_active_racks": most_active_racks,
#         "least_active_racks": least_active_racks,
#         "top_sold_sku": top_sold_items[0][0] if top_sold_items else None,
#         "top_misplaced_sku": top_misplaced_items[0][0] if top_misplaced_items else None,
#         "non_compliance_time": non_compliance_time
#     }












# def add_responses(df, index1, index2):
#     # Retrieve responses for the specified indices
#     response_1 = ast.literal_eval(df.loc[df['index'] == index1, 'response'].values[0])
#     response_2 = ast.literal_eval(df.loc[df['index'] == index2, 'response'].values[0])
    
#     # Sum corresponding values in response dictionaries
#     combined_response = {key: response_1.get(key, 0) + response_2.get(key, 0) for key in set(response_1) | set(response_2)}
    
#     return combined_response


# import ast

def add_responses(df, index1, index2):
    """
    Sums the response dictionaries for all indices from index1 to index2.

    Parameters:
        df (pd.DataFrame): DataFrame containing response data.
        index1 (int): Start index.
        index2 (int): End index.

    Returns:
        dict: Combined response dictionary with summed values.
    """
    # Initialize an empty dictionary to accumulate the combined response
    combined_response = {}

    # Iterate over the range from index1 to index2 inclusive
    for idx in range(index1, index2 + 1):
        # Parse the response at the current index
        response = ast.literal_eval(df.loc[df['index'] == idx, 'response'].values[0])
        
        # Sum the values in the response dictionary
        for key, value in response.items():
            combined_response[key] = combined_response.get(key, 0) + value

    return combined_response





def get_unique_skus_per_shelf(df, row_index=0):
    """
    Extracts unique SKUs for each shelf ID in a specified row of the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the shelf detection data.
        row_index (int): The index of the row to extract SKUs from. Default is 0 (first row).

    Returns:
        dict: A dictionary with shelf IDs as keys and lists of unique SKU names as values.
    """
    # Parse the response field of the specified row to a dictionary
    response = ast.literal_eval(df.loc[row_index, 'response'])

    # Dictionary to store unique SKUs for each shelf
    unique_skus_per_shelf = {}

    # Loop through each shelf in the shelf list for the first bay detection
    for shelf in response['detections'][0]['shelf_list']:
        shelf_id = shelf['shelf_id']
        # Extract SKU names and remove duplicates using a set
        unique_skus = set(sku['Name'] for sku in shelf['SKU_list'] if sku['Name'] != 'Unknown')
        unique_skus_per_shelf[shelf_id] = list(unique_skus)

    return unique_skus_per_shelf



import plotly.graph_objects as go
import plotly.express as px
import base64

def generate_shelf_activity_chart(score, logo_paths, title="Shelf Activity", xaxis_title="Shelf Activity Value", yaxis_title="Shelves"):
    """
    Generates a Plotly chart for shelf activity with embedded images for each shelf, using a color gradient.

    Parameters:
        score (dict): Dictionary with shelf IDs as keys and activity values as values.
        logo_paths (list): List of lists with image file paths for each shelf.
        title (str): Title of the chart.
        xaxis_title (str): Title for the X-axis.
        yaxis_title (str): Title for the Y-axis.

    Returns:
        fig (go.Figure): Plotly figure object for shelf activity.
    """
    # Convert score dictionary into shelves and activity_values lists
    shelves = [f"Shelf {shelf_id + 1}" for shelf_id in score.keys()]
    activity_values = list(score.values())

    # Adjust `logo_paths` to match the number of shelves in `score`
    while len(logo_paths) < len(shelves):
        logo_paths.append([])

    # Normalize activity values for color mapping
    max_value = max(activity_values) if max(activity_values) > 0 else 1
    colors = [px.colors.sequential.Blues[int((val / max_value) * (len(px.colors.sequential.Blues) - 1))] for val in activity_values]

    # Convert images to base64 for shelf activity chart
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    image_base64 = [[img_to_base64(path) for path in paths] for paths in logo_paths[:len(shelves)]]

    # Create a figure for shelf activity
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=shelves,
        x=activity_values,
        orientation='h',
        marker=dict(color=colors),  # Use gradient color for each shelf based on activity values
    ))

    # Add images to each shelf only if activity score >= 1
    for idx, shelf in enumerate(shelves):
        if activity_values[idx] >= 1:
            logos = image_base64[idx]
            for i, logo in enumerate(logos):
                fig.add_layout_image(
                    source=f'data:image/png;base64,{logo}',
                    xref="x",
                    yref="y",
                    x=activity_values[idx] + (i * 6) + 2,
                    y=idx,
                    sizex=6,
                    sizey=5,
                    xanchor="left",
                    yanchor="middle"
                )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(showgrid=False, autorange="reversed"),
        xaxis=dict(range=[0, 100])  # Adjust range to fit the data
    )

    return fig






