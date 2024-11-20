import json
from collections import defaultdict
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import os
from count_shelf_activity import calculate_kpis, calculate_final_kpis


# Load the data
file_path = 'unique_people_count.csv'
df = pd.read_csv(file_path)

# Define experiment start time
EXPERIMENT_START_TIME = datetime.combine(datetime.today(), time(9, 0, 0))

# Convert "Interval (seconds)" to actual time ranges and create 20-second increments
def convert_interval_to_time(row):
    start_sec, end_sec = map(int, row.split('-'))
    start_time = EXPERIMENT_START_TIME + timedelta(seconds=start_sec)
    end_time = EXPERIMENT_START_TIME + timedelta(seconds=end_sec)
    return f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"

df['Time Interval'] = df['Interval (seconds)'].apply(convert_interval_to_time)
df['Start Time'] = df['Interval (seconds)'].apply(lambda x: EXPERIMENT_START_TIME + timedelta(seconds=int(x.split('-')[0])))
intervals = pd.date_range(start=df['Start Time'].min(), end=df['Start Time'].max(), freq='20s').tolist()

# Define promotions with their times and colors
promotions = [
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=60), "color": "red", "label": "Quaker Oats Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=80), "color": "red", "label": "Quaker Oats Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=100), "color": "red", "label": "Quaker Oats Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=100), "color": "blue", "label": "Pringles Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=120), "color": "blue", "label": "Pringles Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=140), "color": "blue", "label": "Pringles Promo"},
    {"time": EXPERIMENT_START_TIME + timedelta(seconds=160), "color": "blue", "label": "Pringles Promo"}

]

# Streamlit interface
st.title("People Count, Shelf and Store Analysis by Time Interval")

# Plotting the People Count chart with circles for promotions
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Start Time'], y=df['People Count'], mode='lines+markers', name='People Count'))

# # Add promotion circles
# for promo in promotions:
#     fig.add_trace(go.Scatter(
#         x=[promo["time"]],
#         y=[df['People Count'].min() - 0.5],  # Position circles slightly below data points
#         mode='markers',
#         marker=dict(size=10, color=promo["color"], symbol='circle'),
#         name=promo["label"],  # This will add an entry to the legend
#         showlegend=True
#     ))

# # Update layout with axis labels and margin adjustments
# fig.update_layout(
#     title="People Count in the Store Over Time",
#     xaxis=dict(title="Time", rangeslider=dict(visible=False), type="date"),
#     yaxis=dict(title="People Count"),
#     margin=dict(b=100),  # Increase bottom margin for spacing
# )

# st.plotly_chart(fig, use_container_width=True)


# Track the number of promotions per time to stack them vertically
promotion_counts = defaultdict(int)

# Add promotion circles stacked below the main plot line
for promo in promotions:
    # Offset each promotion marker vertically based on the number of promotions already at that time
    y_position = df['People Count'].min() - 0.5 - (promotion_counts[promo["time"]] * 0.5)
    promotion_counts[promo["time"]] += 1  # Increment count for this time to stack the next marker further down

    fig.add_trace(go.Scatter(
        x=[promo["time"]],
        y=[y_position],
        mode='markers',
        marker=dict(size=10, color=promo["color"], symbol='circle'),
        name=promo["label"],
        showlegend=(promo["label"] not in [trace.name for trace in fig.data])  # Show legend once per label
    ))

# Update layout with axis labels and margin adjustments
fig.update_layout(
    title="People Count in the Store Over Time",
    xaxis=dict(title="Time", rangeslider=dict(visible=False), type="date"),
    yaxis=dict(title="People Count"),
    margin=dict(b=100),  # Increase bottom margin for spacing
)

st.plotly_chart(fig, use_container_width=True)


##########################################################################################

# Select time intervals for analysis
time_intervals = df['Time Interval'].tolist()
start_interval, end_interval = st.select_slider(
    "Select Time Interval for Store Analysis",
    options=time_intervals,
    value=(time_intervals[0], time_intervals[-1])
)
st.markdown(f"<p style='font-size:18px;'>Selected Time Interval: {start_interval} to {end_interval}</p>", unsafe_allow_html=True)

# Use session_state to track button clicks
if "display_kpis_and_heatmaps" not in st.session_state:
    st.session_state["display_kpis_and_heatmaps"] = False
if "display_plotly_charts" not in st.session_state:
    st.session_state["display_plotly_charts"] = False

# Right-aligned "Get Analysis" button
with st.container():
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Get Analysis"):
            st.session_state["display_kpis_and_heatmaps"] = True
            st.session_state["display_plotly_charts"] = False  # Reset Details state
            
            

start_interval_index = time_intervals.index(start_interval) + 1
end_interval_index = time_intervals.index(end_interval) + 1


kpi_df = pd.read_csv('finalized_interval_data.csv')


# Run the modified function for a sample interval range (adjustable as needed)
final_kpis_result = calculate_final_kpis(kpi_df, start_interval=start_interval_index, end_interval=end_interval_index)
# Extract KPI values

top_sold_items = final_kpis_result['top_sold_items']
top_misplaced_items = final_kpis_result['top_misplaced_items']
most_active_shelves = final_kpis_result['most_active_shelves']
least_active_shelves = final_kpis_result['least_active_shelves']
top_misplaced_rack_details = final_kpis_result['top_misplaced_rack_details']
top_sold_skus = final_kpis_result['top_sold_sku']
top_misplaced_skus = final_kpis_result['top_misplaced_sku']
non_compliance_time = final_kpis_result['non_compliance_time']
# Running the KPI calculation function for a specified interval range



# Show KPIs and Heatmaps if "Get Analysis" is clicked
if st.session_state["display_kpis_and_heatmaps"]:
    
    
    start_interval_index = time_intervals.index(start_interval) + 1
    end_interval_index = time_intervals.index(end_interval) + 1
    total_intervals = end_interval_index - start_interval_index + 1  # Calculate the total number of intervals selected
    
    filtered_df = df[(df['Time Interval'] >= start_interval) & (df['Time Interval'] <= end_interval)]
    
    # Display KPIs
    st.markdown("<h1 style='font-size:24px;'>Key Performance Indicators</h1>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3 = st.columns([2, 2, 2])

    # HTML template for smaller font size and line spacing
    kpi_template = "<p style='font-size:18px; margin: 0;'>{}</p>"
    spacer = "<div style='height:10px;'></div>"  # Spacer for vertical gap

    # Format each list item to display on a new line
    top_sold_items_text = "<br>".join([f"{item[0]} ({item[1]})" for item in top_sold_items])
    top_misplaced_items_text = "<br>".join([f"{item[0]} ({item[1]})" for item in top_misplaced_items])
    most_active_shelves_text = "<br>".join(most_active_shelves)
    least_active_shelves_text = "<br>".join(least_active_shelves)


    # Display each KPI with smaller font size and line breaks between multiple values
    kpi_col1.markdown(f"**Top Sold Items**:<br>{kpi_template.format(top_sold_items_text)}{spacer}", unsafe_allow_html=True)

    kpi_col1.markdown(f"**Top Misplaced Items**:<br>{kpi_template.format(top_misplaced_items_text)}{spacer}", unsafe_allow_html=True)

    kpi_col2.markdown(f"**Most Active Shelves**:<br>{kpi_template.format(most_active_shelves_text)}{spacer}", unsafe_allow_html=True)
    
    kpi_col2.markdown(f"**Least Active Shelves**:<br>{kpi_template.format(least_active_shelves_text)}{spacer}", unsafe_allow_html=True)

    # kpi_col3.markdown(f"**Total Non-Complience Time**:<br>{kpi_template.format(non_compliance_time)}", unsafe_allow_html=True)
    
    non_compliance_duration = total_intervals * 20  # Total non-compliance duration in seconds
    kpi_col3.markdown(f"**Total Non-Compliance Time ({non_compliance_duration} seconds)**:<br>{kpi_template.format(non_compliance_time)}", unsafe_allow_html=True)



    # Add spacing between rows for better visual separation
    st.markdown("<br>", unsafe_allow_html=True)
    # Display heatmaps
    st.markdown("<h1 style='font-size:24px;'>Heatmaps for the Selected Intervals</h1>", unsafe_allow_html=True)
    image_folder_path = "heatmap_images"
    available_images = os.listdir(image_folder_path)
    num_columns = 3
    images_to_display = [(os.path.join(image_folder_path, f"heatmap_interval_{i}.png"), time_interval) 
                         for i, time_interval in zip(range(start_interval_index + 1, end_interval_index + 1), filtered_df['Time Interval'])
                         if f"heatmap_interval_{i}.png" in available_images]

    for i in range(0, len(images_to_display), num_columns):
        cols = st.columns(num_columns)
        for col, (image_path, caption) in zip(cols, images_to_display[i:i + num_columns]):
            with col:
                st.image(image_path, caption=caption, use_column_width=True)

    # Right-aligned "Details" button
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Details"):
                st.session_state["display_plotly_charts"] = True

# Show SKU Movement chart if "Details" button is clicked
if st.session_state["display_plotly_charts"]:
    st.markdown("<h1 style='font-size:24px;'>Detailed SKU Movement</h1>", unsafe_allow_html=True)

    # Aggregate SKU data function
    def aggregate_sku_data(shelves_data_slice):
        aggregated_data = defaultdict(lambda: {"SKU": [], "SKU Count": defaultdict(int), "Type": defaultdict(int)})
        for interval_json in shelves_data_slice:
            shelf_data_dict = json.loads(interval_json)
            for rack_name, rack_data in shelf_data_dict.items():
                for sku, count, sku_type in zip(rack_data["SKU"], rack_data["SKU Count"], rack_data["Type"]):
                    aggregated_data[rack_name]["SKU"].append(sku)
                    aggregated_data[rack_name]["SKU Count"][sku] += count
                    aggregated_data[rack_name]["Type"][sku] += (1 if sku_type == "Newly Added" else 0)

        final_data = {}
        for rack_name, rack_data in aggregated_data.items():
            final_data[rack_name] = {
                "SKU": list(rack_data["SKU Count"].keys()),
                "SKU Count": list(rack_data["SKU Count"].values()),
                "Type": ["Newly Added" if rack_data["Type"][sku] > 0 else "Net" for sku in rack_data["SKU Count"].keys()]
            }

        return final_data

    # SKU Movement Plot Function
    # Define the SKU movement plot function
    def plot_sku_movement(shelf_data, shelf_name):
        def get_color(count):
            return "blue" if count < 0 else "red"

        fig = go.Figure()
        categoryarray = []
        tickvals = []

        for rack_name, rack_data in shelf_data.items():
            for sku, count, type_ in zip(rack_data["SKU"], rack_data["SKU Count"], rack_data["Type"]):
                fig.add_trace(go.Bar(
                    x=[count],
                    y=[f"{rack_name.capitalize()} - {sku}"],
                    orientation='h',
                    marker_color=get_color(count),
                    showlegend=False
                ))
                categoryarray.append(f"{rack_name.capitalize()} - {sku}")

            spacer_count = 3
            categoryarray.extend([f"Spacer {rack_name}{i}" for i in range(1, spacer_count + 1)])

        tickvals = [item for item in categoryarray if not item.startswith("Spacer")]

        fig.update_layout(
            title=f"SKU Movement Across Racks for {shelf_name}: Additions and Reductions",
            xaxis_title="SKU Count",
            yaxis_title="SKU",
            yaxis=dict(
                automargin=True,
                categoryorder='array',
                categoryarray=categoryarray[::-1],  # Reverse the array order
                tickvals=tickvals
            ),
            barmode='relative',
            height=700,
            width=800,
            template='plotly_white',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Paths to shelf CSV files
    sku_data_folder = "sku data"
    shelf_files = {
        "Shelf 1": os.path.join(sku_data_folder, "shelf1_json.csv"),
        "Shelf 2": os.path.join(sku_data_folder, "shelf2_json.csv"),
        "Shelf 3": os.path.join(sku_data_folder, "shelf3_json.csv")
    }

    # Load and plot data for each shelf
    for shelf_name, file_path in shelf_files.items():
        # Load the CSV file for the shelf
        shelf_df = pd.read_csv(file_path)
        
        # Adjust the interval slice as needed
        shelves_data_slice = list(shelf_df['Data'])[start_interval_index:end_interval_index]
        
        # Aggregate data
        aggregated_shelf_data = aggregate_sku_data(shelves_data_slice)
        
        # Plot the SKU Movement for the current shelf
        plot_sku_movement(aggregated_shelf_data, shelf_name)
