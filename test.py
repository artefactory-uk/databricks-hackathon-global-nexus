import gradio as gr
import pandas as pd

# Function to generate a table with selected options as columns
def generate_table(selected_options):
    print(selected_options)
    # Create a dictionary to store data for the table
    data = {}
    # Iterate through selected options
    for option in selected_options:
        # Generate sample data for each option
        data[option] = [f"{option} data 1", f"{option} data 2", f"{option} data 3"]
    # Create a DataFrame using the dictionary
    df = pd.DataFrame(data)
    # Return the DataFrame
    return df

# Define options for the dropdown
options = ["Option 1", "Option 2", "Option 3", "Option 4"]

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_table,
    inputs=gr.Dropdown(
            ["Number of participants", "Phase I success rate", "Phase II success rate", "Randomisation Type"], value=["Number of participants"], multiselect=True, label="Activity", info="XXX"
        ),
    outputs="dataframe",
    title="Dropdown Selection to Table",
    live=True,
)

interface.launch()
