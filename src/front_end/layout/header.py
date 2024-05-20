import gradio as gr
import pandas as pd






def header(callback_function: callable) -> gr.Blocks:
    css = """ <style>
    table.dataframe {
        width: 100%;
    }
    table.dataframe td, table.dataframe th {
        border: 1px solid #000;
        padding: 8px;
        word-wrap: break-word;
        max-width: 150px; /* Adjust the max-width as per your requirement */
    }
    </style>"""
    with gr.Blocks(
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.cyan
        ),
        title="Ask us about Medical Research Papers!",
    ) as front_end:
        gr.HTML("<img src='file/src/front_end/assets/logo.png'>")
        gr.Markdown(
            """
            # Retrieve relevant research papers to your query
            """
        )
        textbox = gr.Textbox(label="Please enter your question:")

        
        #dropdown =gr.Dropdown(["Number of Participants", "Phase I Success Rate","Advantages/Disadvantages", "Study Type"], value=["Number of Participants"], multiselect=True, label="Activity", info="XXX")
        dropdown =gr.Dropdown(allow_custom_value=True,  multiselect=True, label="Fields of Interest", info="Select fields to extract from papers ")
        
        with gr.Row():
            button1 = gr.Button("Submit", variant="primary")
            
        with gr.Column():
            output_chat = gr.Textbox(label="Chat response:")
            output_summary = gr.Textbox(label="Summarisation:")
            
            output_relevant_docs = gr.List(type="pandas", label="Relevant documents:", column_widths=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10], wrap=True)
        

        button1.click(
            callback_function,
            inputs = [textbox, dropdown],
            outputs=[output_relevant_docs, output_chat, output_summary],
        
        )


    return front_end
