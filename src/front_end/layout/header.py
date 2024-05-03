import gradio as gr


def header(callback_function: callable) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.cyan
        ),
        title="Ask us about Medical Research Papers!",
    ) as front_end:
        gr.HTML("<img src='file/src/front_end/assets/background.png'>")
        gr.Markdown(
            """
            # Retreive relevant research papers to your query
            """
        )
        textbox = gr.Textbox(label="Please enter your question:")
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_chat = gr.Textbox(label="Chat response:")
            output_summary = gr.Textbox(label="Summarisation:")
            output_relevant_docs = gr.List(type="pandas", label="Relevant documents:")

        button.click(
            callback_function,
            textbox,
            outputs=[output_relevant_docs, output_chat, output_summary],
        )

    return front_end
