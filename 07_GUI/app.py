import gradio as gr
from Layout import Layout


if __name__ == '__main__':
    print("Starting Gradio Server...")
    Layout.launch(server_name="localhost", server_port=8080)