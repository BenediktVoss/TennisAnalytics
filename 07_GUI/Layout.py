import gradio as gr
from Dataset import Dataset
from Video import Video

theme = gr.themes.Monochrome(
    primary_hue="stone",
    neutral_hue="gray",
    radius_size="sm",
    text_size="lg",
)
theme.set(
    button_primary_background_fill='*primary_600',
    button_primary_background_fill_hover='*primary_400'
)

css= """
.caption-label{
    background-color: transparent;
}
.tab-nav button {
    font-size: 1.3rem;
}
"""

with gr.Blocks(theme=theme, css=css) as Layout:
    gr.Markdown("# Tennis Analytics")

    with gr.Row(): # Tabbed Interface
        tabbed_interface = gr.TabbedInterface(
            [Dataset, Video],
            tab_names=["Dataset", "Video"],
        )

    gr.Markdown("### Created By: Benedikt Voß")
