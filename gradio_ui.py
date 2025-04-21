# -*- coding: utf-8 -*-
import requests
import json
import os
import gradio as gr
from typing import Dict

CONFIG_FILE = "client_config.json"


class ConfigManager:
    @staticmethod
    def get_server_config() -> Dict:
        """Get server configuration"""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {"server_url": "http://localhost:5000"}

    @staticmethod
    def save_config(config: Dict):
        """Save server configuration"""
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)


class APIClient:
    def __init__(self):
        self.config = ConfigManager.get_server_config()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def test_connection(self, server_url: str) -> tuple:
        """Test server connection"""
        try:
            response = self.session.get(f"{server_url}/publish_task", timeout=3)
            if response.status_code == 200:
                ConfigManager.save_config({"server_url": server_url})
                return "🟢 Connection successful!", server_url
            return f"🔴 Connection failed (HTTP {response.status_code})", server_url
        except Exception as e:
            return f"🔴 Connection error: {str(e)}", server_url

    def send_message(self, server_url: str, message: str) -> tuple:
        """Send message to server"""
        if not message.strip():
            return "⚠️ Message cannot be empty", ""

        try:
            response = self.session.post(
                f"{server_url}/publish_task", json={"task": message}, timeout=300
            )
            response.raise_for_status()
            return "🟢 Message sent successfully", json.dumps(response.json(), indent=2)
        except requests.exceptions.HTTPError as e:
            return f"🔴 Server error (HTTP {e.response.status_code})", str(e)
        except Exception as e:
            return f"🔴 Error: {str(e)}", ""


def create_gradio_interface():
    client = APIClient()
    current_config = client.config

    # Custom tech-inspired theme
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    ).set(
        button_primary_background_fill="linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #4F46E5 0%, #06B6D4 70%)",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_dark="linear-gradient(90deg, #6366F1 0%, #0891B2 100%)",
        button_secondary_background_fill="linear-gradient(90deg, #1E293B 0%, #334155 100%)",
        button_secondary_background_fill_hover="linear-gradient(90deg, #1E293B 0%, #334155 70%)",
        button_secondary_text_color="#E2E8F0",
        slider_color="#06B6D4",
        slider_color_dark="#0891B2",
        block_title_text_color="*primary_500",
        block_label_text_color="*primary_300",
        input_background_fill_dark="#1E293B",
        input_border_color_dark="#334155",
    )

    with gr.Blocks(
        title="RoboOS API Client",
        theme=theme,
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        }
        .dark .gradio-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        }
        .gradio-header {
            border-bottom: 1px solid #334155 !important;
        }
        .gradio-interface {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .gradio-box {
            border-radius: 8px !important;
            border: 1px solid #334155 !important;
            background: rgba(15, 23, 42, 0.7) !important;
            backdrop-filter: blur(10px) !important;
        }
        .gradio-button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .gradio-button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        .gradio-input, .gradio-output {
            border-radius: 6px !important;
        }
        .gradio-markdown {
            font-family: 'JetBrains Mono', monospace !important;
        }
        .gradio-markdown h1 {
            background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            letter-spacing: -0.5px;
        }
        .gradio-markdown h3 {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
            margin-top: 1rem !important;
        }
        .logo-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        .logo-text {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -1px;
        }
        .tagline {
            color: #94A3B8 !important;
            margin-top: 0 !important;
            font-size: 0.9rem !important;
        }
        .status-bar {
            width: 100%; 
            text-align: center; 
            color: #64748B; 
            font-size: 0.8rem; 
            padding: 0.5rem 0; 
            border-top: 1px solid #334155;
            font-family: 'JetBrains Mono', monospace;
        }
        """,
    ) as app:
        # Header section
        with gr.Row():
            gr.Markdown(
                """
            <div class="logo-container">
                <h1 class="logo-text">🤖RoboOS</h1>
            </div>
            <p class="tagline">Advanced API Client Interface v2.0</p>
            """
            )

        # Configuration section
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                server_url = gr.Textbox(
                    label="Server URL",
                    value=current_config["server_url"],
                    placeholder="http://localhost:5000",
                    lines=1,
                    max_lines=1,
                    interactive=True,
                )
                with gr.Row():
                    test_btn = gr.Button("Test Connection", variant="secondary")
                    connection_status = gr.Textbox(
                        label="Connection Status", interactive=False, scale=4
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                save_btn = gr.Button("Save Settings", variant="secondary")

        # Message composition section
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown("### Message Composition")
                message_input = gr.Textbox(
                    label="",
                    placeholder="Enter your API message here...",
                    lines=5,
                    max_lines=10,
                    elem_classes=["code-input"],
                )
                with gr.Row():
                    send_btn = gr.Button("Send Message", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=0)

        # Response section
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown("### Server Response")
                with gr.Tabs():
                    with gr.TabItem("JSON Response"):
                        response_output = gr.JSON(
                            label="", elem_classes=["code-output"]
                        )
                    with gr.TabItem("Raw Response"):
                        raw_output = gr.Textbox(
                            label="",
                            interactive=False,
                            lines=10,
                            elem_classes=["code-output"],
                        )
                status_output = gr.Textbox(
                    label="Status", interactive=False, elem_classes=["status-box"]
                )

        # Status bar
        with gr.Row():
            gr.Markdown(
                f"""
            <div class="status-bar">
                RoboOS API Client • Connected to <span id="current-server">{current_config["server_url"]}</span> • System Online
            </div>
            """
            )

        # Event handlers
        test_btn.click(
            client.test_connection,
            inputs=[server_url],
            outputs=[connection_status, server_url],
        )

        save_btn.click(
            lambda url: ConfigManager.save_config({"server_url": url}),
            inputs=[server_url],
            outputs=None,
        ).then(
            lambda: gr.Info("Configuration saved successfully!"),
            inputs=None,
            outputs=None,
        )

        send_btn.click(
            client.send_message,
            inputs=[server_url, message_input],
            outputs=[status_output, response_output],
        ).then(lambda x: x, inputs=[response_output], outputs=[raw_output])

        clear_btn.click(
            lambda: ("", "", ""),
            inputs=None,
            outputs=[message_input, response_output, raw_output],
        )

        # Enter key submission
        message_input.submit(
            client.send_message,
            inputs=[server_url, message_input],
            outputs=[status_output, response_output],
        ).then(lambda x: x, inputs=[response_output], outputs=[raw_output])

        # Dynamic status bar update
        server_url.change(
            lambda x: f"""<script>document.getElementById('current-server').textContent = '{x}'</script>""",
            inputs=[server_url],
            outputs=None,
        )

    return app


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        favicon_path=None,
        show_error=True,
    )
