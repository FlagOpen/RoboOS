from flask import Flask, request, jsonify
from agents.agent import GlobalAgent

app = Flask(__name__)

master_agent = GlobalAgent(config_path="config.yaml")


@app.route("/publish_task", methods=["POST", "GET"])
def publish_task():
    """
    Publish a task to the Redis channel.

    Request JSON format:
    {
        "task": "task_content"  # The task to be published
    }

    Returns:
        JSON response with status or error message
    """
    if request.method == "GET":
        return jsonify({"statis": "success"}), 200
    try:
        data = request.get_json()
        if not data or "task" not in data:
            return jsonify({"error": "Invalid request - 'task' field required"}), 400
        if not isinstance(data["task"], list):
            data["task"] = [data["task"]]

        for task in data["task"]:
            if not isinstance(task, str):
                return jsonify({"error": "Invalid task format - must be a string"}), 400
            subtask_list = master_agent.publish_global_task(data["task"])

        return jsonify(
            {
                "status": "success",
                "message": "Task published successfully",
                "subtask_list": subtask_list,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
