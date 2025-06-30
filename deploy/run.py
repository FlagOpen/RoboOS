import os
import ast
import json
import socket
import subprocess
from pathlib import Path

import redis
from flask import Flask, jsonify, render_template, request, send_from_directory
from ruamel.yaml import YAML

from utils import recursive_update, split_dot_keys

yaml = YAML()
yaml.preserve_quotes = True

app = Flask(__name__, static_folder="assets")

services = {
    "inference": {"pid": None, "port": None},
    "master": {"pid": None, "port": 6000},
    "slaver": {"pid": None},
}


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/release")
def release():
    return render_template("release.html")


@app.route("/deploy")
def deploy():
    return render_template("deploy.html")


@app.route("/js/<path:filename>")
def js_file(filename):
    return send_from_directory("templates/js", filename)


@app.route("/config")
def config():
    path = request.args.get("path")
    if not path:
        return jsonify({"status": 400, "message": "Illegal request"})
    if not os.path.exists(path):
        return jsonify({"status": 400, "message": "Configuration does not exist"})

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    return jsonify({"status": 200, "data": data})


@app.route("/saveconfig", methods=["POST"])
def saveconfig():
    try:
        data = request.json

        if not data or "file_path" not in data or "config_data" not in data:
            return (
                jsonify(
                    {
                        "status": 400,
                        "message": "Required parameter missing: file_path 、config_data",
                    }
                ),
                400,
            )

        file_path = data["file_path"]
        config_data = data["config_data"]
        node = data.get("node", "master")
        # TODO Save to environment variable

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


        with open(file_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.load(f)

        processed_config = split_dot_keys(config_data)

        print(processed_config)

        yaml_data = recursive_update(yaml_data, processed_config)


        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f)

        return jsonify(
            {"status": 200, "message": "Configuration file saved successfully"}
        )

    except Exception as e:
        return jsonify({"status": 500, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/validate-config", methods=["POST"])
def validate_config():
    data = request.json
    required_fields = ["conda_env", "startup_command", "master_config", "slaver_config"]

    for field in required_fields:
        if not data.get(field):
            return (
                jsonify(
                    {"success": False, "message": f"Missing necessary fields: {field}"}
                ),
                400,
            )

    for file_path in [data["master_config"], data["slaver_config"]]:
        if not os.path.exists(file_path):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"The model path does not exist: {file_path}",
                    }
                ),
                400,
            )

    # Check if the port is occupied
    is_flagscale = data.get("is_flagscale", True)
    serve_port = [5000]
    if is_flagscale:
        serve_port.append(4567)
        
    for port in serve_port:
        if is_port_in_use(port):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"The port is already occupied: {port}",
                    }
                ),
                400,
            )

    # master
    master_config = data["master_config"]
    with open(master_config, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    # collaborator
    collaborator = data.get("collaborator", None)
    if not collaborator:
        return (
            jsonify(
                {"success": False, "message": f"Lack of collaborator configuration"}
            ),
            400,
        )
    try:
        r = redis.StrictRedis(
            host=collaborator["HOST"],
            port=collaborator["PORT"],
            password=collaborator["PASSWORD"],
            db=collaborator["DB"],
            socket_connect_timeout=5,
        )
        r.ping()
    except Exception as e:

        return (
            jsonify(
                {"success": False, "message": f"communicator connection failed: {e}"}
            ),
            400,
        )

    return jsonify(
        {
            "success": True,
            "message": "Configuration verification passed",
            "validated_config": data,
        }
    )


@app.route("/api/start-inference", methods=["POST"])
def start_inference():
    data = request.json

    conda_env = data.get("conda_env")
    startup_command = data.get("startup_command")
    is_flagscale = data.get("is_flagscale", True)
    if not is_flagscale:
        return jsonify(
            {
                "success": True,
                "message": f"Skip deployment of inference service",
            }
        )

    if not conda_env or not startup_command:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Missing required fields: 'conda_env' and/or 'startup_command'",
                }
            ),
            400,
        )

    try:
        log_file = "vllm.log"

        bash_command = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {startup_command}"

        with open(log_file, "w") as f:
            subprocess.Popen(
                ["bash", "-c", bash_command], stdout=f, stderr=f, preexec_fn=os.setpgrp
            )

        return jsonify(
            {
                "success": True,
                "message": f"Model has been started in conda env '{conda_env}'. Log file: {log_file}",
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to start inference service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/start-master", methods=["POST"])
def start_master():
    data = request.json
    conda_env = data.get("conda_env")
    master_config = data.get("master_config")
    
    master_path = Path(master_config)
    pwd = master_path.parent 
    
    try:
        bash_command = """
            if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
                source ~/miniconda3/etc/profile.d/conda.sh
            elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
                source ~/anaconda3/etc/profile.d/conda.sh
            else
                echo "conda.sh not found!" >&2
                exit 1
            fi
            conda activate {env} && python run.py
            """.format(env=conda_env).strip()
            
        with open("master.log", "w") as log:
            subprocess.Popen(
                ["bash", "-c", bash_command],
                stdout=log,
                stderr=log,
                cwd=pwd,
                preexec_fn=os.setpgrp,
            )
        return jsonify(
            {
                "success": True,
                "message": "Master service started successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to start Master service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/start-slaver", methods=["POST"])
def start_slaver():
    data = request.json

    conda_env = data.get("conda_env")
    
    slaver_config = data.get("slaver_config")
    
    slaver_path = Path(slaver_config)
    
    pwd = slaver_path.parent 

    try:
        bash_command = """
            if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
                source ~/miniconda3/etc/profile.d/conda.sh
            elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
                source ~/anaconda3/etc/profile.d/conda.sh
            else
                echo "conda.sh not found!" >&2
                exit 1
            fi
            conda activate {env} && python run.py
            """.format(env=conda_env).strip()
            
        with open("slaver.log", "w") as log:
            subprocess.Popen(
                ["bash", "-c", bash_command],
                stdout=log,
                stderr=log,
                cwd=pwd,
                preexec_fn=os.setpgrp,
            )
        return jsonify(
            {
                "success": True,
                "message": "slaver service started successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Failed to start slaver service: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/get_tool_config", methods=["POST"])
def get_tool_config():
    
    data = request.json
    
    slaver_config = data.get("slaver_config")
    
    slaver_path = Path(slaver_config)
    
    parent = slaver_path.parent 
    
    try:
        tool_path = f"{parent}/robot_tools/skill.py"
        with open(tool_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=tool_path)
        results = []
        for node in tree.body:
            if (isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.FunctionDef) ) and node.decorator_list:
                for deco in node.decorator_list:
                    if (
                        isinstance(deco, ast.Call)
                        and hasattr(deco.func, "attr")
                        and deco.func.attr == "tool"
                    ):
                        func_name = node.name
                        docstring = ast.get_docstring(node) or ""
                        parameters = []

                        total_args = len(node.args.args)
                        defaults = [None] * (total_args - len(node.args.defaults)) + node.args.defaults

                        for arg, default in zip(node.args.args, defaults):
                            if arg.arg == "self":
                                continue
                            arg_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                            default_val = ast.unparse(default) if default else None

                            parameters.append({
                                "name": arg.arg,
                                "type": arg_type,
                                "default": default_val
                            })

                        results.append({
                            "name": func_name,
                            "description": docstring.strip(),
                            "parameters": parameters
                        })
        
        return jsonify({
            "success": True,
            "data": results
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "data": []
            }),400


@app.route("/api/save_tool_config", methods=["POST"])
def save_tool_config():
    try:
        config = request.json
        required_sections = ["navigate", "grasp", "place"]
        for section in required_sections:
            if section not in config:
                return jsonify({"error": f"Missing section: {section}"}), 400

        with open("/workspace/RoboOS/slaver/robot_tools/robot_profile.yaml", "w") as f:
            json.dump(config, f, indent=4)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False, use_reloader=False)
