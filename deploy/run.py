import os
import json
import copy
import redis
import  socket
import subprocess
import time
from ruamel.yaml import YAML
from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path

yaml = YAML()
yaml.preserve_quotes = True 

app = Flask(__name__, static_folder="assets")

services = {
    'inference': {'pid': None, 'port': None},
    'master': {'pid': None, 'port': 6000},
    'slaver': {'pid': None}
}

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0
    

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
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
        
    return jsonify({"status": 200, "data": data})



@app.route("/saveconfig", methods=["POST"])
def saveconfig():
    try:
        data = request.json
        
        if not data or 'file_path' not in data or 'config_data' not in data:
            return jsonify({
                "status": 400,
                "message": "Required parameter missing: file_path 、config_data"
            }), 400
        
        file_path = data['file_path']
        config_data = data['config_data']
        node = data.get("node", "master")
        # TODO Save to environment variable
    
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        def recursive_update(yaml_obj, new_values):
            for key, value in new_values.items():
                if key in yaml_obj:
                    if isinstance(value, dict) and isinstance(yaml_obj[key], dict):
                        recursive_update(yaml_obj[key], value)
                    elif isinstance(value, list) and isinstance(yaml_obj[key], list):
                        yaml_obj[key] = copy.deepcopy(value)  
                    else:
                        yaml_obj[key] = value

        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.load(f)

        recursive_update(yaml_data, config_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f)
    
        
        return jsonify({
            "status": 200,
            "message": "Configuration file saved successfully"
        })
    
    except Exception as e:
        return jsonify({
            "status": 500,
            "message": f"Server error: {str(e)}"
        }), 500
        

@app.route('/api/validate-config', methods=['POST'])
def validate_config():
    data = request.json
    required_fields = ['model_path', 'inference_port', 'master_config', 'slaver_config']
    
    for field in required_fields:
        if not data.get(field):
            return jsonify({
                "success": False,
                "message": f"Missing necessary fields: {field}"
            }), 400

    for file_path in  [data['model_path'], data['master_config'], data['slaver_config']]:
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "message": f"The model path does not exist: {file_path}"
            }), 400
    
    # Check if the port is occupied
    for port in [data['inference_port'], 5000]:
        if is_port_in_use(int(port)):
            return jsonify({
                "success": False,
                "message": f"The port is already occupied: {port}"
            }), 400
    
    # master
    master_config = data["master_config"]
    with open(master_config, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
    # communicator 
    communicator = data.get("communicator", None)
    if not communicator:
        return jsonify({
                "success": False,
                "message": f"Lack of communicator configuration: {port}"
            }), 400
    try:
        r = redis.StrictRedis(
            host=communicator["HOST"],
            port=communicator["PORT"],
            password=communicator["PASSWORD"],
            db=communicator["DB"],
            socket_connect_timeout=5  
        )
        r.ping()
    except Exception as e:
        
        return jsonify({
                "success": False,
                "message": f"communicator connection failed: {e}"
            }), 400


    return jsonify({
        "success": True,
        "message": "Configuration verification passed",
        "validated_config": data
    })

@app.route('/api/start-inference', methods=['POST'])
def start_inference():
    data = request.json
    
    model_path = data["master_config"]
    port = data["inference_port"]
    try:
        command = [
            "nohup", "vllm", "serve", model_path,
            "--trust-remote-code",
            "--served-model-name", "robobrain",
            "--gpu-memory-utilization", "0.92",
            "--tensor-parallel-size", "1",
            "--port", str(port),
            "--max-model-len", "10000",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes"
        ]

        # log
        log_file = f"vllm_{port}.log"
        with open(log_file, "w") as f:
            subprocess.Popen(command, stdout=f, stderr=f, preexec_fn=os.setpgrp)
        
        return jsonify({
            "success": True,
            "message": f"Model robobrain has been started, listening on port {port}, log file:{log_file}",
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to start inference service: {str(e)}"
        }), 500

@app.route('/api/start-master', methods=['POST'])
def start_master():
    try:
        with open("master.log", "w") as log:
            subprocess.Popen(
                ["nohup", "python", 'run.py'],
                stdout=log,
                stderr=log,
                cwd="/home/gw/code/RoboOS/master",
                preexec_fn=os.setpgrp
            )
        return jsonify({
            "success": True,
            "message": "Master service started successfully",
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to start Master service: {str(e)}"
        }), 500

@app.route('/api/start-slaver', methods=['POST'])
def start_slaver():
    try:
        with open("slaver.log", "w") as log:
            subprocess.Popen(
                ["nohup", "python", 'run.py'],
                stdout=log,
                stderr=log,
                cwd="/home/gw/code/RoboOS/slaver",
                preexec_fn=os.setpgrp
            )
        return jsonify({
            "success": True,
            "message": "slaver service started successfully",
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to start slaver service: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)