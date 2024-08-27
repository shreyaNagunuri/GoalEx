from flask import Flask, render_template, request, redirect, url_for, jsonify
import sys
import json
import os
import time
import datetime
import uuid
import subprocess
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        clustering_goal = request.form["clusteringGoal"]
        overlap_penalty = request.form["overlapPenalty"]
        num_clusters = request.form["numClusters"]
        max_cluster_size = request.form["maxClusterSize"]
        
        file = request.files["fileUpload"]
        if file and file.filename.endswith('.txt'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
                
            with open(file_path, 'r') as f:
                file_content = f.read()
                
            data = {
                "goal": clustering_goal,
                "texts": [file_content],
                "example_descriptions": []
            }                
            # Save JSON to a file
            time = f"/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            folder_name = file.filename + time
            
            folder_path = 'processed_data/' + folder_name
            file_path = folder_path 
            
            os.makedirs(folder_path, exist_ok=True)
            
            json_dump = file_path + '/data.json'
            with open(json_dump, 'w') as f:
                json.dump(data, f, indent=4)

            exp_directory = 'experiments/' + folder_name
            os.makedirs(exp_directory, exist_ok=True)
             
            command = [
                "python", "/data/ersp2023/GoalEx/src/iterative_cluster.py",
                "--data_path", folder_path,
                "--exp_dir", exp_directory,
                "--subsample", "1024",
                "--proposer_model", "gpt-4",
                "--assigner_name", "google/flan-t5-xl",
                "--proposer_num_descriptions_to_propose", "30",
                "--assigner_for_final_assignment_template", "/data/ersp2023/GoalEx/template/t5_multi_assigner_one_output.txt",
                "--cluster_overlap_penalty", overlap_penalty,
                "--max_cluster_fraction", max_cluster_size,
                "--cluster_num_clusters", num_clusters,
                "--turn_off_approval_before_running",
                "--verbose"
            ]
            
            result = subprocess.run(command)
            
            result_path = os.path.join("./uploads", "cluster_info.txt")
            with open(result_path, 'r') as f:
                result_txt_file = f.read()
            
            result_png_file = os.path.join("./uploads", "plot.png")
            
            return render_template("index.html", txt_file=result_txt_file, png_file=result_png_file)
        
        else:
            return "Please upload a .txt file.", 400
    else:    
        return render_template("index.html")

@app.route('/clear_cache')
def clear_cache():
    with app.app_context():
        app.jinja_env.cache.clear()
    return "Cache cleared!"

if __name__=='__main__':
    app.run(debug=True)

