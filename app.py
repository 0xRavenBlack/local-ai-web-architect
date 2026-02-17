import os
import json
import torch
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv, set_key
from pathlib import Path
import threading
import glob
import ollama
import random
import time
import re
import zipfile
import io

# Diffusers imports
from diffusers import StableDiffusionXLPipeline
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    DDIMScheduler
)

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
app = Flask(__name__)
ENV_FILE = '.env'
load_dotenv(ENV_FILE)

app_state = {
    "status": "idle", 
    "logs": [], 
    "progress": 0,
    "current_phase": "Idle"
}



# ==========================================
# MODEL MANAGER
# ==========================================
class ModelManager:
    def __init__(self):
        self.sdxl_pipe = None

    def get_ollama_model(self):
        return os.getenv("OLLAMA_MODEL", "llama3")

    def get_sdxl_path(self):
        return os.getenv("SDXL_MODEL_PATH", "")
    
    def get_sdxl_settings(self):
        return {
            "cfg": float(os.getenv("SDXL_CFG", "7.5")),
            "steps": int(os.getenv("SDXL_STEPS", "20")),
            "sampler": os.getenv("SDXL_SAMPLER", "Euler a")
        }

    def log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        vram_used = "N/A"
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                vram_used = f"{allocated:.2f}GB"
            except:
                vram_used = "0.00GB"
        
        formatted_msg = f"[{timestamp}][VRAM:{vram_used}] {message}"
        log_entry = f"[{level}] {formatted_msg}"
        print(log_entry)
        app_state["logs"].append(log_entry)

    def load_ollama(self):
        if self.sdxl_pipe is not None:
            self.log("Unloading SDXL to swap for Ollama...", "SYSTEM")
            del self.sdxl_pipe
            self.sdxl_pipe = None
            torch.cuda.empty_cache()
        
        model = self.get_ollama_model()
        self.log(f"Loading Ollama Model: {model}", "SYSTEM")
        self.log("Waiting for model response...", "SYSTEM")
        self.log("Ollama Ready.", "SYSTEM")

    def unload_ollama(self):
        model = self.get_ollama_model()
        self.log(f"Forcing Ollama model '{model}' out of memory...", "SYSTEM")
        try:
            ollama.chat(model=model, messages=[], keep_alive=0)
            self.log("Ollama unloaded successfully.", "SYSTEM")
        except Exception as e:
            self.log(f"Ollama unload warning: {e}", "SYSTEM")

    def load_sdxl(self):
        self.unload_ollama()
        
        if self.sdxl_pipe is None:
            path = self.get_sdxl_path()
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"SDXL Model not found at: {path}")
            
            self.log(f"Loading SDXL Pipeline from: {path}", "SYSTEM")
            self.log("Initializing weights...", "SYSTEM")
            
            self.sdxl_pipe = StableDiffusionXLPipeline.from_single_file(
                path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            # Memory Optimizations
            self.sdxl_pipe.enable_sequential_cpu_offload()
            self.sdxl_pipe.enable_attention_slicing()
            self.sdxl_pipe.enable_vae_slicing()
            
            self.apply_sampler(os.getenv("SDXL_SAMPLER", "Euler a"))
            self.log("SDXL Pipeline Ready (CPU Offload Enabled).", "SYSTEM")

    def apply_sampler(self, sampler_name):
        if self.sdxl_pipe is None: return
        
        scheduler_map = {
            "Euler a": EulerAncestralDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DDIM": DDIMScheduler
        }
        
        scheduler_class = scheduler_map.get(sampler_name, EulerAncestralDiscreteScheduler)
        self.log(f"Applying Noise Scheduler: {sampler_name}", "SDXL")
        self.sdxl_pipe.scheduler = scheduler_class.from_config(self.sdxl_pipe.scheduler.config)

    def sdxl_callback(self, pipe, step, timestep, callback_kwargs):
        total_steps = pipe.num_timesteps
        if step % 5 == 0 or step == total_steps - 1:
            self.log(f"Denoising Step {step+1}/{total_steps}", "SDXL")
        return callback_kwargs

    def generate_image(self, prompt, output_path, width, height, seed):
        if self.sdxl_pipe is None:
            self.load_sdxl()
        
        settings = self.get_sdxl_settings()
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator("cuda").manual_seed(seed)
        self.log(f"Generating: {width}x{height} | Seed: {seed}", "SDXL")
        
        try:
            torch.cuda.empty_cache() # Clear before allocation
            
            image = self.sdxl_pipe(
                prompt=prompt, 
                width=width, 
                height=height,
                generator=generator,
                num_inference_steps=settings["steps"],
                guidance_scale=settings["cfg"],
                callback_on_step_end=self.sdxl_callback
            ).images[0]
            
            image.save(output_path)
            torch.cuda.empty_cache() # Clear after allocation
            self.log(f"Image saved: {output_path}", "SDXL")
        except Exception as e:
            self.log(f"Image generation failed: {e}", "ERROR")
            torch.cuda.empty_cache()
            raise e

    def run_text_agent(self, plan_json, user_idea):
        """Agent 1: Specialized in copywriting and content."""
        self.log("Starting Text Content Agent...", "AGENT")
        
        prompt = f"""
        You are a Professional Copywriter. Based on the plan, write actual text content.
        
        ORIGINAL USER REQUEST: "{user_idea}"
        Plan: {json.dumps(plan_json)}
        
        CRITICAL:
        1. Output ONLY valid JSON. No newlines inside strings.
        2. ALL content MUST relate to "{user_idea}".
        3. Expand on 'sections' content. Make it engaging.
        4. Refine image prompts to include keywords from "{user_idea}".
        
        JSON Structure Required:
        {{ "title": "", "description": "", "images": [{{ "type": "", "prompt": "", "width": 0, "height": 0, "seed": -1 }}], "sections": [{{ "title": "", "content": "" }}] }}
        """
        
        response = ollama.chat(
            model=self.get_ollama_model(), 
            messages=[{'role': 'user', 'content': prompt}],
            options={"num_ctx": 8192}
        )
        raw_content = response['message']['content']
        
        try:
            return clean_and_parse_json(raw_content)
        except Exception as e:
            self.log(f"Text Agent JSON Error: {e}", "ERROR")
            return plan_json

    def run_design_agent(self, plan_json, image_paths, user_idea):
        """Agent 2: Specialized in Frontend Dev and CSS."""
        self.log("Starting Design Agent...", "AGENT")
        
        image_context = []
        for i, img_data in enumerate(plan_json.get('images', [])):
            # Use relative path for HTML
            rel_path = f"images/gen_{i}.png"
            image_context.append({
                "index": i,
                "type": img_data.get('type', 'generic'),
                "path": rel_path
            })
        
        prompt = f"""
        You are a Senior Frontend Engineer. Create a single HTML file with embedded CSS/JS.
        
        WEBSITE TOPIC: "{user_idea}"
        Content: {json.dumps(plan_json)}
        Images to embed: {json.dumps(image_context)}
        
        CRITICAL REQUIREMENTS:
        1. Use EXACT image paths from the "path" field above (e.g., "images/gen_0.png").
        2. Hero Image (type='hero'): Full width banner at top.
        3. Gallery Images (type='gallery'): CSS Grid section with 3 columns.
        4. Feature Images (type='feature'): Small cards with icons.
        5. ALL images MUST be embedded with <img src="images/gen_X.png"> tags.
        6. Add fade-in animations on scroll.
        7. Color scheme should match "{user_idea}" theme.
        
        Output ONLY HTML code. No markdown wrappers.
        """
        
        response = ollama.chat(
            model=self.get_ollama_model(), 
            messages=[{'role': 'user', 'content': prompt}],
            options={"num_ctx": 8192}
        )
        html_content = response['message']['content']
        
        # Clean markdown code blocks
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0]
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0]
            
        return html_content

manager = ModelManager()
# ... imports ...

# ==========================================
# HELPER: ROBUST JSON PARSER
# ==========================================
def clean_and_parse_json(text):
    """
    Attempts to clean and parse JSON from LLM output.
    Fixes common issues like newlines inside strings.
    """
    manager.log("Attempting to sanitize LLM JSON output...", "DEBUG")
    
    # 1. Extract content between { } if it exists, ignoring surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = text

    # 2. Aggressive Cleaning: 
    # Sometimes LLMs put literal newlines inside quotes. 
    # We try to escape them: replace raw \n with \\n inside the string logic.
    # This is a heuristic but works well for SDXL prompts.
    
    # A simple regex to find "key": "value" and clean the value part
    def fix_newlines_in_values(m):
        val = m.group(1)
        # Replace actual newlines with escaped newlines
        val = val.replace('\n', '\\n') 
        return f': "{val}"' # Assuming the key part is fine, rebuild the pair

    # This regex is a bit simple, looking for : " ... " pairs. 
    # For complex nested JSON it might need a proper parser, but for our plan it's usually enough.
    # Note: This is a fallback. Ideally, the prompt fixes it.
    
    try:
        # First try standard parse
        return json.loads(json_str)
    except json.JSONDecodeError:
        manager.log("Standard parse failed, attempting aggressive cleaning...", "DEBUG")
        try:
            # If standard fails, try replacing all literal newlines inside the string with spaces 
            # (often valid for prompts, just loses formatting)
            # This is safer than trying to escape them blindly.
            clean_str = json_str.replace('\n', ' ')
            return json.loads(clean_str)
        except Exception as e:
            manager.log(f"Failed to parse JSON even after cleaning: {e}", "ERROR")
            # Log the bad string to file for debugging
            with open("debug_json_error.txt", "w") as f:
                f.write(text)
            raise e

def cleanup_output_folder():
    """Remove all files from output folder before new generation"""
    try:
        # Remove all images
        if os.path.exists("output/images"):
            for file in os.listdir("output/images"):
                file_path = os.path.join("output/images", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    manager.log(f"Cleaned up: {file_path}", "DEBUG")
        
        # Remove old HTML
        if os.path.exists("output/index.html"):
            os.remove("output/index.html")
            manager.log("Cleaned up: output/index.html", "DEBUG")
        
        manager.log("Output folder cleaned successfully", "SYSTEM")
    except Exception as e:
        manager.log(f"Cleanup warning: {e}", "ERROR")

# ==========================================
# AGENT LOGIC (UPDATED)
# ==========================================

def run_text_agent(self, plan_json):
    """
    Agent 1: Specialized in copywriting, tone, and SEO.
    """
    self.log("Starting Text Content Agent...", "AGENT")
        
    prompt = f"""
    You are a Professional Copywriter and Content Strategist.
    Based on the following website plan, write the actual text content.
        
    Plan: {json.dumps(plan_json)}
        
    CRITICAL INSTRUCTIONS:
    1. Output ONLY valid JSON.
    2. Do NOT use newlines inside string values.
    3. Expand on the 'sections' content. Make it engaging.
    4. Refine image prompts if necessary to match the new text tone.
        
    JSON Structure Required (match input structure but fill content):
    {{
        "title": "String",
        "description": "String",
        "images": [{{ "prompt": "String", "width": 1024, "height": 1024, "seed": -1 }}],
        "sections": [{{ "title": "String", "content": "String" }}]
    }}
    """
        
    response = ollama.chat(
            model=self.get_ollama_model(), 
            messages=[{'role': 'user', 'content': prompt}],
            options={"num_ctx": 4096}
    )
    raw_content = response['message']['content']
        
    try:
        return clean_and_parse_json(raw_content)
    except Exception as e:
        self.log(f"Text Agent JSON Error: {e}", "ERROR")
    return plan_json # Fallback to original plan

def run_design_agent(self, plan_json, image_paths):
    self.log("Starting Design Agent...", "AGENT")
        
    # Map images to their types for the designer
    image_context = []
    for i, img_data in enumerate(plan_json.get('images', [])):
        image_context.append(f"Image {i}: Type='{img_data.get('type')}', Path='{image_paths[i]}'")
        
    prompt = f"""
    You are a Senior Frontend Engineer.
    Create a single HTML file with embedded CSS and JS.
        
    Content Data: {json.dumps(plan_json)}
    Image Manifest:
    {json.dumps(image_context)}
        
    DESIGN REQUIREMENTS:
    1. Hero Image: Full width or large centered banner with parallax effect.
    2. Gallery Images: Display in a responsive CSS Grid (3 columns on desktop, 1 on mobile). Add hover zoom effect.
    3. Feature Images: Display as small icons/cards above text sections.
    4. Animations: Use IntersectionObserver to fade elements in as user scrolls.
    5. Colors: Extract a color palette from the concept "{plan_json.get('title')}" and apply consistently.
        
    Output ONLY the HTML code.
    """
        
    response = ollama.chat(
            model=self.get_ollama_model(), 
            messages=[{'role': 'user', 'content': prompt}],
            options={"num_ctx": 4096}
    )
    html_content = response['message']['content']
        
    # Clean markdown code blocks if present
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0]
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0]   
    return html_content

def run_generation_process(user_idea):
    try:
        app_state["status"] = "running"
        app_state["logs"] = []
        app_state["progress"] = 0
        os.makedirs("output/images", exist_ok=True)

		# === NEW: CLEANUP OUTPUT FOLDER ===
        cleanup_output_folder()

        # --- PHASE 1: PLANNING & TEXT (OLLAMA) ---
        # Keep Ollama loaded for both steps to avoid reload overhead
        app_state["current_phase"] = "Planning & Content"
        manager.load_ollama() 
        app_state["progress"] = 10
        
        # 1. Planning
        plan_prompt = f"""
        You are a Web Architect. Create a JSON plan for: "{user_idea}".
        
        CRITICAL IMAGE PROMPT RULES:
        1. EVERY image prompt MUST include keywords from "{user_idea}".
        2. Be specific about visual style (e.g., "modern", "professional", "vibrant").
        3. Avoid generic terms like "beautiful" or "nice".
        
        Example for "{user_idea}" = "Coffee Shop":
        - Hero: "Professional coffee shop interior, warm lighting, espresso machine, modern cafe design, photorealistic"
        - Gallery: "Fresh roasted coffee beans closeup, latte art pouring, cozy cafe seating area"
        - Feature: "Coffee cup icon, coffee bean illustration, barista tools"
        
        IMAGE STRATEGY:
        1. One "hero" image (1024x1024) - Main visual.
        2. Three "gallery" images (1024x1024) - For portfolio/showcase section.
        3. Three "feature" images (1024x1024) - For service/icons section.
        
        Output ONLY valid JSON. No newlines in strings.
        
        JSON Structure Required:
        {{ 
            "title": "String", 
            "description": "String", 
            "images": [
                {{ "type": "hero", "prompt": "String with topic keywords", "width": 1344, "height": 768, "seed": -1 }},
                {{ "type": "gallery", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }},
                {{ "type": "gallery", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }},
                {{ "type": "gallery", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }},
                {{ "type": "feature", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }},
                {{ "type": "feature", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }},
                {{ "type": "feature", "prompt": "String with topic keywords", "width": 1024, "height": 1024, "seed": -1 }}
            ], 
            "sections": [{{ "title": "String", "content": "String" }}] 
        }}
        """
        manager.log("Architect Agent planning...", "AGENT")
        response = ollama.chat(
            model=manager.get_ollama_model(), 
            messages=[{'role': 'user', 'content': plan_prompt}],
            options={"num_ctx": 8192}
        )
        plan_json = clean_and_parse_json(response['message']['content'])
        app_state["progress"] = 20

        # 2. Text Content (Pass user_idea)
        manager.log("Handing off to Text Content Agent...", "AGENT")
        plan_json = manager.run_text_agent(plan_json, user_idea)
        app_state["progress"] = 30
        
        # Ollama tasks done. Unload to free VRAM for SDXL
        manager.unload_ollama()
        if manager.sdxl_pipe is not None:
            del manager.sdxl_pipe
            manager.sdxl_pipe = None
        torch.cuda.empty_cache()

        # --- PHASE 2: IMAGING (SDXL) ---
        app_state["current_phase"] = "Imaging"
        manager.load_sdxl()
        
        image_paths = []
        total_images = len(plan_json.get('images', []))
        
        for i, img_data in enumerate(plan_json.get('images', [])):
            # Use consistent naming that HTML expects
            img_path = f"output/images/gen_{i}.png"
            progress_per_image = 40 / max(total_images, 1)
            app_state["progress"] = int(30 + (i * progress_per_image))

            manager.generate_image(
                prompt=img_data.get('prompt', f'{user_idea} themed image'),
                output_path=img_path,
                width=img_data.get('width', 1024),
                height=img_data.get('height', 1024),
                seed=img_data.get('seed', -1)
            )
            image_paths.append(img_path)

		# Validate images exist before generating HTML
        for img_path in image_paths:
            if not os.path.exists(img_path):
                manager.log(f"WARNING: Image not found: {img_path}", "ERROR")
            else:
                manager.log(f"Image verified: {img_path}", "DEBUG")
        
        app_state["progress"] = 70
        
        # SDXL tasks done. Unload to free VRAM for final Ollama step
        if manager.sdxl_pipe is not None:
            del manager.sdxl_pipe
            manager.sdxl_pipe = None
        torch.cuda.empty_cache()

        # --- PHASE 3: DESIGN (OLLAMA) ---
        app_state["current_phase"] = "Design & Assembly"
        manager.load_ollama()
        
        manager.log("Handing off to Design Agent...", "AGENT")
        # Pass user_idea to Design Agent
        html_content = manager.run_design_agent(plan_json, image_paths, user_idea)
        
        with open("output/index.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        manager.log("Website generated successfully!", "SYSTEM")
        app_state["status"] = "completed"
        app_state["progress"] = 100
        
        # Final Cleanup
        manager.unload_ollama()
        torch.cuda.empty_cache()

    except Exception as e:
        manager.log(f"Critical Error: {str(e)}", "ERROR")
        app_state["status"] = "error"
        # Ensure cleanup on error
        try:
            manager.unload_ollama()
            if manager.sdxl_pipe is not None:
                del manager.sdxl_pipe
                manager.sdxl_pipe = None
            torch.cuda.empty_cache()
        except:
            pass

# ... (Rest of app.py: routes, etc) ...

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    config = {
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
        "sdxl_path": os.getenv("SDXL_MODEL_PATH", ""),
        "cfg": os.getenv("SDXL_CFG", "7.5"),
        "steps": os.getenv("SDXL_STEPS", "20"),
        "sampler": os.getenv("SDXL_SAMPLER", "Euler a"),
        "available_models": get_available_safetensors()
    }
    torch.cuda.empty_cache()
    return render_template('index.html', config=config)

def get_available_safetensors():
    files = glob.glob("models/*.safetensors")
    return [os.path.basename(f) for f in files]

@app.route('/api/save_config', methods=['POST'])
def save_config():
    data = request.json
    if not os.path.exists(ENV_FILE): Path(ENV_FILE).touch()
    
    set_key(ENV_FILE, "OLLAMA_MODEL", data.get('ollama_model', 'llama3'))
    set_key(ENV_FILE, "SDXL_MODEL_PATH", data.get('sdxl_path', ''))
    set_key(ENV_FILE, "SDXL_CFG", str(data.get('cfg', '7.5')))
    set_key(ENV_FILE, "SDXL_STEPS", str(data.get('steps', '20')))
    set_key(ENV_FILE, "SDXL_SAMPLER", data.get('sampler', 'Euler a'))
    
    load_dotenv(ENV_FILE, override=True)
    return jsonify({"status": "success"})

@app.route('/api/download', methods=['POST'])
def download_zip():
    """Create and send ZIP file of output folder"""
    try:
        # Create ZIP in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add HTML file
            if os.path.exists("output/index.html"):
                zf.write("output/index.html", "index.html")
            
            # Add all images
            if os.path.exists("output/images"):
                for img_file in os.listdir("output/images"):
                    img_path = os.path.join("output/images", img_file)
                    if os.path.isfile(img_path):
                        zf.write(img_path, f"images/{img_file}")
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='website.zip'
        )
    except Exception as e:
        manager.log(f"ZIP creation error: {e}", "ERROR")
        return jsonify({"error": str(e)}), 500

# Add to routes section in app.py
@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from output folder"""
    from flask import send_from_directory
    return send_from_directory('output', filename)

@app.route('/api/generate', methods=['POST'])
def generate():
    if app_state["status"] == "running": return jsonify({"error": "Busy"}), 400
    idea = request.json.get('idea', '')
    thread = threading.Thread(target=run_generation_process, args=(idea,))
    thread.start()
    return jsonify({"status": "started"})

@app.route('/api/status')
def status():
    return jsonify(app_state)

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    app.run(debug=True, port=5000)
