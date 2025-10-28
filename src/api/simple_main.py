"""
Simple FastAPI Server for Visionary Prompt Architect
Author: Balaji Koneti

Simplified version to get the system running quickly.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Simple models
class CompletionRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class CompletionResponse(BaseModel):
    content: str
    model: str
    response_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Create FastAPI app
app = FastAPI(
    title="Visionary Prompt Architect API",
    description="Advanced AI-powered prompt engineering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

# Simple completion endpoint
@app.post("/completion", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Generate completion using a simple mock response"""
    start_time = time.time()
    
    # Mock response for now
    mock_response = f"Mock response for prompt: '{request.prompt[:50]}...' using model {request.model}"
    
    response_time = time.time() - start_time
    
    return CompletionResponse(
        content=mock_response,
        model=request.model,
        response_time=response_time,
        timestamp=datetime.now()
    )

# Minimal browser UI
@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Visionary Prompt Architect</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; max-width: 920px; }
    h1 { margin: 0 0 12px; font-size: 22px; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin: 12px 0; }
    label { font-weight: 600; display: block; margin-bottom: 8px; }
    textarea { width: 100%; min-height: 140px; padding: 10px; font-family: inherit; font-size: 14px; }
    input, select { padding: 8px; font-size: 14px; }
    button { padding: 10px 16px; background: #111827; color: #fff; border: 0; border-radius: 8px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .muted { color: #6b7280; font-size: 12px; }
    pre { background: #0b1020; color: #e5e7eb; padding: 12px; border-radius: 8px; overflow:auto; }
    .result { min-height: 60px; }
  </style>
  <script>
  async function runCompletion() {
    const btn = document.getElementById('run');
    const out = document.getElementById('output');
    const err = document.getElementById('error');
    err.textContent = '';
    out.textContent = '';
    btn.disabled = true;
    try {
      const prompt = document.getElementById('prompt').value.trim();
      const model = document.getElementById('model').value;
      const temperature = parseFloat(document.getElementById('temperature').value || '0.7');
      if (!prompt) { err.textContent = 'Please enter a prompt.'; btn.disabled = false; return; }
      const res = await fetch('/completion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model, temperature })
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error('Request failed: ' + res.status + ' ' + txt);
      }
      const data = await res.json();
      out.textContent = data.content || JSON.stringify(data, null, 2);
      document.getElementById('meta').textContent = `model: ${data.model} • response_time: ${data.response_time?.toFixed?.(3) ?? data.response_time}s`;
    } catch (e) {
      err.textContent = e.message || String(e);
    } finally {
      btn.disabled = false;
    }
  }
  </script>
</head>
<body>
  <h1>Visionary Prompt Architect</h1>
  <div class=\"muted\">Quick UI to send a prompt to the API running on this server.</div>

  <div class=\"card\">
    <label for=\"prompt\">Prompt</label>
    <textarea id=\"prompt\" placeholder=\"Type your prompt here...\"></textarea>

    <div class=\"row\" style=\"margin:12px 0\"> 
      <div>
        <label for=\"model\">Model</label>
        <select id=\"model\">
          <option value=\"gpt-4o-mini\" selected>gpt-4o-mini</option>
          <option value=\"gpt-4o\">gpt-4o</option>
        </select>
      </div>
      <div>
        <label for=\"temperature\">Temperature</label>
        <input id=\"temperature\" type=\"number\" min=\"0\" max=\"2\" step=\"0.1\" value=\"0.7\" />
      </div>
      <div style=\"align-self:flex-end\">
        <button id=\"run\" onclick=\"runCompletion()\">Run</button>
      </div>
    </div>
  </div>

  <div class=\"card\">
    <div class=\"muted\" id=\"meta\"></div>
    <div id=\"error\" style=\"color:#b91c1c; margin:6px 0\"></div>
    <pre class=\"result\" id=\"output\"></pre>
  </div>

  <div class=\"muted\">Tip: switch to the full API later to call real providers.</div>
</body>
</html>"""

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Visionary Prompt Architect API",
        "version": "1.0.0",
        "author": "Balaji Koneti",
        "docs": "/docs",
        "health": "/health"
    }

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "src.api.simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
