import os
import json
import asyncio
import smtplib
import uuid
import torch
import sentencepiece
import sphn
import numpy as np
from datetime import datetime
from email.message import EmailMessage

from fastapi import FastAPI, WebSocket, Request, Form, Response, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from PyPDF2 import PdfReader
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Moshi specific imports
from moshi.models import loaders, LMGen
from kernel import Kernel

load_dotenv()

# --- DATABASE SETUP ---
Base = declarative_base()
engine = create_engine('sqlite:///dynamo_api.db')
SessionLocal = sessionmaker(bind=engine)

class Workflow(Base):
    __tablename__ = 'workflow'
    id = Column(Integer, primary_key=True)
    intent_label = Column(String(100), unique=True)
    description = Column(Text)
    steps = relationship("Step", backref="workflow", order_by="Step.order_index")

class Step(Base):
    __tablename__ = 'step'
    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey('workflow.id'))
    action_type = Column(String(50))
    selector = Column(Text)
    order_index = Column(Integer)

Base.metadata.create_all(bind=engine)

# --- MOSHI SERVICE CLASS ---
class MoshiService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mimi = None
        self.moshi = None
        self.lm_gen = None
        self.text_tokenizer = None
        self.frame_size = None

    def initialize(self):
        print(f"📡 Loading Moshi on {self.device}...")
        mimi_weight = hf_hub_download("kyutai/moshika-pytorch-bf16", loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        moshi_weight = hf_hub_download("kyutai/moshika-pytorch-bf16", loaders.MOSHI_NAME)
        self.moshi = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.lm_gen = LMGen(self.moshi, temp=0.8, top_k=250)
        
        tokenizer_config = hf_hub_download("kyutai/moshika-pytorch-bf16", loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_config)
        
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        print("✅ Moshi Initialized.")

# Initialize Global Service
moshi_svc = MoshiService()
moshi_svc.initialize()

app = FastAPI()
pending_leads = {}

# --- UTILS ---
async def send_report_email(recipient, lead_name, transcript):
    msg = EmailMessage()
    msg.set_content(f"Lead: {lead_name}\n\nTranscript:\n{transcript}")
    msg['Subject'] = f"Moshi Demo Report: {lead_name}"
    msg['From'] = os.getenv("EMAIL_SENDER")
    msg['To'] = recipient
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
            server.send_message(msg)
    except Exception as e: print(f"Email Error: {e}")

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def register_page():
    with open("register.html", "r") as f: return f.read()

@app.post("/verify")
async def verify_lead(name: str = Form(...), email: str = Form(...), phone: str = Form(...)):
    token = str(uuid.uuid4())
    pending_leads[token] = {"name": name, "email": email, "phone": phone}
    link = f"http://localhost:8000/demo?token={token}"
    print(f"Access Link: {link}")
    return HTMLResponse("<h1>Check terminal for link</h1>")

@app.get("/demo", response_class=HTMLResponse)
async def demo_page(token: str):
    if token not in pending_leads: return RedirectResponse("/")
    with open("index.html", "r") as f: return f.read()

# --- CORE WEBSOCKET ORCHESTRATOR ---
@app.websocket("/ws/{token}")
async def s2s_handler(websocket: WebSocket, token: str):
    await websocket.accept()
    lead = pending_leads.get(token)
    
    # Init Browser via Kernel
    kernel = Kernel(api_key=os.getenv("KERNEL_API_KEY"))
    try:
        k_browser = kernel.browsers.create() 
        await websocket.send_json({"type": "browser_url", "url": k_browser.browser_live_view_url})
    except: k_browser = None

    # Moshi Session State
    opus_in = sphn.OpusStreamReader(moshi_svc.mimi.sample_rate)
    opus_out = sphn.OpusStreamWriter(moshi_svc.mimi.sample_rate)
    moshi_svc.mimi.reset_streaming()
    moshi_svc.lm_gen.reset_streaming()
    
    transcript = []
    closed = False

    async def recv_loop():
        nonlocal closed
        try:
            while not closed:
                data = await websocket.receive()
                if "bytes" in data:
                    # Moshi client sends audio with a 0x01 prefix
                    raw_data = data["bytes"]
                    if raw_data[0] == 0x01:
                        opus_in.append_bytes(raw_data[1:])
                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "close_session":
                        closed = True
        except WebSocketDisconnect: closed = True

    async def inference_loop():
        nonlocal closed
        all_pcm = None
        try:
            with torch.no_grad():
                while not closed:
                    await asyncio.sleep(0.001)
                    pcm = opus_in.read_pcm()
                    if pcm is None or pcm.size == 0: continue
                    
                    all_pcm = np.concatenate((all_pcm, pcm)) if all_pcm is not None else pcm
                    
                    while all_pcm.shape[-1] >= moshi_svc.frame_size:
                        chunk = all_pcm[:moshi_svc.frame_size]
                        all_pcm = all_pcm[moshi_svc.frame_size:]
                        chunk_t = torch.from_numpy(chunk).to(moshi_svc.device)[None, None]
                        
                        codes = moshi_svc.mimi.encode(chunk_t)
                        for c in range(codes.shape[-1]):
                            tokens = moshi_svc.lm_gen.step(codes[:, :, c:c+1])
                            if tokens is None: continue

                            # Decode Audio
                            out_pcm = moshi_svc.mimi.decode(tokens[:, 1:]).cpu()
                            opus_out.append_pcm(out_pcm[0, 0].detach().numpy())

                            # Decode Text
                            text_tok = tokens[0, 0, 0].item()
                            if text_tok not in (0, 3):
                                text = moshi_svc.text_tokenizer.id_to_piece(text_tok).replace("▁", " ")
                                transcript.append(text)
                                await websocket.send_bytes(b"\x02" + bytes(text, "utf-8"))
        except Exception as e:
            print(f"Inference Error: {e}")
            closed = True

    async def send_loop():
        nonlocal closed
        try:
            while not closed:
                await asyncio.sleep(0.005)
                audio_bytes = opus_out.read_bytes()
                if audio_bytes:
                    await websocket.send_bytes(b"\x01" + audio_bytes)
        except: closed = True

    await asyncio.gather(recv_loop(), inference_loop(), send_loop())
    
    # Cleanup & Email
    full_text = "".join(transcript)
    await send_report_email(os.getenv("SITE_OWNER_EMAIL"), lead['name'], full_text)
    if k_browser: k_browser.close()
