import sys
import os
import threading
import time
import csv
import io
import psutil
import numpy as np
from flask import Flask, render_template_string, Response, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.preprocess import preprocess
from src.dqn_agent import DQNAgent

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

SAVE_PATH = "models/dqn_firewall.keras"

firewall_started = False
firewall_lock    = threading.Lock()
is_paused        = False
is_stopped       = False
logs             = []
session_stats    = {}
total_test_size  = 0

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AdaptiveShield — AI Firewall Console</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ═══════════════════════════════════════════════
   DESIGN TOKENS — Deep Crimson Firewall Theme
   ═══════════════════════════════════════════════ */
:root{
  --bg:       #04010a;
  --bg2:      #080414;
  --panel:    #0c0618;
  --panel2:   #100820;
  --border:   #2a0a4a;
  --border2:  #4a0a2a;
  --fire:     #ff4500;
  --fire2:    #ff6a00;
  --fire3:    #ffaa00;
  --cyan:     #00e5ff;
  --purple:   #bf00ff;
  --green:    #00ff9d;
  --red:      #ff1744;
  --yellow:   #ffd600;
  --orange:   #ff6d00;
  --text:     #e8d5ff;
  --muted:    #6a4a8a;
  --glow-f:   0 0 20px rgba(255,69,0,0.5);
  --glow-c:   0 0 16px rgba(0,229,255,0.4);
  --glow-p:   0 0 16px rgba(191,0,255,0.4);
  --font-hud: 'Orbitron',sans-serif;
  --font-body:'Rajdhani',sans-serif;
  --font-mono:'Share Tech Mono',monospace;
}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;overflow-x:hidden;}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--font-body);font-size:14px;
}

/* ═══ GLOBAL GRID BG ═══ */
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(255,69,0,0.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,69,0,0.025) 1px,transparent 1px);
  background-size:60px 60px;
}
body::after{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 0%,rgba(191,0,255,0.06) 0%,transparent 65%),
             radial-gradient(ellipse at 0% 100%,rgba(255,69,0,0.05) 0%,transparent 60%),
             radial-gradient(ellipse at 100% 50%,rgba(0,229,255,0.04) 0%,transparent 60%);
}

/* ════════════════════════════════════════════════
   STAGE 1 — PURE BLACK BOOT
   ════════════════════════════════════════════════ */
#stage-boot{
  position:fixed;inset:0;z-index:2000;
  background:#000;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:0;
}
.boot-line{
  font-family:var(--font-mono);font-size:12px;
  color:#333;letter-spacing:1px;
  opacity:0;transform:translateX(-8px);
  transition:opacity 0.3s,transform 0.3s,color 0.3s;
  margin-bottom:3px;
  white-space:nowrap;
}
.boot-line.show{opacity:1;transform:none;}
.boot-line.ok{color:#00ff9d;}
.boot-line.warn{color:#ffd600;}
.boot-line.fire{color:#ff4500;}
.boot-cursor{
  font-family:var(--font-mono);font-size:12px;color:#ff4500;
  animation:blink 0.7s step-end infinite;margin-top:4px;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* ════════════════════════════════════════════════
   STAGE 2 — CINEMATIC LOGO
   ════════════════════════════════════════════════ */
#stage-logo{
  position:fixed;inset:0;z-index:1900;
  background:#000;
  display:none;flex-direction:column;align-items:center;justify-content:center;
  overflow:hidden;
}
/* Scanlines */
#stage-logo::before{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(
    0deg,transparent,transparent 3px,
    rgba(0,0,0,0.18) 3px,rgba(0,0,0,0.18) 4px
  );z-index:1;
}
/* Vignette */
#stage-logo::after{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,0.85) 100%);
  z-index:1;
}

/* SVG shield canvas */
#logo-canvas-wrap{
  position:relative;width:220px;height:220px;
  display:flex;align-items:center;justify-content:center;
  margin-bottom:40px;z-index:2;
}
#logo-svg{opacity:0;}
.logo-ring{
  position:absolute;inset:0;border-radius:50%;
  border:1px solid rgba(255,69,0,0.4);
  opacity:0;
}

#logo-text-wrap{position:relative;z-index:2;text-align:center;}
.logo-brand{
  font-family:var(--font-hud);font-size:52px;font-weight:900;
  letter-spacing:6px;line-height:1;
  background:linear-gradient(180deg,#ffffff 0%,#ff6a00 60%,#ff4500 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
  opacity:0;filter:blur(16px);
}
.logo-rl-tag{
  font-family:var(--font-mono);font-size:11px;letter-spacing:5px;
  color:var(--cyan);text-transform:uppercase;
  margin-top:10px;opacity:0;
}
.logo-divider{
  width:0;height:1px;margin:16px auto;
  background:linear-gradient(90deg,transparent,var(--fire),var(--fire2),transparent);
  opacity:0;
}
.logo-motto{
  font-family:var(--font-hud);font-size:13px;font-weight:400;
  letter-spacing:3px;color:var(--text);
  opacity:0;filter:blur(6px);
}
.logo-motto em{color:var(--fire2);font-style:normal;}

/* ════════════════════════════════════════════════
   STAGE 3 — LOADING
   ════════════════════════════════════════════════ */
#stage-load{
  position:fixed;inset:0;z-index:1800;
  background:var(--bg);
  display:none;flex-direction:column;align-items:center;justify-content:center;
}
.load-shield{font-size:44px;margin-bottom:20px;animation:shield-pulse 1.8s ease-in-out infinite;}
@keyframes shield-pulse{
  0%,100%{filter:drop-shadow(0 0 6px #ff4500);}
  50%{filter:drop-shadow(0 0 22px #ff4500) drop-shadow(0 0 44px #ff6a00);}
}
.load-title{
  font-family:var(--font-hud);font-size:18px;font-weight:700;
  color:var(--fire);letter-spacing:4px;margin-bottom:4px;
}
.load-sub{font-family:var(--font-mono);font-size:10px;color:var(--muted);letter-spacing:3px;margin-bottom:28px;}
.load-bar-track{
  width:340px;height:3px;background:#1a0830;
  border-radius:2px;overflow:hidden;margin-bottom:10px;
  box-shadow:0 0 8px rgba(255,69,0,0.2);
}
.load-bar-fill{
  height:100%;width:0%;border-radius:2px;
  background:linear-gradient(90deg,var(--purple),var(--fire),var(--fire2),var(--fire3));
  transition:width 0.4s ease;
  box-shadow:0 0 12px var(--fire);
}
.load-msg{font-family:var(--font-mono);font-size:10px;color:var(--muted);letter-spacing:1px;margin-bottom:24px;}
.load-steps{display:flex;flex-direction:column;gap:5px;width:340px;}
.load-step{
  font-family:var(--font-mono);font-size:10px;
  padding:5px 10px;border-radius:3px;
  border:1px solid var(--border);color:var(--muted);
  display:flex;align-items:center;gap:8px;transition:all 0.3s;
}
.load-step.done  {border-color:var(--green);color:var(--green);background:rgba(0,255,157,0.04);}
.load-step.active{border-color:var(--fire); color:var(--fire); background:rgba(255,69,0,0.06);}
.load-step-dot{width:5px;height:5px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:all 0.3s;}
.load-step.done   .load-step-dot{background:var(--green);}
.load-step.active .load-step-dot{background:var(--fire);animation:blink 0.8s infinite;}

/* ════════════════════════════════════════════════
   MAIN DASHBOARD
   ════════════════════════════════════════════════ */
#main-dash{display:none;position:relative;z-index:1;padding:12px;min-height:100vh;}

/* ── TOPBAR ── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 20px;
  background:linear-gradient(90deg,rgba(12,6,24,0.98),rgba(16,8,32,0.98));
  border:1px solid var(--border);border-radius:6px;
  margin-bottom:10px;
  box-shadow:0 0 0 1px rgba(255,69,0,0.08) inset, var(--glow-f);
  position:relative;overflow:hidden;
}
.topbar::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--fire),var(--fire2),transparent);
}
.brand{display:flex;align-items:center;gap:14px;}
.brand-icon{
  width:42px;height:42px;
  background:linear-gradient(135deg,rgba(255,69,0,0.15),rgba(255,106,0,0.25));
  border:1px solid rgba(255,69,0,0.4);border-radius:6px;
  display:flex;align-items:center;justify-content:center;font-size:22px;
  box-shadow:var(--glow-f);animation:shield-pulse 2.5s ease-in-out infinite;
}
.brand-name{
  font-family:var(--font-hud);font-size:18px;font-weight:900;
  letter-spacing:3px;
  background:linear-gradient(90deg,#fff,var(--fire2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.brand-sub{font-family:var(--font-mono);font-size:9px;color:var(--muted);letter-spacing:3px;margin-top:2px;}
.topbar-right{display:flex;align-items:center;gap:12px;}
.threat-box{
  display:flex;flex-direction:column;align-items:center;
  padding:6px 14px;border-radius:4px;
  border:1px solid var(--red);background:rgba(255,23,68,0.06);
  transition:all 0.4s;min-width:90px;
}
.threat-label{font-family:var(--font-mono);font-size:8px;color:var(--muted);letter-spacing:2px;margin-bottom:2px;}
.threat-value{font-family:var(--font-hud);font-size:11px;font-weight:700;color:var(--red);letter-spacing:1px;transition:color 0.4s;}
.clock{font-family:var(--font-mono);font-size:16px;color:var(--fire2);letter-spacing:3px;text-shadow:var(--glow-f);}
.status-pill{
  display:flex;align-items:center;gap:7px;
  padding:5px 14px;border-radius:20px;
  border:1px solid var(--green);background:rgba(0,255,157,0.05);
  font-family:var(--font-mono);font-size:10px;color:var(--green);letter-spacing:1px;
  transition:all 0.4s;
}
.pulse-dot{
  width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:pulse-anim 1.5s infinite;transition:background 0.4s;
}
@keyframes pulse-anim{0%,100%{opacity:1;box-shadow:0 0 0 0 currentColor;}50%{opacity:0.5;box-shadow:0 0 0 4px transparent;}}

/* ── CONTROLS ── */
.controls{
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:8px 14px;
  background:rgba(12,6,24,0.95);border:1px solid var(--border);
  border-radius:6px;margin-bottom:10px;
  position:relative;overflow:hidden;
}
.controls::before{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--border2),transparent);
}
.ctrl-label{font-family:var(--font-mono);font-size:9px;color:var(--muted);letter-spacing:2px;margin-right:4px;}
.btn{
  display:flex;align-items:center;gap:5px;
  padding:6px 14px;border-radius:4px;
  font-family:var(--font-body);font-size:12px;font-weight:600;letter-spacing:1px;
  cursor:pointer;border:1px solid transparent;transition:all 0.2s;
}
.btn-stop    {background:rgba(255,23,68,0.12);  border-color:var(--red);    color:var(--red);}
.btn-stop:hover    {background:rgba(255,23,68,0.28);box-shadow:0 0 10px rgba(255,23,68,0.3);}
.btn-pause   {background:rgba(255,214,0,0.10);  border-color:var(--yellow); color:var(--yellow);}
.btn-pause:hover   {background:rgba(255,214,0,0.22);}
.btn-resume  {background:rgba(0,255,157,0.08);  border-color:var(--green);  color:var(--green);}
.btn-resume:hover  {background:rgba(0,255,157,0.20);}
.btn-restart {background:rgba(191,0,255,0.10);  border-color:var(--purple); color:var(--purple);}
.btn-restart:hover {background:rgba(191,0,255,0.22);}
.btn-dl      {background:rgba(0,229,255,0.08);  border-color:var(--cyan);   color:var(--cyan);}
.btn-dl:hover      {background:rgba(0,229,255,0.20);}
.btn:disabled{opacity:0.3;cursor:not-allowed;box-shadow:none;}
.spacer{flex:1;}
.prog-wrap{display:flex;align-items:center;gap:8px;}
.prog-track{width:160px;height:4px;background:#1a0830;border-radius:2px;overflow:hidden;}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--fire),var(--fire3));border-radius:2px;transition:width 0.3s;box-shadow:0 0 6px var(--fire);}
.prog-lbl{font-family:var(--font-mono);font-size:9px;color:var(--muted);white-space:nowrap;}
.log-badge{font-family:var(--font-mono);font-size:9px;color:var(--muted);}

/* ── METRICS ── */
.metrics{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-bottom:10px;}
@media(max-width:1400px){.metrics{grid-template-columns:repeat(4,1fr);}}
.metric{
  background:var(--panel);border:1px solid var(--border);
  border-radius:6px;padding:11px 8px;text-align:center;
  position:relative;overflow:hidden;cursor:pointer;
  transition:all 0.25s;
}
.metric:hover{transform:translateY(-2px);border-color:rgba(255,69,0,0.3);}
.metric::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.metric::after{
  content:'⤢';position:absolute;bottom:4px;right:6px;
  font-size:9px;color:var(--muted);opacity:0.5;
}
.m-acc::before  {background:var(--green); box-shadow:0 0 8px var(--green);}
.m-total::before{background:var(--cyan);  box-shadow:0 0 8px var(--cyan);}
.m-blk::before  {background:var(--red);   box-shadow:0 0 8px var(--red);}
.m-alw::before  {background:var(--green);}
.m-fp::before   {background:var(--yellow);box-shadow:0 0 8px var(--yellow);}
.m-fn::before   {background:var(--orange);box-shadow:0 0 8px var(--orange);}
.m-f1::before   {background:var(--purple);box-shadow:0 0 8px var(--purple);}
.m-prec::before {background:var(--cyan);  box-shadow:0 0 8px var(--cyan);}
.metric-icon{font-size:15px;margin-bottom:3px;}
.metric-lbl{font-family:var(--font-mono);font-size:8px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.metric-val{font-family:var(--font-hud);font-size:20px;font-weight:700;line-height:1;transition:all 0.3s;}
.m-acc   .metric-val{color:var(--green); text-shadow:0 0 10px var(--green);}
.m-total .metric-val{color:var(--cyan);}
.m-blk   .metric-val{color:var(--red);   text-shadow:0 0 10px var(--red);}
.m-alw   .metric-val{color:var(--green);}
.m-fp    .metric-val{color:var(--yellow);}
.m-fn    .metric-val{color:var(--orange);text-shadow:0 0 8px var(--orange);}
.m-f1    .metric-val{color:var(--purple);}
.m-prec  .metric-val{color:var(--cyan);}
.metric-sub{font-family:var(--font-mono);font-size:8px;color:var(--muted);margin-top:3px;}

/* ── MAIN GRID ── */
.main-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;}
.btm-grid {display:grid;grid-template-columns:2fr 1fr;         gap:8px;}

/* ── PANEL ── */
.panel{
  background:var(--panel);border:1px solid var(--border);
  border-radius:6px;padding:13px;
  position:relative;overflow:hidden;
  transition:all 0.25s;
}
.panel::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,69,0,0.2),transparent);
}
/* corner accents */
.panel::after{
  content:'';position:absolute;bottom:0;right:0;
  width:12px;height:12px;
  border-bottom:1px solid rgba(255,69,0,0.25);
  border-right:1px solid rgba(255,69,0,0.25);
}
.panel:hover{border-color:rgba(255,69,0,0.2);}
.panel-hd{
  font-family:var(--font-hud);font-size:9px;font-weight:700;
  color:var(--fire2);letter-spacing:3px;text-transform:uppercase;
  margin-bottom:10px;display:flex;align-items:center;gap:8px;cursor:pointer;
  user-select:none;
}
.panel-hd::before{
  content:'';width:2px;height:10px;
  background:linear-gradient(180deg,var(--fire),var(--purple));
  border-radius:1px;box-shadow:var(--glow-f);
}
.expand-btn{
  margin-left:auto;font-size:10px;color:var(--muted);
  transition:color 0.2s;cursor:pointer;
}
.panel-hd:hover .expand-btn{color:var(--fire2);}

/* ── CHARTS ── */
.chart-wrap{height:155px;}
.pie-wrap  {height:155px;display:flex;align-items:center;justify-content:center;}

/* ── THREAT BARS ── */
.threat-bars{display:flex;flex-direction:column;gap:6px;}
.tbar-row{display:flex;align-items:center;gap:7px;}
.tbar-lbl{width:90px;font-family:var(--font-mono);font-size:9px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tbar-track{flex:1;height:7px;background:#100820;border-radius:3px;overflow:hidden;}
.tbar-fill{height:100%;border-radius:3px;transition:width 0.5s ease;}
.tbar-n{width:30px;text-align:right;font-family:var(--font-mono);font-size:9px;}
.tbar-fn{width:28px;text-align:right;font-family:var(--font-mono);font-size:8px;color:var(--orange);}

/* ── FEED ── */
#feed{height:230px;overflow-y:auto;font-family:var(--font-mono);font-size:10px;}
#feed::-webkit-scrollbar{width:3px;}
#feed::-webkit-scrollbar-thumb{background:var(--border);}
.feed-row{
  display:grid;
  grid-template-columns:46px 38px 1fr 60px 58px 50px;
  gap:5px;align-items:center;
  padding:3px 5px;border-radius:2px;margin-bottom:2px;
  border-left:2px solid transparent;
  animation:slideIn 0.18s ease;
}
@keyframes slideIn{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}
.feed-row.r-block{background:rgba(255,23,68,0.06);  border-left-color:var(--red);}
.feed-row.r-allow{background:rgba(0,255,157,0.03);  border-left-color:var(--green);}
.feed-row.r-fn   {background:rgba(255,109,0,0.07);  border-left-color:var(--orange);}
.feed-act{text-align:center;font-size:8px;font-weight:700;padding:1px 5px;border-radius:2px;letter-spacing:1px;}
.feed-act.block{background:rgba(255,23,68,0.18); color:var(--red);   border:1px solid var(--red);}
.feed-act.allow{background:rgba(0,255,157,0.10); color:var(--green); border:1px solid var(--green);}
.feed-act.fn   {background:rgba(255,109,0,0.16); color:var(--orange);border:1px solid var(--orange);}

/* ── SYSTEM ── */
.sys-row{display:flex;flex-direction:column;gap:3px;margin-bottom:8px;}
.sys-lbl{font-family:var(--font-mono);font-size:9px;color:var(--muted);display:flex;justify-content:space-between;}
.sys-bar{height:5px;background:#1a0830;border-radius:2px;overflow:hidden;}
.sys-fill{height:100%;border-radius:2px;transition:width 0.6s ease;}
.sf-cpu{background:linear-gradient(90deg,var(--cyan),#0080ff);}
.sf-mem{background:linear-gradient(90deg,var(--green),#00cc66);}
.sf-thr{background:linear-gradient(90deg,var(--fire),var(--fire2));}

/* ── DQN INFO ── */
.dqn-row{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:9px;margin-bottom:4px;}
.dqn-row span:first-child{color:var(--muted);}

/* ── REWARD CHART ── */
.reward-wrap{height:80px;margin-top:8px;}

/* ═══════════════════════════════════════════════
   EXPAND OVERLAY — full screen panel popup
   ═══════════════════════════════════════════════ */
#expand-overlay{
  display:none;position:fixed;inset:0;z-index:500;
  background:rgba(4,1,10,0.92);backdrop-filter:blur(6px);
  align-items:center;justify-content:center;padding:24px;
}
#expand-overlay.open{display:flex;}
#expand-box{
  background:var(--panel2);
  border:1px solid var(--border);border-radius:8px;
  width:100%;max-width:1100px;max-height:88vh;
  display:flex;flex-direction:column;
  box-shadow:0 0 60px rgba(255,69,0,0.15),0 0 0 1px rgba(255,69,0,0.06) inset;
  animation:expandIn 0.3s cubic-bezier(.2,.8,.2,1);
  overflow:hidden;
  position:relative;
}
@keyframes expandIn{from{opacity:0;transform:scale(0.93)}to{opacity:1;transform:none}}
#expand-box::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--fire),var(--fire2),transparent);
}
.exp-hd{
  display:flex;align-items:center;gap:10px;
  padding:14px 18px;border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.exp-title{
  font-family:var(--font-hud);font-size:13px;font-weight:700;
  color:var(--fire2);letter-spacing:3px;
}
.exp-close{
  margin-left:auto;width:32px;height:32px;border-radius:4px;
  background:rgba(255,23,68,0.1);border:1px solid var(--red);
  color:var(--red);font-size:16px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all 0.2s;font-family:var(--font-mono);
}
.exp-close:hover{background:rgba(255,23,68,0.25);}
.exp-body{flex:1;overflow:auto;padding:18px;}
.exp-body canvas{width:100%!important;height:320px!important;}
#exp-feed-body{height:480px;overflow-y:auto;font-family:var(--font-mono);font-size:11px;}
#exp-feed-body .feed-row{grid-template-columns:60px 50px 1fr 80px 70px 60px;}

/* ═══════════════════════════════════════════════
   SUMMARY MODAL
   ═══════════════════════════════════════════════ */
#modal-overlay{
  display:none;position:fixed;inset:0;z-index:600;
  background:rgba(4,1,10,0.9);backdrop-filter:blur(8px);
  align-items:center;justify-content:center;
}
#modal-overlay.open{display:flex;}
.modal{
  background:var(--panel2);
  border:1px solid rgba(255,69,0,0.35);
  border-radius:8px;padding:30px;max-width:540px;width:92%;
  box-shadow:0 0 60px rgba(255,69,0,0.2);
  animation:expandIn 0.3s cubic-bezier(.2,.8,.2,1);
  position:relative;overflow:hidden;
}
.modal::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--fire),var(--fire2),transparent);
}
.modal-title{
  font-family:var(--font-hud);font-size:15px;font-weight:700;
  color:var(--fire2);letter-spacing:3px;text-align:center;margin-bottom:5px;
}
.modal-sub{font-family:var(--font-mono);font-size:10px;color:var(--muted);text-align:center;letter-spacing:1px;margin-bottom:20px;}
.modal-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;}
.modal-stat{
  background:#0c0618;border:1px solid var(--border);
  border-radius:6px;padding:12px;text-align:center;
}
.modal-val{font-family:var(--font-hud);font-size:26px;font-weight:700;color:var(--green);}
.modal-lbl{font-family:var(--font-mono);font-size:8px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:3px;}
.cmp-wrap{background:#0c0618;border:1px solid var(--border);border-radius:6px;padding:12px;margin-bottom:16px;}
.cmp-title{font-family:var(--font-mono);font-size:8px;color:var(--fire2);letter-spacing:2px;margin-bottom:8px;}
.cmp-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;}
.cmp-lbl{width:130px;font-family:var(--font-mono);font-size:9px;color:var(--muted);}
.cmp-track{flex:1;height:5px;background:#1a0830;border-radius:2px;overflow:hidden;}
.cmp-fill{height:100%;border-radius:2px;}
.cmp-val{width:42px;text-align:right;font-family:var(--font-mono);font-size:9px;}
.btn-modal{
  display:block;width:100%;padding:9px;
  background:rgba(255,69,0,0.10);border:1px solid var(--fire);
  border-radius:4px;color:var(--fire);
  font-family:var(--font-hud);font-size:11px;font-weight:700;
  letter-spacing:2px;cursor:pointer;transition:all 0.2s;
}
.btn-modal:hover{background:rgba(255,69,0,0.22);}

/* ── scrollbar global ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--panel);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════
     STAGE 1: BOOT TERMINAL
═══════════════════════════════════════════════ -->
<div id="stage-boot">
  <div id="boot-lines"></div>
  <div class="boot-cursor" id="boot-cursor">█</div>
</div>

<!-- ══════════════════════════════════════════════
     STAGE 2: CINEMATIC LOGO
═══════════════════════════════════════════════ -->
<div id="stage-logo">
  <div id="logo-canvas-wrap">
    <div class="logo-ring" id="lr1"></div>
    <div class="logo-ring" id="lr2"></div>
    <div class="logo-ring" id="lr3"></div>
    <!-- SVG Shield — drawn procedurally like Kali dragon -->
    <svg id="logo-svg" viewBox="0 0 180 200" width="180" height="200" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sg1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#ff4500"/>
          <stop offset="50%" stop-color="#ff6a00"/>
          <stop offset="100%" stop-color="#bf00ff"/>
        </linearGradient>
        <filter id="glow-f">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-strong">
          <feGaussianBlur stdDeviation="6" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <!-- Outer shield path -->
      <path id="sh-outer" d="M90 8 L168 38 L168 100 Q168 158 90 192 Q12 158 12 100 L12 38 Z"
        stroke="url(#sg1)" stroke-width="2" fill="none" filter="url(#glow-f)"
        stroke-dasharray="500" stroke-dashoffset="500"/>
      <!-- Inner shield -->
      <path id="sh-inner" d="M90 24 L152 48 L152 100 Q152 146 90 176 Q28 146 28 100 L28 48 Z"
        stroke="rgba(255,106,0,0.4)" stroke-width="1" fill="rgba(255,69,0,0.04)"
        stroke-dasharray="400" stroke-dashoffset="400"/>
      <!-- DQN neural lines -->
      <g id="neural-lines" opacity="0">
        <line x1="90" y1="60"  x2="60"  y2="90"  stroke="rgba(0,229,255,0.5)" stroke-width="0.8"/>
        <line x1="90" y1="60"  x2="90"  y2="95"  stroke="rgba(0,229,255,0.5)" stroke-width="0.8"/>
        <line x1="90" y1="60"  x2="120" y2="90"  stroke="rgba(0,229,255,0.5)" stroke-width="0.8"/>
        <line x1="60"  y1="90"  x2="75"  y2="125" stroke="rgba(191,0,255,0.5)" stroke-width="0.8"/>
        <line x1="90"  y1="95"  x2="75"  y2="125" stroke="rgba(191,0,255,0.5)" stroke-width="0.8"/>
        <line x1="90"  y1="95"  x2="105" y2="125" stroke="rgba(191,0,255,0.5)" stroke-width="0.8"/>
        <line x1="120" y1="90"  x2="105" y2="125" stroke="rgba(191,0,255,0.5)" stroke-width="0.8"/>
        <!-- nodes -->
        <circle cx="90"  cy="60"  r="4" fill="#ff4500" filter="url(#glow-f)"/>
        <circle cx="60"  cy="90"  r="3" fill="#00e5ff" filter="url(#glow-f)"/>
        <circle cx="90"  cy="95"  r="3" fill="#00e5ff" filter="url(#glow-f)"/>
        <circle cx="120" cy="90"  r="3" fill="#00e5ff" filter="url(#glow-f)"/>
        <circle cx="75"  cy="125" r="3" fill="#bf00ff" filter="url(#glow-f)"/>
        <circle cx="105" cy="125" r="3" fill="#bf00ff" filter="url(#glow-f)"/>
        <!-- center glow -->
        <circle cx="90" cy="95" r="18" fill="none" stroke="rgba(255,69,0,0.15)" stroke-width="8"/>
      </g>
      <!-- Shield tip accent -->
      <path id="sh-tip" d="M90 165 L78 148 L90 155 L102 148 Z"
        fill="url(#sg1)" opacity="0" filter="url(#glow-f)"/>
    </svg>
  </div>

  <div id="logo-text-wrap">
    <div class="logo-brand" id="logo-brand">ADAPTIVESHIELD</div>
    <div class="logo-rl-tag" id="logo-rl">⬡ Deep Q-Network · Reinforcement Learning · CICIDS 2017</div>
    <div class="logo-divider" id="logo-div"></div>
    <div class="logo-motto" id="logo-motto">Where others follow <em>rules</em>, we <em>learn</em>.</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════
     STAGE 3: LOADING
═══════════════════════════════════════════════ -->
<div id="stage-load">
  <div class="load-shield">🛡️</div>
  <div class="load-title">ADAPTIVESHIELD</div>
  <div class="load-sub">INITIALISING AI FIREWALL ENGINE</div>
  <div class="load-bar-track"><div class="load-bar-fill" id="load-bar"></div></div>
  <div class="load-msg" id="load-msg">Connecting to server...</div>
  <div class="load-steps">
    <div class="load-step" id="step-0"><div class="load-step-dot"></div>Loading CICIDS 2017 dataset</div>
    <div class="load-step" id="step-1"><div class="load-step-dot"></div>Running preprocessing pipeline</div>
    <div class="load-step" id="step-2"><div class="load-step-dot"></div>Initialising DQN neural network</div>
    <div class="load-step" id="step-3"><div class="load-step-dot"></div>Loading trained model weights</div>
    <div class="load-step" id="step-4"><div class="load-step-dot"></div>Starting packet analysis engine</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════
     EXPAND OVERLAY
═══════════════════════════════════════════════ -->
<div id="expand-overlay">
  <div id="expand-box">
    <div class="exp-hd">
      <span>🛡️</span>
      <div class="exp-title" id="exp-title">PANEL</div>
      <button class="exp-close" onclick="closeExpand()">✕</button>
    </div>
    <div class="exp-body" id="exp-body"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════
     SUMMARY MODAL
═══════════════════════════════════════════════ -->
<div id="modal-overlay">
  <div class="modal">
    <div class="modal-title">⚡ SESSION COMPLETE</div>
    <div class="modal-sub">AdaptiveShield — Final Performance Report</div>
    <div class="modal-grid">
      <div class="modal-stat"><div class="modal-val" id="sum-acc">--</div><div class="modal-lbl">Accuracy</div></div>
      <div class="modal-stat"><div class="modal-val" id="sum-f1" style="color:var(--purple)">--</div><div class="modal-lbl">F1-Score</div></div>
      <div class="modal-stat"><div class="modal-val" id="sum-dr" style="color:var(--cyan)">--</div><div class="modal-lbl">Detection Rate</div></div>
      <div class="modal-stat"><div class="modal-val" id="sum-fpr" style="color:var(--yellow)">--</div><div class="modal-lbl">False Pos. Rate</div></div>
      <div class="modal-stat"><div class="modal-val" id="sum-tp" style="color:var(--green)">--</div><div class="modal-lbl">Threats Blocked (TP)</div></div>
      <div class="modal-stat"><div class="modal-val" id="sum-fn" style="color:var(--orange)">--</div><div class="modal-lbl">Missed Attacks (FN)</div></div>
    </div>
    <div class="cmp-wrap">
      <div class="cmp-title">BASELINE COMPARISON</div>
      <div class="cmp-row"><div class="cmp-lbl">Rule-Based Firewall</div><div class="cmp-track"><div class="cmp-fill" style="width:60%;background:var(--red)"></div></div><div class="cmp-val" style="color:var(--red)">~60%</div></div>
      <div class="cmp-row"><div class="cmp-lbl">Random Forest (ML)</div><div class="cmp-track"><div class="cmp-fill" style="width:95%;background:var(--yellow)"></div></div><div class="cmp-val" style="color:var(--yellow)">~95%</div></div>
      <div class="cmp-row"><div class="cmp-lbl">AdaptiveShield (DQN)</div><div class="cmp-track"><div class="cmp-fill" id="cmp-bar" style="width:99%;background:var(--green)"></div></div><div class="cmp-val" style="color:var(--green)" id="cmp-val">99.09%</div></div>
    </div>
    <div style="text-align:center;font-family:var(--font-mono);font-size:9px;color:var(--muted);margin-bottom:14px;" id="sum-note"></div>
    <button class="btn-modal" onclick="closeModal()">✓ CLOSE REPORT</button>
  </div>
</div>

<!-- ══════════════════════════════════════════════
     MAIN DASHBOARD
═══════════════════════════════════════════════ -->
<div id="main-dash">

  <div class="topbar">
    <div class="brand">
      <div class="brand-icon">🛡️</div>
      <div>
        <div class="brand-name">ADAPTIVESHIELD</div>
        <div class="brand-sub">DQN · Reinforcement Learning · Autonomous Firewall</div>
      </div>
    </div>
    <div class="topbar-right">
      <div class="threat-box" id="threat-box">
        <div class="threat-label">THREAT LEVEL</div>
        <div class="threat-value" id="threat-val">EVALUATING</div>
      </div>
      <div class="clock" id="clock">00:00:00</div>
      <div class="status-pill" id="status-pill">
        <div class="pulse-dot" id="pulse-dot"></div>
        <span id="status-txt">FIREWALL ACTIVE</span>
      </div>
    </div>
  </div>

  <div class="controls">
    <span class="ctrl-label">CONTROLS</span>
    <button class="btn btn-stop"    id="btn-stop"    onclick="doStop()">⛔ Stop</button>
    <button class="btn btn-pause"   id="btn-pause"   onclick="doPause()">⏸ Pause</button>
    <button class="btn btn-resume"  id="btn-resume"  onclick="doResume()" disabled>▶ Resume</button>
    <button class="btn btn-restart" id="btn-restart" onclick="doRestart()" disabled>🔄 Restart</button>
    <button class="btn btn-dl"      id="btn-dl"      onclick="doDownload()">⬇ Logs</button>
    <div class="spacer"></div>
    <div class="prog-wrap">
      <span class="prog-lbl">PROGRESS</span>
      <div class="prog-track"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
      <span class="prog-lbl" id="prog-lbl">0 / —</span>
      <span class="log-badge" id="log-badge">0 logged</span>
    </div>
  </div>

  <!-- 8 METRIC CARDS -->
  <div class="metrics">
    <div class="metric m-acc"   onclick="expandMetric('acc')"  title="Click to expand"><div class="metric-icon">🎯</div><div class="metric-lbl">Accuracy</div><div class="metric-val" id="m-acc">--%</div><div class="metric-sub">Live</div></div>
    <div class="metric m-total" onclick="expandMetric('total')" title="Click to expand"><div class="metric-icon">📡</div><div class="metric-lbl">Packets</div><div class="metric-val" id="m-total" style="color:var(--cyan)">0</div><div class="metric-sub">Analysed</div></div>
    <div class="metric m-blk"   onclick="expandMetric('blocked')" title="Click to expand"><div class="metric-icon">🚫</div><div class="metric-lbl">Blocked</div><div class="metric-val" id="m-blk">0</div><div class="metric-sub">Threats</div></div>
    <div class="metric m-alw"   onclick="expandMetric('allowed')" title="Click to expand"><div class="metric-icon">✅</div><div class="metric-lbl">Allowed</div><div class="metric-val" id="m-alw">0</div><div class="metric-sub">Benign</div></div>
    <div class="metric m-fp"    onclick="expandMetric('fp')"    title="Click to expand"><div class="metric-icon">⚠️</div><div class="metric-lbl">False Pos.</div><div class="metric-val" id="m-fp">0</div><div class="metric-sub">Benign blocked</div></div>
    <div class="metric m-fn"    onclick="expandMetric('fn')"    title="Click to expand"><div class="metric-icon">🔴</div><div class="metric-lbl">False Neg.</div><div class="metric-val" id="m-fn" style="color:var(--orange)">0</div><div class="metric-sub">Attacks missed</div></div>
    <div class="metric m-f1"    onclick="expandMetric('f1')"    title="Click to expand"><div class="metric-icon">⚡</div><div class="metric-lbl">F1-Score</div><div class="metric-val" id="m-f1">--%</div><div class="metric-sub">Harmonic</div></div>
    <div class="metric m-prec"  onclick="expandMetric('prec')"  title="Click to expand"><div class="metric-icon">🔬</div><div class="metric-lbl">Precision</div><div class="metric-val" id="m-prec" style="color:var(--cyan)">--%</div><div class="metric-sub">Attack acc.</div></div>
  </div>

  <!-- MAIN CHARTS ROW -->
  <div class="main-grid">
    <div class="panel" id="panel-acc">
      <div class="panel-hd" onclick="expandPanel('acc-chart')">ACCURACY TIMELINE<span class="expand-btn">⤢ EXPAND</span></div>
      <div class="chart-wrap"><canvas id="accChart"></canvas></div>
    </div>
    <div class="panel" id="panel-atk">
      <div class="panel-hd" onclick="expandPanel('atk-dist')">ATTACK DISTRIBUTION <span style="margin-left:4px;font-size:8px;color:var(--orange)">🟠=MISSED</span><span class="expand-btn">⤢ EXPAND</span></div>
      <div class="threat-bars" id="threat-bars">
        <div style="color:var(--muted);font-size:10px;text-align:center;padding-top:40px;font-family:var(--font-mono)">Awaiting threat data...</div>
      </div>
    </div>
    <div class="panel" id="panel-pie">
      <div class="panel-hd" onclick="expandPanel('pie-chart')">TRAFFIC BREAKDOWN<span class="expand-btn">⤢ EXPAND</span></div>
      <div class="pie-wrap"><canvas id="pieChart"></canvas></div>
    </div>
  </div>

  <!-- BOTTOM ROW -->
  <div class="btm-grid">
    <div class="panel">
      <div class="panel-hd" onclick="expandPanel('feed')">LIVE PACKET ANALYSIS FEED<span class="log-badge" id="feed-count" style="margin-left:auto">0 events</span><span class="expand-btn" style="margin-left:8px">⤢ EXPAND</span></div>
      <div style="display:grid;grid-template-columns:46px 38px 1fr 60px 58px 50px;gap:5px;padding:0 5px 5px;border-bottom:1px solid var(--border);margin-bottom:5px">
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted)">#PKT</span>
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted)">PROTO</span>
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted)">CLASSIFICATION</span>
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted);text-align:right">REWARD</span>
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted);text-align:center">DECISION</span>
        <span style="font-family:var(--font-mono);font-size:8px;color:var(--muted);text-align:right">TIME</span>
      </div>
      <div id="feed"></div>
    </div>

    <div class="panel">
      <div class="panel-hd" onclick="expandPanel('sys')">SYSTEM & DQN ENGINE<span class="expand-btn">⤢ EXPAND</span></div>
      <div class="sys-row"><div class="sys-lbl"><span>CPU</span><span id="cpu-val">0%</span></div><div class="sys-bar"><div class="sys-fill sf-cpu" id="cpu-bar" style="width:0%"></div></div></div>
      <div class="sys-row"><div class="sys-lbl"><span>MEMORY</span><span id="mem-val">0%</span></div><div class="sys-bar"><div class="sys-fill sf-mem" id="mem-bar" style="width:0%"></div></div></div>
      <div class="sys-row"><div class="sys-lbl"><span>THREAT RATE</span><span id="thr-val">0%</span></div><div class="sys-bar"><div class="sys-fill sf-thr" id="thr-bar" style="width:0%"></div></div></div>
      <div style="border-top:1px solid var(--border);padding-top:9px;margin-top:9px">
        <div style="font-family:var(--font-mono);font-size:8px;color:var(--fire2);letter-spacing:2px;margin-bottom:7px">DQN ENGINE</div>
        <div class="dqn-row"><span>Algorithm</span><span>Deep Q-Network</span></div>
        <div class="dqn-row"><span>Architecture</span><span>78→256→128→64→2</span></div>
        <div class="dqn-row"><span>Parameters</span><span>61,506</span></div>
        <div class="dqn-row"><span>Episodes Trained</span><span style="color:var(--green)">500</span></div>
        <div class="dqn-row"><span>Dataset</span><span>CICIDS 2017</span></div>
        <div class="dqn-row"><span>Precision</span><span id="dqn-prec" style="color:var(--cyan)">--%</span></div>
        <div class="dqn-row"><span>Recall</span><span id="dqn-recall" style="color:var(--purple)">--%</span></div>
      </div>
      <div style="border-top:1px solid var(--border);padding-top:9px;margin-top:9px">
        <div style="font-family:var(--font-mono);font-size:8px;color:var(--fire2);letter-spacing:2px;margin-bottom:6px">REWARD HISTORY</div>
        <div class="reward-wrap"><canvas id="rewardChart"></canvas></div>
      </div>
    </div>
  </div>
</div><!-- /main-dash -->

<script>
const socket = io();
let TOTAL_PACKETS = 0, feedCount = 0, loadInterval;
const PROTOS = ['TCP','UDP','ICMP','HTTP','HTTPS','DNS','FTP','SSH'];
const ATK_CLR = {
  'BENIGN':'#00ff9d','DDoS':'#ff1744','DoS Hulk':'#ff6d00',
  'PortScan':'#ffd600','Bot':'#bf00ff','FTP-Patator':'#e53935',
  'SSH-Patator':'#b71c1c','DoS GoldenEye':'#ff8f00',
  'Heartbleed':'#ff1744','Infiltration':'#aa00ff',
  'Web Attack':'#ff6d00','DoS Slowloris':'#dd4400',
  'DoS Slowhttptest':'#cc3300'
};
function aClr(l){for(const k in ATK_CLR)if(l.includes(k))return ATK_CLR[k];return '#4a4a8a';}
setInterval(()=>{document.getElementById('clock').textContent=new Date().toTimeString().slice(0,8);},1000);

/* ═══════════════════════════════════════════════
   STAGE 1: BOOT TERMINAL
═══════════════════════════════════════════════ */
const BOOT_LINES = [
  {t:100, txt:'[  0.000000] AdaptiveShield Kernel v3.0 — DQN Firewall Engine',  cls:''},
  {t:250, txt:'[  0.142857] Initialising threat detection subsystem...',          cls:''},
  {t:420, txt:'[  0.285714] Loading CICIDS 2017 intrusion dataset...',            cls:'ok'},
  {t:580, txt:'[  0.428571] Deep Q-Network agent: ONLINE',                        cls:'ok'},
  {t:750, txt:'[  0.571428] WARNING: Legacy rule-based firewalls detected',       cls:'warn'},
  {t:900, txt:'[  0.571429] STATUS: Outperformed by 39.09 percentage points.',   cls:'warn'},
  {t:1080,txt:'[  0.714285] Loading adaptive neural firewall policies...',        cls:''},
  {t:1240,txt:'[  0.857142] Asymmetric reward function calibrated.',              cls:'ok'},
  {t:1400,txt:'[  0.999999] 99.09% detection rate — ready for deployment.',      cls:'fire'},
  {t:1560,txt:'[  1.000000] AdaptiveShield ACTIVE — all threats will be learned.', cls:'fire'},
];

(function runBoot(){
  const container = document.getElementById('boot-lines');
  BOOT_LINES.forEach(({t,txt,cls})=>{
    setTimeout(()=>{
      const d = document.createElement('div');
      d.className='boot-line'+(cls?' '+cls:'');
      d.textContent=txt;
      container.appendChild(d);
      requestAnimationFrame(()=>requestAnimationFrame(()=>d.classList.add('show')));
    }, t);
  });
  const lastT = BOOT_LINES[BOOT_LINES.length-1].t;
  setTimeout(()=>{
    document.getElementById('boot-cursor').style.display='none';
    const boot = document.getElementById('stage-boot');
    boot.style.transition='opacity 0.7s';
    boot.style.opacity='0';
    setTimeout(()=>{
      boot.style.display='none';
      runLogo();
    }, 700);
  }, lastT + 700);
})();

/* ═══════════════════════════════════════════════
   STAGE 2: CINEMATIC SVG LOGO
═══════════════════════════════════════════════ */
function runLogo(){
  const stage = document.getElementById('stage-logo');
  stage.style.display='flex';

  const shOuter  = document.getElementById('sh-outer');
  const shInner  = document.getElementById('sh-inner');
  const neural   = document.getElementById('neural-lines');
  const shTip    = document.getElementById('sh-tip');
  const logoSvg  = document.getElementById('logo-svg');
  const brand    = document.getElementById('logo-brand');
  const rlTag    = document.getElementById('logo-rl');
  const div      = document.getElementById('logo-div');
  const motto    = document.getElementById('logo-motto');
  const rings    = [document.getElementById('lr1'),document.getElementById('lr2'),document.getElementById('lr3')];

  // T=0 — SVG appears
  logoSvg.style.transition='opacity 0.4s';
  logoSvg.style.opacity='1';

  // T=200 — outer shield draws
  setTimeout(()=>{
    shOuter.style.transition='stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)';
    shOuter.style.strokeDashoffset='0';
  }, 200);

  // T=900 — inner shield
  setTimeout(()=>{
    shInner.style.transition='stroke-dashoffset 0.9s ease';
    shInner.style.strokeDashoffset='0';
  }, 900);

  // T=1200 — neural network lights up
  setTimeout(()=>{
    neural.style.transition='opacity 0.6s';
    neural.style.opacity='1';
  }, 1200);

  // T=1500 — tip
  setTimeout(()=>{
    shTip.style.transition='opacity 0.4s';
    shTip.style.opacity='1';
  }, 1500);

  // T=1600 — rings start pulsing
  setTimeout(()=>{
    rings.forEach((r,i)=>{
      r.style.animation=`ringPulse 2.4s ease-out ${i*0.7}s infinite`;
    });
  }, 1600);

  // T=1800 — brand name
  setTimeout(()=>{
    brand.style.transition='opacity 0.9s, filter 0.9s, transform 0.9s';
    brand.style.opacity='1';
    brand.style.filter='blur(0)';
    brand.style.transform='none';
  }, 1800);

  // T=2400 — RL tag
  setTimeout(()=>{
    rlTag.style.transition='opacity 0.8s';
    rlTag.style.opacity='1';
  }, 2400);

  // T=2800 — divider
  setTimeout(()=>{
    div.style.transition='width 0.7s ease, opacity 0.7s';
    div.style.width='240px';
    div.style.opacity='1';
  }, 2800);

  // T=3200 — motto
  setTimeout(()=>{
    motto.style.transition='opacity 1s, filter 1s';
    motto.style.opacity='1';
    motto.style.filter='blur(0)';
  }, 3200);

  // T=4800 — fade out to loading
  setTimeout(()=>{
    stage.style.transition='opacity 0.8s';
    stage.style.opacity='0';
    setTimeout(()=>{
      stage.style.display='none';
      runLoading();
    }, 800);
  }, 4800);
}

/* CSS keyframe for rings injected */
const style=document.createElement('style');
style.textContent=`
@keyframes ringPulse{
  0%{transform:scale(0.5);opacity:0;}
  15%{opacity:0.6;}
  100%{transform:scale(2.8);opacity:0;}
}
#lr1,#lr2,#lr3{
  position:absolute;inset:0;border-radius:50%;
  border:1px solid rgba(255,69,0,0.4);opacity:0;
}
`;
document.head.appendChild(style);

/* ═══════════════════════════════════════════════
   STAGE 3: LOADING
═══════════════════════════════════════════════ */
const LOAD_MSGS=['Loading CICIDS 2017 dataset...','Running preprocessing pipeline...','Initialising DQN neural network...','Loading trained model weights...','Starting packet analysis engine...'];
function runLoading(){
  document.getElementById('stage-load').style.display='flex';
  let lp=0;
  loadInterval=setInterval(()=>{
    lp=Math.min(lp+Math.random()*7,90);
    document.getElementById('load-bar').style.width=lp+'%';
    const si=Math.min(Math.floor(lp/20),4);
    document.getElementById('load-msg').textContent=LOAD_MSGS[si];
    for(let i=0;i<5;i++){
      const el=document.getElementById('step-'+i);
      el.className='load-step'+(i<si?' done':i===si?' active':'');
    }
  },400);
}

/* ═══════════════════════════════════════════════
   CHARTS
═══════════════════════════════════════════════ */
const chartCfgBase={animation:false,maintainAspectRatio:false};

const accChart=new Chart(document.getElementById('accChart').getContext('2d'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'DQN Accuracy',data:[],borderColor:'#00ff9d',backgroundColor:'rgba(0,255,157,0.05)',fill:true,tension:0.4,pointRadius:0,borderWidth:2},
    {label:'Rule-Based',  data:[],borderColor:'#ff1744',borderDash:[4,4],fill:false,tension:0,pointRadius:0,borderWidth:1},
    {label:'Rand. Forest',data:[],borderColor:'#ffd600',borderDash:[2,4],fill:false,tension:0,pointRadius:0,borderWidth:1}
  ]},
  options:{...chartCfgBase,plugins:{legend:{labels:{color:'#6a4a8a',font:{size:9},boxWidth:10}}},
    scales:{
      y:{min:40,max:100,ticks:{color:'#6a4a8a',callback:v=>v+'%',font:{size:8}},grid:{color:'#1a0830'}},
      x:{ticks:{color:'#6a4a8a',maxTicksLimit:6,font:{size:8}},grid:{color:'#1a0830'}}
    }}
});

const pieChart=new Chart(document.getElementById('pieChart').getContext('2d'),{
  type:'doughnut',
  data:{labels:['TP (Blocked)','TN (Allowed)','FP','FN (Missed)'],
    datasets:[{data:[0,0,0,0],backgroundColor:['#ff1744','#00ff9d','#ffd600','#ff6d00'],borderWidth:0,hoverOffset:4}]},
  options:{...chartCfgBase,cutout:'62%',plugins:{legend:{labels:{color:'#6a4a8a',font:{size:9},boxWidth:8}}}}
});

const rewardChart=new Chart(document.getElementById('rewardChart').getContext('2d'),{
  type:'bar',
  data:{labels:[],datasets:[{label:'Reward',data:[],
    backgroundColor:ctx=>ctx.raw>=0?'rgba(0,255,157,0.45)':'rgba(255,23,68,0.45)',
    borderColor:ctx=>ctx.raw>=0?'#00ff9d':'#ff1744',borderWidth:1}]},
  options:{...chartCfgBase,plugins:{legend:{display:false}},
    scales:{y:{ticks:{color:'#6a4a8a',font:{size:8}},grid:{color:'#1a0830'}},x:{display:false}}}
});

const attackBlocked={}, attackMissed={};

/* ═══════════════════════════════════════════════
   CONTROLS
═══════════════════════════════════════════════ */
function setStatus(txt,color){
  ['status-pill','pulse-dot'].forEach(id=>document.getElementById(id).style.borderColor=color||'');
  document.getElementById('status-pill').style.color=color;
  document.getElementById('pulse-dot').style.background=color;
  document.getElementById('status-txt').textContent=txt;
}
function doStop(){
  if(!confirm('Stop the firewall?'))return;
  socket.emit('control','stop');
  ['btn-stop','btn-pause'].forEach(id=>document.getElementById(id).disabled=true);
  document.getElementById('btn-restart').disabled=false;
  setStatus('STOPPED','var(--red)');
}
function doPause(){
  socket.emit('control','pause');
  document.getElementById('btn-pause').disabled=true;
  document.getElementById('btn-resume').disabled=false;
  setStatus('PAUSED','var(--yellow)');
}
function doResume(){
  socket.emit('control','resume');
  document.getElementById('btn-pause').disabled=false;
  document.getElementById('btn-resume').disabled=true;
  setStatus('FIREWALL ACTIVE','var(--green)');
}
function doRestart(){
  if(!confirm('Restart analysis?'))return;
  socket.emit('control','restart');
  feedCount=0;
  Object.keys(attackBlocked).forEach(k=>delete attackBlocked[k]);
  Object.keys(attackMissed).forEach(k=>delete attackMissed[k]);
  document.getElementById('feed').innerHTML='';
  document.getElementById('feed-count').textContent='0 events';
  document.getElementById('log-badge').textContent='0 logged';
  ['m-acc','m-f1','m-prec'].forEach(id=>document.getElementById(id).textContent='--%');
  ['m-total','m-blk','m-alw','m-fp','m-fn'].forEach(id=>document.getElementById(id).textContent='0');
  document.getElementById('dqn-prec').textContent=document.getElementById('dqn-recall').textContent='--%';
  document.getElementById('prog-fill').style.width='0%';
  document.getElementById('prog-lbl').textContent='0 / '+TOTAL_PACKETS.toLocaleString();
  accChart.data.labels=[];accChart.data.datasets.forEach(d=>d.data=[]);accChart.update();
  pieChart.data.datasets[0].data=[0,0,0,0];pieChart.update();
  rewardChart.data.labels=[];rewardChart.data.datasets[0].data=[];rewardChart.update();
  document.getElementById('threat-bars').innerHTML='<div style="color:var(--muted);font-size:10px;text-align:center;padding-top:40px;font-family:var(--font-mono)">Awaiting threat data...</div>';
  document.getElementById('btn-stop').disabled=false;
  document.getElementById('btn-pause').disabled=false;
  document.getElementById('btn-restart').disabled=true;
  setStatus('FIREWALL ACTIVE','var(--green)');
}
function doDownload(){window.location.href='/download-logs';}
function closeModal(){document.getElementById('modal-overlay').classList.remove('open');}

/* ═══════════════════════════════════════════════
   EXPAND PANEL
═══════════════════════════════════════════════ */
let expandChart=null;
function expandPanel(type){
  const ov=document.getElementById('expand-overlay');
  const body=document.getElementById('exp-body');
  const title=document.getElementById('exp-title');
  body.innerHTML='';
  if(expandChart){expandChart.destroy();expandChart=null;}

  if(type==='acc-chart'){
    title.textContent='ACCURACY TIMELINE — FULL VIEW';
    const cv=document.createElement('canvas');body.appendChild(cv);
    expandChart=new Chart(cv.getContext('2d'),{
      type:'line',
      data:JSON.parse(JSON.stringify(accChart.data)),
      options:{...chartCfgBase,plugins:{legend:{labels:{color:'#6a4a8a',font:{size:11},boxWidth:12}}},
        scales:{
          y:{min:40,max:100,ticks:{color:'#6a4a8a',callback:v=>v+'%'},grid:{color:'#1a0830'}},
          x:{ticks:{color:'#6a4a8a',maxTicksLimit:10},grid:{color:'#1a0830'}}
        }}
    });
  } else if(type==='pie-chart'){
    title.textContent='TRAFFIC BREAKDOWN — FULL VIEW';
    const cv=document.createElement('canvas');cv.style.maxWidth='420px';cv.style.margin='0 auto';body.appendChild(cv);
    expandChart=new Chart(cv.getContext('2d'),{
      type:'doughnut',
      data:JSON.parse(JSON.stringify(pieChart.data)),
      options:{...chartCfgBase,cutout:'55%',plugins:{legend:{labels:{color:'#c8c0e0',font:{size:13},boxWidth:14}}}}
    });
  } else if(type==='atk-dist'){
    title.textContent='ATTACK DISTRIBUTION — FULL VIEW';
    body.style.overflowY='auto';
    body.innerHTML=document.getElementById('threat-bars').innerHTML.replace(/style="width:(\d+)%/g,(m,p)=>`style="width:${Math.max(p,2)}%;height:18px`);
  } else if(type==='feed'){
    title.textContent='LIVE PACKET FEED — FULL VIEW';
    const hdr=document.createElement('div');
    hdr.style.cssText='display:grid;grid-template-columns:60px 50px 1fr 80px 70px 60px;gap:5px;padding:0 5px 8px;border-bottom:1px solid #2a0a4a;margin-bottom:8px;font-family:var(--font-mono);font-size:9px;color:#6a4a8a';
    hdr.innerHTML='<span>#PKT</span><span>PROTO</span><span>CLASSIFICATION</span><span style="text-align:right">REWARD</span><span style="text-align:center">DECISION</span><span style="text-align:right">TIME</span>';
    body.appendChild(hdr);
    const fd=document.createElement('div');fd.id='exp-feed-body';
    fd.innerHTML=document.getElementById('feed').innerHTML;
    body.appendChild(fd);
  } else if(type==='sys'){
    title.textContent='SYSTEM MONITOR & DQN ENGINE';
    body.innerHTML=document.querySelector('.btm-grid .panel:last-child').innerHTML.replace(/<div class="panel-hd"[^>]*>.*?<\/div>/s,'');
  }
  ov.classList.add('open');
}

function expandMetric(type){
  const labels={'acc':'Live Accuracy','total':'Total Packets Analysed','blocked':'Blocked Packets','allowed':'Allowed Packets','fp':'False Positives','fn':'False Negatives (Missed Attacks)','f1':'F1-Score','prec':'Precision'};
  const vals={'acc':document.getElementById('m-acc').textContent,'total':document.getElementById('m-total').textContent,'blocked':document.getElementById('m-blk').textContent,'allowed':document.getElementById('m-alw').textContent,'fp':document.getElementById('m-fp').textContent,'fn':document.getElementById('m-fn').textContent,'f1':document.getElementById('m-f1').textContent,'prec':document.getElementById('m-prec').textContent};
  const descs={'acc':'Rolling accuracy of the DQN agent across all packets processed in this session.','total':'Total number of network flow records from CICIDS 2017 test set processed so far.','blocked':'Packets classified as malicious and blocked. Includes True Positives and False Positives.','allowed':'Packets classified as benign and allowed through. Includes True Negatives and False Negatives.','fp':'Benign packets incorrectly blocked. Target: minimise to reduce service disruption.','fn':'Attack packets incorrectly allowed through — the most critical security failure metric.','f1':'Harmonic mean of Precision and Recall. Best balance metric for imbalanced datasets.','prec':'Of all packets the DQN blocked, what percentage were actually attacks?'};
  document.getElementById('exp-title').textContent=labels[type]||type.toUpperCase();
  const body=document.getElementById('exp-body');
  body.innerHTML=`
    <div style="text-align:center;padding:40px 20px;">
      <div style="font-family:var(--font-hud);font-size:72px;font-weight:900;color:var(--fire2);text-shadow:0 0 30px rgba(255,106,0,0.4);margin-bottom:20px">${vals[type]}</div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--muted);letter-spacing:1px;max-width:480px;margin:0 auto;line-height:1.8">${descs[type]}</div>
    </div>`;
  document.getElementById('expand-overlay').classList.add('open');
}

function closeExpand(){
  document.getElementById('expand-overlay').classList.remove('open');
  if(expandChart){expandChart.destroy();expandChart=null;}
}
document.getElementById('expand-overlay').addEventListener('click',function(e){
  if(e.target===this)closeExpand();
});
document.getElementById('modal-overlay').addEventListener('click',function(e){
  if(e.target===this)closeModal();
});

/* ═══════════════════════════════════════════════
   SOCKET EVENTS
═══════════════════════════════════════════════ */
socket.on('ready',function(d){
  clearInterval(loadInterval);
  TOTAL_PACKETS=d.total_packets||0;
  document.getElementById('load-bar').style.width='100%';
  document.getElementById('load-msg').textContent='System ready — '+TOTAL_PACKETS.toLocaleString()+' test packets loaded.';
  for(let i=0;i<5;i++)document.getElementById('step-'+i).className='load-step done';
  setTimeout(()=>{
    document.getElementById('stage-load').style.transition='opacity 0.5s';
    document.getElementById('stage-load').style.opacity='0';
    setTimeout(()=>{
      document.getElementById('stage-load').style.display='none';
      document.getElementById('main-dash').style.display='block';
      document.getElementById('prog-lbl').textContent='0 / '+TOTAL_PACKETS.toLocaleString();
    },500);
  },800);
});

socket.on('update',function(d){
  document.getElementById('m-acc').textContent  =d.accuracy+'%';
  document.getElementById('m-total').textContent=d.total.toLocaleString();
  document.getElementById('m-blk').textContent  =d.blocked.toLocaleString();
  document.getElementById('m-alw').textContent  =d.allowed.toLocaleString();
  document.getElementById('m-fp').textContent   =d.fp.toLocaleString();
  document.getElementById('m-fn').textContent   =d.fn.toLocaleString();
  document.getElementById('m-f1').textContent   =d.f1+'%';
  document.getElementById('m-prec').textContent =d.precision+'%';
  document.getElementById('dqn-prec').textContent  =d.precision+'%';
  document.getElementById('dqn-recall').textContent=d.recall+'%';
  document.getElementById('log-badge').textContent =d.total.toLocaleString()+' logged';

  const pct=TOTAL_PACKETS>0?Math.round((d.total/TOTAL_PACKETS)*100):0;
  document.getElementById('prog-fill').style.width=pct+'%';
  document.getElementById('prog-lbl').textContent=d.total.toLocaleString()+' / '+TOTAL_PACKETS.toLocaleString();

  // Threat level
  const risk=Math.max(d.total>0?(d.fp/d.total)*100:0, d.total>0?(d.fn/d.total)*100:0);
  const tv=document.getElementById('threat-val'), tb=document.getElementById('threat-box');
  if(risk<3){tv.textContent='LOW';    tv.style.color='var(--green)'; tb.style.borderColor='var(--green)'; tb.style.background='rgba(0,255,157,0.05)';}
  else if(risk<10){tv.textContent='MODERATE';tv.style.color='var(--yellow)';tb.style.borderColor='var(--yellow)';tb.style.background='rgba(255,214,0,0.05)';}
  else{tv.textContent='ELEVATED';tv.style.color='var(--red)';   tb.style.borderColor='var(--red)';   tb.style.background='rgba(255,23,68,0.06)';}

  accChart.data.labels.push(d.total);
  accChart.data.datasets[0].data.push(parseFloat(d.accuracy));
  accChart.data.datasets[1].data.push(60);
  accChart.data.datasets[2].data.push(95);
  if(accChart.data.labels.length>60){accChart.data.labels.shift();accChart.data.datasets.forEach(s=>s.data.shift());}
  accChart.update();

  pieChart.data.datasets[0].data=[d.blocked-d.fp,d.allowed-d.fn,d.fp,d.fn];
  pieChart.update();

  document.getElementById('cpu-val').textContent=d.cpu+'%';
  document.getElementById('mem-val').textContent=d.mem+'%';
  const tr=d.total>0?Math.round((d.blocked/d.total)*100):0;
  document.getElementById('thr-val').textContent=tr+'%';
  document.getElementById('cpu-bar').style.width=d.cpu+'%';
  document.getElementById('mem-bar').style.width=d.mem+'%';
  document.getElementById('thr-bar').style.width=tr+'%';

  // Attack distribution
  const lbl=d.last_label;
  if(lbl!=='BENIGN'){
    if(d.last_action==='BLOCK') attackBlocked[lbl]=(attackBlocked[lbl]||0)+1;
    else attackMissed[lbl]=(attackMissed[lbl]||0)+1;
    const all=new Set([...Object.keys(attackBlocked),...Object.keys(attackMissed)]);
    const sorted=[...all].map(k=>({k,b:attackBlocked[k]||0,m:attackMissed[k]||0})).sort((a,b)=>(b.b+b.m)-(a.b+a.m)).slice(0,8);
    const mx=sorted[0]?sorted[0].b+sorted[0].m:1;
    document.getElementById('threat-bars').innerHTML=sorted.map(({k,b,m})=>`
      <div class="tbar-row">
        <div class="tbar-lbl" title="${k}">${k}</div>
        <div class="tbar-track"><div class="tbar-fill" style="width:${((b+m)/mx)*100}%;background:${aClr(k)}"></div></div>
        <div class="tbar-n" style="color:${aClr(k)}">${b}</div>
        <div class="tbar-fn">${m>0?'▲'+m:''}</div>
      </div>`).join('');
  }

  rewardChart.data.labels.push(d.total);
  rewardChart.data.datasets[0].data.push(d.last_reward);
  if(rewardChart.data.labels.length>40){rewardChart.data.labels.shift();rewardChart.data.datasets[0].data.shift();}
  rewardChart.update();

  // Feed
  feedCount++;
  document.getElementById('feed-count').textContent=feedCount.toLocaleString()+' events';
  const feed=document.getElementById('feed');
  const isFN=(d.last_action==='ALLOW'&&d.last_label!=='BENIGN');
  const cls=isFN?'fn':(d.last_action==='BLOCK'?'block':'allow');
  const proto=PROTOS[d.total%PROTOS.length];
  const now=new Date().toTimeString().slice(0,8);
  const rs=d.last_reward>0?'+':'';
  const row=document.createElement('div');
  row.className='feed-row r-'+cls;
  row.innerHTML=`
    <span style="color:var(--muted)">#${d.total}</span>
    <span style="color:var(--yellow)">${proto}</span>
    <span style="color:${aClr(d.last_label)};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${d.last_label}${isFN?' ⚠️':''}</span>
    <span style="text-align:right;color:${d.last_reward>0?'var(--green)':'var(--red)'}">${rs}${d.last_reward.toFixed(1)}</span>
    <span class="feed-act ${cls}">${isFN?'MISSED':d.last_action}</span>
    <span style="color:var(--muted);text-align:right;font-size:9px">${now}</span>`;
  feed.prepend(row);
  while(feed.children.length>100)feed.removeChild(feed.lastChild);
});

socket.on('done',function(d){
  setStatus('ANALYSIS COMPLETE','var(--cyan)');
  document.getElementById('prog-fill').style.width='100%';
  document.getElementById('btn-stop').disabled=document.getElementById('btn-pause').disabled=true;
  document.getElementById('btn-restart').disabled=false;
  document.getElementById('sum-acc').textContent=d.accuracy+'%';
  document.getElementById('sum-f1').textContent=d.f1+'%';
  document.getElementById('sum-dr').textContent=d.recall+'%';
  document.getElementById('sum-fpr').textContent=d.fpr+'%';
  document.getElementById('sum-tp').textContent=d.tp.toLocaleString();
  document.getElementById('sum-fn').textContent=d.fn.toLocaleString();
  document.getElementById('sum-note').textContent=d.total.toLocaleString()+' packets analysed · '+d.total.toLocaleString()+' rows logged';
  document.getElementById('cmp-bar').style.width=Math.min(parseFloat(d.accuracy),100)+'%';
  document.getElementById('cmp-val').textContent=d.accuracy+'%';
  setTimeout(()=>document.getElementById('modal-overlay').classList.add('open'),700);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/download-logs")
def download_logs():
    if not logs:
        return "No logs yet.", 400
    def generate():
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'packet_id','timestamp','label','is_attack',
            'action','decision','reward','accuracy','precision','recall','f1'
        ])
        writer.writeheader()
        yield output.getvalue(); output.truncate(0); output.seek(0)
        for row in logs:
            writer.writerow(row)
            yield output.getvalue(); output.truncate(0); output.seek(0)
    return Response(generate(), headers={
        'Content-Disposition':'attachment; filename=adaptiveshield_logs.csv',
        'Content-Type':'text/csv'
    })

@app.route("/stats")
def stats():
    return jsonify(session_stats)

def run_firewall():
    global is_paused, is_stopped, logs, session_stats, total_test_size
    print("[AdaptiveShield] Loading data and model...")
    X_train, X_test, y_train, y_test, scaler, le = preprocess()
    agent = DQNAgent(state_size=X_test.shape[1], action_size=2)
    agent.load(SAVE_PATH)
    agent.epsilon = 0.0
    classes = list(le.classes_)
    total_test_size = len(X_test)
    socketio.emit('ready', {'total_packets': total_test_size})
    print(f"[AdaptiveShield] Ready — {total_test_size:,} test packets")
    total=blocked=allowed=fp=fn=correct=tp=0
    precision=recall=f1=accuracy=0.0
    logs=[]
    for i in range(total_test_size):
        if is_stopped: break
        while is_paused and not is_stopped: time.sleep(0.2)
        state=X_test[i]; true_label=int(y_test[i]); is_attack=int(true_label!=0)
        action=agent.act(state)
        label_name=classes[true_label] if true_label<len(classes) else "Unknown"
        if   action==1 and is_attack==1: reward=+1.0; tp+=1
        elif action==0 and is_attack==0: reward=+0.5
        elif action==0 and is_attack==1: reward=-1.0; fn+=1
        else:                            reward=-0.5; fp+=1
        total+=1
        blocked+=1 if action==1 else 0
        allowed+=1 if action==0 else 0
        if reward>0: correct+=1
        accuracy =round((correct/total)*100,2)
        precision=round(tp/(tp+fp)*100,2) if (tp+fp)>0 else 0.0
        recall   =round(tp/(tp+fn)*100,2) if (tp+fn)>0 else 0.0
        f1       =round(2*precision*recall/(precision+recall),2) if (precision+recall)>0 else 0.0
        cpu=round(psutil.cpu_percent(),1); mem=round(psutil.virtual_memory().percent,1)
        logs.append({'packet_id':total,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
            'label':label_name,'is_attack':is_attack,'action':action,
            'decision':'BLOCK' if action==1 else 'ALLOW','reward':reward,
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1})
        if total%10==0:
            socketio.emit('update',{
                'total':total,'blocked':blocked,'allowed':allowed,'fp':fp,'fn':fn,
                'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,
                'last_action':'BLOCK' if action==1 else 'ALLOW',
                'last_label':label_name,'last_reward':reward,'cpu':cpu,'mem':mem})
            time.sleep(0.05)
    if not is_stopped:
        tb=sum(1 for y in y_test if y==0)
        fpr=round((fp/tb)*100,2) if tb>0 else 0.0
        session_stats={'total':total,'blocked':blocked,'allowed':allowed,'tp':tp,'fp':fp,'fn':fn,
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'fpr':fpr}
        socketio.emit('done',session_stats)
        print(f"[AdaptiveShield] Done — Acc:{accuracy}% F1:{f1}% FPR:{fpr}% Logs:{len(logs):,}")

@socketio.on('connect')
def on_connect():
    global firewall_started
    with firewall_lock:
        if not firewall_started:
            firewall_started=True
            threading.Thread(target=run_firewall,daemon=True).start()

@socketio.on('control')
def on_control(cmd):
    global is_paused,is_stopped,firewall_started
    if cmd=='pause':   is_paused=True;  print("[AdaptiveShield] Paused")
    elif cmd=='resume':is_paused=False; print("[AdaptiveShield] Resumed")
    elif cmd=='stop':  is_stopped=True; is_paused=False; print("[AdaptiveShield] Stopped")
    elif cmd=='restart':
        is_stopped=False; is_paused=False; firewall_started=True
        print("[AdaptiveShield] Restarting...")
        threading.Thread(target=run_firewall,daemon=True).start()

if __name__=="__main__":
    socketio.run(app,debug=False,port=5000,allow_unsafe_werkzeug=True)