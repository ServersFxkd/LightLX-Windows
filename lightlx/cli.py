# LightLX CLI — PyTorch Windows-compatible version (with Sampling Fix)

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

import torch
from transformers import AutoTokenizer

from .generic import GenericStreamingModel, ResidentModel
from .state import add_recent, load_state, save_state

BANNER = r"""
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX (PyTorch)
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
"""

HELP = """commands
  /menu          settings — reply length, switch model
  /tokens N      set max tokens per reply
  /clear         forget the conversation, start fresh
  /model         load a different model
  /help          show this
  /exit          quit
type anything else to send it to the model   ·   Ctrl-C stops a reply"""

def _dim(s): return f"\033[2m{s}\033[0m"
def _bold(s): return f"\033[1m{s}\033[0m"

def nice_name(path): return os.path.basename(path.rstrip("/\\")) or path
def _shorten(path, width=46):
    p = path.replace(os.path.expanduser("~"), "~")
    return p if len(p) <= width else "…" + p[-(width - 1):]

class StatusLine:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.text = ""
        self.frame = 0
        self.running = False
        self.thread = None

    def start(self):
        if not self.enabled: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            with self.lock:
                self.frame = (self.frame + 1) % len(self.FRAMES)
                self._draw()
            time.sleep(0.1)

    def _draw(self):
        s = f" {self.FRAMES[self.frame]} {self.text}"
        sys.stdout.write("\033[K" + s + f"\033[{len(s)}D")
        sys.stdout.flush()

    def update(self, text):
        with self.lock: self.text = text

    def emit(self, text):
        with self.lock:
            if self.enabled: sys.stdout.write("\033[K")
            sys.stdout.write(text)
            sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=0.3)
        with self.lock:
            if self.enabled:
                sys.stdout.write("\033[K")
                sys.stdout.flush()

def clean_path(p: str) -> str:
    p = p.strip()
    if p.startswith("@"): p = p[1:].strip()
    p = p.strip("'\"").strip()
    p = p.replace("\\ ", " ")
    return os.path.expanduser(p)

def is_model_dir(d: str) -> bool:
    if not d or not Path(d, "config.json").exists(): return False
    p = Path(d)
    return (p / "model.safetensors.index.json").exists() or any(p.glob("*.safetensors"))

def pick_model(state) -> str | None:
    recent = [p for p in state.get("recent_models", []) if is_model_dir(p)]
    if recent:
        print("\nRecent models")
        for i, p in enumerate(recent, 1):
            print(f"  {i}  {nice_name(p):<26} {_dim(_shorten(p))}")
        print(_dim("\nPick a number — or drag in / paste a model folder.  (q to quit)"))
    else:
        print("\nDrag a model folder here, or paste its path to begin.  " + _dim("(q to quit)"))
        print(_dim("a model folder contains config.json and .safetensors weights"))
    while True:
        try: raw = input("\n" + _bold("›") + " ").strip()
        except (EOFError, KeyboardInterrupt): return None
        if raw.lower() in ("q", "quit", "exit"): return None
        if not raw: continue
        if raw.isdigit() and recent and 1 <= int(raw) <= len(recent): return recent[int(raw) - 1]
        d = clean_path(raw)
        if is_model_dir(d): return d
        print(_dim(f"  not a model folder: {d}  — pick a number, paste a valid path, or q"))

def _total_ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1e9
        else:
            return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / 1e9
    except Exception:
        return 24.0

def build_model(model_dir, max_layers, verbose, force_stream=False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    size_gb = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors")) / 1e9
    fits = size_gb < 0.65 * _total_ram_gb()
    
    if fits and not force_stream and not max_layers:
        if verbose: print(f"  {size_gb:.1f} GB — fits in memory, loading resident (fast)")
        return ResidentModel(model_dir, verbose=verbose, device=DEVICE)
    
    if verbose:
        why = "forced" if force_stream else ("debug" if max_layers else f"{size_gb:.0f} GB > memory")
        print(f"  {why} — streaming from disk (slow but it runs) on {DEVICE.upper()}")
    return GenericStreamingModel(model_dir, verbose=verbose, max_layers=max_layers, device=DEVICE)

def load_tokenizer(model_dir):
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def _fmt_eta(s: float) -> str: return f"{s:.0f}s" if s < 90 else f"{s/60:.1f}m"

def _encode(tok, messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list): ids = ids[0]
    return ids

def generate(model, tok, eos, messages, max_tokens, verbose, ctx_limit=8192):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    msgs = list(messages)
    ids = _encode(tok, msgs)
    while len(ids) + max_tokens > ctx_limit and len(msgs) > 1:
        msgs = msgs[2:] if len(msgs) > 2 else msgs[-1:]
        ids = _encode(tok, msgs)
        
    cache = model.make_cache()
    sl = StatusLine(enabled=verbose)
    state = {"phase": "prefill", "tok": 0, "win": deque(maxlen=10)}

    def on_layer(done, total, gb_read, elapsed):
        win = state["win"]
        win.append((elapsed, gb_read))
        e0, b0 = win[0]
        de, db = elapsed - e0, gb_read - b0
        if de > 1e-3:
            gbps = (db / 1e9) / de
            eta = (de / max(len(win) - 1, 1)) * (total - done)
            state["last"] = (gbps, eta)
        else:
            gbps, eta = state.get("last", (0.0, 0.0))
        head = "prefill" if state["phase"] == "prefill" else f"tok {state['tok']}/{max_tokens}"
        sl.update(f"{head} · layer {done}/{total} · {gbps:.2f} GB/s · ~{_fmt_eta(eta)} left")

    streaming = getattr(model, "streaming", True)
    print(_dim(f"\n{'streaming from disk · ' if streaming else 'resident · '}\n"))
    sl.start()
    sl.update("thinking…")
    t0 = time.time()
    n = 0
    gen_ids = []
    
    # --- Sampling parameters to prevent looping ---
    TEMP = 0.7
    TOP_K = 50
    
    def sample_next(logits_tensor):
        # Apply temperature
        l = logits_tensor[0, -1] / TEMP
        # Filter to top_k
        top_k_logits, top_k_indices = torch.topk(l, TOP_K)
        probs = torch.softmax(top_k_logits, dim=-1)
        # Sample from the filtered distribution
        return top_k_indices[torch.multinomial(probs, num_samples=1)].item()

    try:
        state["win"] = deque(maxlen=10)
        input_ids = torch.tensor([ids], device=DEVICE)
        # Prefill: returns (logits, updated_kv_cache)
        logits, cache = model(input_ids, cache, on_layer=on_layer)
        state["phase"] = "decode"
        
        next_token = sample_next(logits)
        for _ in range(max_tokens):
            if next_token in eos: break
            sl.emit(tok.decode([next_token]))
            gen_ids.append(next_token)
            n += 1
            state["tok"] = n
            state["win"] = deque(maxlen=10)
            
            # Decode: pass single new token + accumulated KV cache
            input_ids = torch.tensor([[next_token]], device=DEVICE)
            logits, cache = model(input_ids, cache, on_layer=on_layer)
            next_token = sample_next(logits)
            
    except KeyboardInterrupt:
        sl.stop()
        print(_dim("\n— stopped —"))
        return tok.decode(gen_ids).strip() if gen_ids else ""
    sl.stop()
    dt = time.time() - t0
    if verbose:
        extra = f" · {model.w.bytes_read/1e9:.0f} GB read" if streaming and model.w.bytes_read else ""
        rate = n / max(dt, 1e-9)
        speed = f"{rate:.2f} tok/s" if rate >= 0.01 else f"{dt/max(n,1):.0f}s/token"
        print(_dim(f"\n\n  {n} tokens · {dt:.0f}s · {speed}{extra}"))
    else:
        print()
    return tok.decode(gen_ids).strip()

class Session:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self._pref = {"max_tokens": int(state["prefs"].get("max_tokens", 512))}
        self.max_tokens = args.max_tokens or self._pref["max_tokens"]
        self.model = self.tok = self.eos = None
        self.model_dir = self.name = self.arch = None
        self.history = []

    def load(self, model_dir, announce=True):
        if announce: print(f"\nloading {_bold(nice_name(model_dir))} …")
        self.model = build_model(model_dir, self.args.max_layers, not self.args.quiet, self.args.stream)
        self.tok = load_tokenizer(model_dir)
        eos = json.load(open(Path(model_dir) / "config.json")).get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.model_dir = model_dir
        self.name = nice_name(model_dir)
        self.history = []
        add_recent(self.state, model_dir)
        self.persist()

    @property
    def mode(self): return "resident" if not getattr(self.model, "streaming", True) else "streaming"
    @property
    def ctx_limit(self): return 8192

    def persist(self):
        self.state["prefs"] = dict(self._pref)
        save_state(self.state)

def _prompt_line(sess):
    return "\n" + _dim(sess.mode) + "  " + _bold(sess.name) + " " + _bold("›") + " "

def set_tokens(sess, arg=None):
    val = arg
    if val is None:
        try: val = input(_dim(f"  reply length in tokens (now {sess.max_tokens}) › ")).strip()
        except (EOFError, KeyboardInterrupt): return
    if val.isdigit() and int(val) > 0:
        sess.max_tokens = sess._pref["max_tokens"] = int(val)
        sess.persist()
        print(_dim(f"  reply length = {sess.max_tokens} tokens"))
    else:
        print(_dim("  enter a positive number"))

def switch_model(sess) -> bool:
    md = pick_model(sess.state)
    if md is None: return False
    sess.load(md)
    print(_dim(f"  loaded {sess.name} · {sess.mode}"))
    return True

def settings_menu(sess) -> str:
    while True:
        print("\n  " + _bold("Settings") + _dim("   number to change · Enter for back"))
        print(f"   1  Reply length   {_bold(str(sess.max_tokens))} {_dim('tokens')}")
        print(f"   2  Switch model   {_dim(sess.name)}")
        print(f"   q  Quit LightLX")
        try: c = input("  " + _bold("›") + " ").strip().lower()
        except (EOFError, KeyboardInterrupt): return ""
        if c in ("", "b", "back"): return ""
        elif c == "1": set_tokens(sess)
        elif c == "2":
            if switch_model(sess): return ""
        elif c in ("q", "quit"): return "quit"
        else: print(_dim("  pick a number"))

def repl(sess):
    print(_dim(f"\n  {sess.name} · {sess.mode} · {sess.max_tokens} tokens max · remembers the chat"))
    print(_dim("  message the model, or /menu for settings · /help · /clear · /exit"))
    while True:
        try: line = input(_prompt_line(sess)).strip()
        except (EOFError, KeyboardInterrupt): return
        if not line: continue
        if line in ("/exit", "/quit", "exit", "quit", "/q"): return
        if line in ("/menu", "/", "/settings"):
            if settings_menu(sess) == "quit": return
            continue
        if line == "/help": print(HELP); continue
        if line in ("/clear", "/reset", "/new"):
            sess.history = []
            print(_dim("  conversation cleared — fresh start"))
            continue
        if line == "/model" or line == "/switch":
            switch_model(sess); continue
        if line.startswith("/tokens"):
            parts = line.split()
            set_tokens(sess, parts[1] if len(parts) == 2 else None); continue
        if line.startswith("/"):
            print(_dim(f"  unknown command {line} — try /help")); continue
        sess.history.append({"role": "user", "content": line})
        reply = generate(sess.model, sess.tok, sess.eos, sess.history, sess.max_tokens,
                         verbose=not sess.args.quiet, ctx_limit=sess.ctx_limit)
        if reply: sess.history.append({"role": "assistant", "content": reply})
        else: sess.history.pop()

def main():
    ap = argparse.ArgumentParser(description="LightLX PyTorch — run any model, big or small")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    print(BANNER)
    state = load_state()
    sess = Session(a, state)

    model_dir = clean_path(a.model_dir) if a.model_dir else None
    if a.model_dir and not is_model_dir(model_dir):
        print(f"✗ not a model folder: {model_dir}"); model_dir = None
    if model_dir is None:
        model_dir = pick_model(state)
        if model_dir is None: print("bye."); return
    sess.load(model_dir)

    if a.prompt:
        print("\n" + _bold("›") + f" {a.prompt}")
        generate(sess.model, sess.tok, sess.eos, [{"role": "user", "content": a.prompt}],
                 sess.max_tokens, verbose=not a.quiet, ctx_limit=sess.ctx_limit)
        return

    repl(sess)
    sess.persist()
    print(_dim("\nsaved. see you next time."))

if __name__ == "__main__":
    main()