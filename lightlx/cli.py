# LightLX CLI — fully input-driven. No flags required, ever:
#
#   lightlx          → pick a model (remembers recent ones), then chat
#
# Flags still exist for scripting/power use, but the default experience is a
# guided, menu-driven session that remembers your models and preferences.

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_lm.models.glm_moe_dsa import ModelArgs

from .model import StreamingGLM
from .state import add_recent, load_state, save_state

BANNER = r"""
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
"""

HELP = """commands
  /menu          settings — reasoning, reply length, switch model
  /think         toggle reasoning (deeper answers, much slower)
  /tokens N      set max tokens per reply
  /clear         forget the conversation, start fresh
  /model         load a different model
  /fast          GLM-5.2 only — 4-bit skeleton, reloads (~1.2× faster)
  /help          show this
  /exit          quit
type anything else to send it to the model   ·   Ctrl-C stops a reply"""


def _dim(s):
    return f"\033[2m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def nice_name(path):
    return os.path.basename(path.rstrip("/")) or path


def _shorten(path, width=46):
    p = path.replace(os.path.expanduser("~"), "~")
    return p if len(p) <= width else "…" + p[-(width - 1):]


# ---------------------------------------------------------------- status line

class StatusLine:
    """Animated single-line status that rides as a suffix after streamed text."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.text = ""
        self.frame = 0
        self.running = False
        self.thread = None

    def start(self):
        if not self.enabled:
            return
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
        with self.lock:
            self.text = text

    def emit(self, text):
        with self.lock:
            if self.enabled:
                sys.stdout.write("\033[K")
            sys.stdout.write(text)
            sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
        with self.lock:
            if self.enabled:
                sys.stdout.write("\033[K")
                sys.stdout.flush()


# ---------------------------------------------------------------- model paths

def clean_path(p: str) -> str:
    p = p.strip()
    if p.startswith("@"):
        p = p[1:].strip()
    p = p.strip("'\"").strip()
    p = p.replace("\\ ", " ")  # shell drag-and-drop escapes spaces
    return os.path.expanduser(p)


def is_model_dir(d: str) -> bool:
    if not d or not Path(d, "config.json").exists():
        return False
    p = Path(d)
    return (p / "model.safetensors.index.json").exists() or any(p.glob("*.safetensors"))


def pick_model(state) -> str | None:
    """Interactive picker: choose a remembered model by number, or drag/paste a
    folder. Returns a valid model dir, or None if the user quits."""
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
        try:
            raw = input("\n" + _bold("›") + " ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if not raw:
            continue
        if raw.isdigit() and recent and 1 <= int(raw) <= len(recent):
            return recent[int(raw) - 1]
        d = clean_path(raw)
        if is_model_dir(d):
            return d
        print(_dim(f"  not a model folder: {d}  — pick a number, paste a valid path, or q"))


# ---------------------------------------------------------------- model build

def build_model(model_dir, max_layers, verbose, expert_cache_gb, pin_attn_layers, wired_gb,
                skeleton_bits, prefetch=False, force_stream=False):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    mtype = cfg.get("model_type")
    from .model import _total_ram_gb
    size_gb = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors")) / 1e9
    fits = size_gb < 0.65 * _total_ram_gb()  # leave room for KV cache, activations, OS

    if fits and not force_stream and not max_layers:          # fits in RAM → resident (fast)
        from .generic import ResidentModel
        if verbose:
            print(f"  {size_gb:.1f} GB — fits in memory, loading resident (fast)")
        return ResidentModel(model_dir, verbose=verbose)

    if verbose:                                               # too big → stream from disk
        why = "forced" if force_stream else ("debug" if max_layers else f"{size_gb:.0f} GB > memory")
        print(f"  {why} — streaming from disk (slow but it runs)")
    if mtype == "glm_moe_dsa":
        args = ModelArgs.from_dict(cfg)
        if max_layers:
            args.num_hidden_layers = min(max_layers, args.num_hidden_layers)
        return StreamingGLM(model_dir, args, verbose=verbose, expert_cache_gb=expert_cache_gb,
                            pin_attn_layers=pin_attn_layers, wired_gb=wired_gb, skeleton_bits=skeleton_bits,
                            prefetch=prefetch)
    from .generic import GenericStreamingModel
    return GenericStreamingModel(model_dir, verbose=verbose, max_layers=max_layers)


def load_tokenizer(model_dir):
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def model_arch(model_dir):
    return json.load(open(Path(model_dir) / "config.json")).get("model_type", "?")


# ---------------------------------------------------------------- generation

def _fmt_eta(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s/60:.1f}m"


def _encode(tok, messages, think):
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_dict=True, enable_thinking=think)
    except TypeError:  # tokenizer built without the enable_thinking kwarg
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def generate(model, tok, eos, messages, max_tokens, verbose, think=False, ctx_limit=8192):
    # `messages` is the whole conversation. Drop oldest turns if prompt + reply budget
    # would overflow the context window, so multi-turn memory stays within bounds.
    msgs = list(messages)
    ids = _encode(tok, msgs, think)
    while len(ids) + max_tokens > ctx_limit and len(msgs) > 1:
        msgs = msgs[2:] if len(msgs) > 2 else msgs[-1:]  # drop an oldest user/assistant pair
        ids = _encode(tok, msgs, think)
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
            # window just reset (layer 1 of a token) -> reuse the previous reading
            # instead of flashing 0.00 GB/s · ~0s
            gbps, eta = state.get("last", (0.0, 0.0))
        head = "prefill" if state["phase"] == "prefill" else f"tok {state['tok']}/{max_tokens}"
        sl.update(f"{head} · layer {done}/{total} · {gbps:.2f} GB/s · ~{_fmt_eta(eta)} left")

    streaming = getattr(model, "streaming", True)
    label = "reasoning — long" if think else "direct"
    print(_dim(f"\n{'streaming from disk · ' if streaming else 'resident · '}{label}\n"))
    sl.start()
    sl.update("thinking…")
    t0 = time.time()
    n = 0
    gen_ids = []
    try:
        state["win"] = deque(maxlen=10)
        logits = model(mx.array([ids]), cache, on_layer=on_layer)
        state["phase"] = "decode"
        for _ in range(max_tokens):
            nxt = int(mx.argmax(logits[0, -1]))
            if nxt in eos:
                break
            sl.emit(tok.decode([nxt]))
            gen_ids.append(nxt)
            n += 1
            state["tok"] = n
            state["win"] = deque(maxlen=10)
            logits = model(mx.array([[nxt]]), cache, on_layer=on_layer)
    except KeyboardInterrupt:
        sl.stop()
        print(_dim("\n— stopped —"))
        return tok.decode(gen_ids).strip() if gen_ids else ""  # keep partial reply in history
    sl.stop()
    dt = time.time() - t0
    if verbose:
        extra = f" · {model.w.bytes_read/1e9:.0f} GB read" if streaming and model.w.bytes_read else ""
        rate = n / max(dt, 1e-9)
        speed = f"{rate:.2f} tok/s" if rate >= 0.01 else f"{dt/max(n,1):.0f}s/token"  # s/tok for slow streamed runs
        print(_dim(f"\n\n  {n} tokens · {dt:.0f}s · {speed}{extra}"))
    else:
        print()
    return tok.decode(gen_ids).strip()


# ---------------------------------------------------------------- session

class Session:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        # _pref is the persisted source of truth (only changed by explicit REPL/menu
        # actions). think/max_tokens are the EFFECTIVE values for this run: a --flag
        # is a per-run override that must never be written back to the saved prefs.
        self._pref = {"think": bool(state["prefs"].get("think", False)),
                      "max_tokens": int(state["prefs"].get("max_tokens", 512))}
        self.think = self._pref["think"] or args.think
        self.max_tokens = args.max_tokens or self._pref["max_tokens"]
        self.fast = bool(args.fast)
        self.model = self.tok = self.eos = None
        self.model_dir = self.name = self.arch = None
        self.history = []  # running conversation: [{role, content}, ...]

    def load(self, model_dir, announce=True):
        if announce:
            print(f"\nloading {_bold(nice_name(model_dir))} …")
        self.model = build_model(model_dir, self.args.max_layers, not self.args.quiet,
                                 self.args.expert_cache_gb, self.args.pin_attn_layers, self.args.wired_gb,
                                 skeleton_bits=(4 if self.fast else None), prefetch=self.args.prefetch,
                                 force_stream=self.args.stream)
        self.tok = load_tokenizer(model_dir)
        eos = json.load(open(Path(model_dir) / "config.json")).get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.model_dir = model_dir
        self.name = nice_name(model_dir)
        self.arch = model_arch(model_dir)
        self.history = []  # fresh conversation per loaded model
        add_recent(self.state, model_dir)
        self.persist()

    @property
    def mode(self):
        return "resident" if not getattr(self.model, "streaming", True) else "streaming"

    @property
    def ctx_limit(self):
        # GLM-5.2 streaming engine is exact only below the DSA index_topk=2048 bound
        return 2000 if self.is_glm else 8192

    @property
    def is_glm(self):
        return self.arch == "glm_moe_dsa"

    def persist(self):
        # write only the saved prefs (_pref), never a transient --flag override
        self.state["prefs"] = dict(self._pref)
        save_state(self.state)


# ---------------------------------------------------------------- REPL + menus

def _prompt_line(sess):
    chips = sess.mode + ("·think" if sess.think else "") + ("·fast" if sess.fast and sess.is_glm else "")
    return "\n" + _dim(chips) + "  " + _bold(sess.name) + " " + _bold("›") + " "


def set_tokens(sess, arg=None):
    val = arg
    if val is None:
        try:
            val = input(_dim(f"  reply length in tokens (now {sess.max_tokens}) › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
    if val.isdigit() and int(val) > 0:
        sess.max_tokens = sess._pref["max_tokens"] = int(val)
        sess.persist()
        print(_dim(f"  reply length = {sess.max_tokens} tokens"))
    else:
        print(_dim("  enter a positive number"))


def toggle_fast(sess):
    if not sess.is_glm:
        print(_dim("  /fast is GLM-5.2 only (it 4-bit-quantizes the skeleton)"))
        return
    sess.fast = not sess.fast
    print(_dim(f"  switching to {'fast (4-bit skeleton)' if sess.fast else 'full (BF16)'} — reloading, ~30–60s…"))
    try:
        sess.load(sess.model_dir, announce=False)
        print(_dim(f"  now in {'fast' if sess.fast else 'full'} mode"))
    except Exception as e:
        sess.fast = not sess.fast
        print(_dim(f"  reload failed ({e})"))


def switch_model(sess) -> bool:
    md = pick_model(sess.state)
    if md is None:
        return False
    sess.fast = False  # reset per-model runtime option
    sess.load(md)
    print(_dim(f"  loaded {sess.name} · {sess.mode}"))
    return True


def settings_menu(sess) -> str:
    """Returns 'quit' if the user chose to quit, else '' (back to chat)."""
    while True:
        print("\n  " + _bold("Settings") + _dim("   number to change · Enter for back"))
        print(f"   1  Reasoning      {_bold('on' if sess.think else 'off')}   {_dim('deeper, much slower')}")
        print(f"   2  Reply length   {_bold(str(sess.max_tokens))} {_dim('tokens')}")
        print(f"   3  Switch model   {_dim(sess.name)}")
        if sess.is_glm:
            print(f"   4  Fast mode      {_bold('on' if sess.fast else 'off')}   {_dim('4-bit skeleton, reloads')}")
        print(f"   q  Quit LightLX")
        try:
            c = input("  " + _bold("›") + " ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ""
        if c in ("", "b", "back"):
            return ""
        elif c == "1":
            sess.think = sess._pref["think"] = not sess.think
            sess.persist()
        elif c == "2":
            set_tokens(sess)
        elif c == "3":
            if switch_model(sess):
                return ""
        elif c == "4" and sess.is_glm:
            toggle_fast(sess)
        elif c in ("q", "quit"):
            return "quit"
        else:
            print(_dim("  pick a number"))


def repl(sess):
    print(_dim(f"\n  {sess.name} · {sess.mode} · {sess.max_tokens} tokens max · remembers the chat"))
    print(_dim("  message the model, or /menu for settings · /help · /clear · /exit"))
    while True:
        try:
            line = input(_prompt_line(sess)).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit", "/q"):
            return
        if line in ("/menu", "/", "/settings"):
            if settings_menu(sess) == "quit":
                return
            continue
        if line == "/help":
            print(HELP)
            continue
        if line in ("/clear", "/reset", "/new"):
            sess.history = []
            print(_dim("  conversation cleared — fresh start"))
            continue
        if line == "/think":
            sess.think = sess._pref["think"] = not sess.think
            sess.persist()
            print(_dim(f"  reasoning {'on — deeper, much slower' if sess.think else 'off'}"))
            continue
        if line == "/fast":
            toggle_fast(sess)
            continue
        if line == "/model" or line == "/switch":
            switch_model(sess)
            continue
        if line.startswith("/tokens"):
            parts = line.split()
            set_tokens(sess, parts[1] if len(parts) == 2 else None)
            continue
        if line.startswith("/"):
            print(_dim(f"  unknown command {line} — try /help"))
            continue
        sess.history.append({"role": "user", "content": line})
        reply = generate(sess.model, sess.tok, sess.eos, sess.history, sess.max_tokens,
                         verbose=not sess.args.quiet, think=sess.think, ctx_limit=sess.ctx_limit)
        if reply:
            sess.history.append({"role": "assistant", "content": reply})
        else:
            sess.history.pop()  # nothing generated — drop the dangling user turn


# ---------------------------------------------------------------- entrypoint

def main():
    ap = argparse.ArgumentParser(description="LightLX — run any model, big or small, on Apple Silicon")
    ap.add_argument("--model-dir", default=None, help="skip the picker and load this model")
    ap.add_argument("--prompt", default=None, help="one-shot: answer this and exit (for scripts)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--stream", action="store_true", help="force streaming even if the model fits in RAM")
    ap.add_argument("--max-layers", type=int, default=None, help="debug: run only the first N layers")
    ap.add_argument("--expert-cache-gb", type=float, default=0.0)
    ap.add_argument("--pin-attn-layers", type=int, default=0)
    ap.add_argument("--wired-gb", type=float, default=None)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    print(BANNER)
    state = load_state()
    sess = Session(a, state)

    model_dir = clean_path(a.model_dir) if a.model_dir else None
    if a.model_dir and not is_model_dir(model_dir):
        print(f"✗ not a model folder: {model_dir}")
        model_dir = None
    if model_dir is None:
        model_dir = pick_model(state)
        if model_dir is None:
            print("bye.")
            return
    sess.load(model_dir)

    if a.prompt:  # one-shot for scripting
        print("\n" + _bold("›") + f" {a.prompt}")
        generate(sess.model, sess.tok, sess.eos, [{"role": "user", "content": a.prompt}],
                 sess.max_tokens, verbose=not a.quiet, think=sess.think, ctx_limit=sess.ctx_limit)
        return

    repl(sess)
    sess.persist()
    print(_dim("\nsaved. see you next time."))


if __name__ == "__main__":
    main()
