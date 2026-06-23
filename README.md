# LightLX

```
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
```

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

**Run models too big for your RAM on Windows — any Hugging Face model that
fits on disk but not in memory — by streaming weights layer-by-layer.** Point it
at a model directory and go.

READ THE ORIGINAL REPO
But this is for WINDOWS

Step By Step

How to install and run it on Windows:

Open PowerShell or Command Prompt.

Navigate into the root folder you just created in CMD

cd path\to\LightLX-Windows

Install the package into your Python environment in CMD

pip install -e .

Run the program from anywhere by just typing in CMD

lightlx

(Note: If you have an NVIDIA graphics card, PyTorch will automatically detect it and use CUDA to run the model on your GPU. If you do not, it will fall back to your CPU, which will work but will be slower).

Step 2 — Download a model From HuggingFace

Run this from inside your lightlx/ folder (or anywhere):
cmd
hf download ai name --local-dir ./models/ai name     example: hf download mistralai/Devstral-Small-2-24B-Instruct-2512 --local-dir ./models/Devstral-Small-2-24B

Step 3

Type in lightlx into CMD

## License

MIT — see [LICENSE](LICENSE). Built on [MLX](https://github.com/ml-explore/mlx)
and [mlx-lm](https://github.com/ml-explore/mlx-lm). Models you run carry their own
licenses.
