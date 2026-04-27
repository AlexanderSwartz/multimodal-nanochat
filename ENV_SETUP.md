# Environment steps performed to run CLIP_COCO_loader.ipynb

Date: 2026-04-24

This file records the exact environment actions and commands I ran to get
`CLIP_COCO_loader.ipynb` working in your `nanochat` virtual environment.

Summary
-------
- Virtualenv used: `nanochat/.venv` (Python 3.10.17)
- Registered a Jupyter kernel named `nanochat` (display name "Python (nanochat)")
- Installed missing packages into the `nanochat` venv: `ipykernel`, `pycocotools`,
  `transformers`, `Pillow`, and ensured `numpy<2.0.0`.
- Added a `kernelspec` entry to the notebook so it defaults to the `nanochat` kernel.

Why these steps were needed
--------------------------
- Activating a venv in a shell does not automatically make it a selectable Jupyter
  kernel in VS Code. I installed `ipykernel` and registered the kernel so the
  notebook can be opened with the same Python interpreter used in the terminal.
- The notebook originally raised `ModuleNotFoundError` for packages that were not
  installed in the venv; I installed them into `nanochat/.venv` so the kernel can
  import them.

Exact commands I ran (run from repo root unless noted)
----------------------------------------------------

1) Inspect venv Python and pip:

```bash
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -V
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -m pip --version
```

2) Upgrade packaging tools inside the venv:

```bash
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

3) Install and register the Jupyter kernel for the venv:

```bash
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -m pip install ipykernel
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -m ipykernel install --user --name=nanochat --display-name "Python (nanochat)"
jupyter kernelspec list   # confirm kernel installed
```

4) Install missing runtime packages into the venv (used by the notebook):

```bash
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -m pip install pycocotools transformers "numpy<2.0.0" Pillow --no-cache-dir
```

5) Quick verification commands I ran after installs:

```bash
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -c "import torch, sys; print('python', sys.executable); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -c "import pycocotools; print('pycocotools OK')"
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -c "import transformers; print(transformers.__version__)"
/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python -c "import PIL; print('PIL', getattr(PIL, '__version__', 'unknown'))"
```

Kernelspec and notebook metadata changes
---------------------------------------
- Kernel file installed by `ipykernel` (example path on this system):

  `/home/as7629/.local/share/jupyter/kernels/nanochat/kernel.json`

  Sample `kernel.json` content created by the registration step:

  ```json
  {
    "argv": [
      "/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python",
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
    ],
    "display_name": "Python (nanochat)",
    "language": "python"
  }
  ```

- I also added a `kernelspec` block to the top-level metadata of
  `CLIP_COCO_loader.ipynb` so it will open with `nanochat` by default.
  (If VS Code still prompts, use the kernel picker and choose "Python (nanochat)")

`uv` vs `venv` (why `uv pip` failed)
-----------------------------------
- `venv` is the standard Python virtual environment tool (creates an isolated
  interpreter and site-packages directory). You activate it with
  `source .venv/bin/activate` and then `python`/`pip` refer to that env.
- `uv` is a separate CLI/project tool (not installed by default). The notebook
  instructions referenced `uv pip install ...` as a convenience wrapper. On this
  machine `uv` was not on PATH (`command -v uv` returned nothing), so `uv`
  commands failed; I used the venv's `python -m pip` directly instead.

If you want `uv` available globally (optional):

```bash
# install into your user site so `uv` is on PATH (ensure ~/.local/bin in PATH)
python3 -m pip install --user uv

# or use pipx (clean isolation):
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install uv
```

How to reproduce everything I did (minimal)
-----------------------------------------

```bash
# from repo root
python3 -m venv nanochat/.venv
source nanochat/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ipykernel
python -m ipykernel install --user --name=nanochat --display-name "Python (nanochat)"
python -m pip install pycocotools transformers "numpy<2.0.0" Pillow
jupyter kernelspec list
# Then open CLIP_COCO_loader.ipynb and choose kernel "Python (nanochat)"
```

Notes / troubleshooting
-----------------------
- If the notebook still shows the wrong kernel, restart VS Code or re-open the
  notebook, then use the kernel picker (top-right) to select "Python (nanochat)".
- If you run into permission issues during `ipykernel install --user`, try
  running the install command without `--user` while the venv is activated.
- If a package import still fails inside the notebook, verify `sys.executable`
  inside a notebook cell matches `/home/as7629/multimodal-nanochat/nanochat/.venv/bin/python`.

If you want, I can (A) run the notebook's first cell here to confirm everything
works end-to-end, or (B) leave the environment as-is and help with the tokenizer
work next.
