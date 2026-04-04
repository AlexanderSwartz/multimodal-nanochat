# multimodal-nanochat

# Google Cloud VM Configuration
- Boot image: Google, Deep Learning VM with CUDA 12.4, M129, Debian 11, Python 3.10. With CUDA 12.4 preinstalled. (newest)
- GPU: NVIDIA L4
- Machine type: g2-standard-4 (4 vCPUs, 16 GB Memory) (default)
- Disk: 50 GB (should probably be larger)

# Git Configuration in VM
- SSH into VM (I used VSCode's Remote Explorer extension)
- when prompted, select yes to install NVIDIA drivers
- create SSH key using ssh-keygen
  - `ssh-keygen -t ed25519 -C "your-email@example.com"`
- Copy public key
  - `cat ~/.ssh/id_ed25519.pub`
- paste key in GitHub to authenticate 
- clone this repo

# Pytorch Configuration in VM
- I am using VSCode with Jupyter extensions
- We want to use pre-installed packages as much as possible. I read not to use Anaconda with a VM like this.
- python3 -m venv --system-site-packages nanochat_baseline
- source nanochat_baseline/bin/activate
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124