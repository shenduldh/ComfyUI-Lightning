CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VAE_DIR="${CURRENT_DIR}/../../../models/vae_approx"
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taef1_encoder.pth -P "${VAE_DIR}"
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taef1_decoder.pth -P "${VAE_DIR}"
