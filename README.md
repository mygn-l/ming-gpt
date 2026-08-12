# MingGPT

A minimal GPT written from scratch using Python JAX.

## Training

### Google Compute Engine Stage

Create a Google Cloud project, and set up billing. Navigate to Compute Engine, then "VM Instances", to create instance. When I built this project, I used the `g2-standard-4` instance with a single L4 GPU. Use "Capacity Advisor" to check for available GPUs in your region, then select your desired region when creating the VM. In the "OS and Storage" tab, change your operating system to "Deep Learning on Linux" and allocate at least 100 GB of storage. The right-side panel might warn you about insufficient quota, in which case you should click on "Request Quota Adjustment". Mine got approved instantly. After all these steps, click Create and SSH into the VM.

After SSH authentication, install Python, create venv, switch to venv, then install the libraries
```bash
pip install -U "jax[cuda12]"
pip install datasets tokenizers
```
Clone the repo, then pull the texts
```bash
python download_text.py
python download_ass_text.py
```
Then run
```bash
python vocabularizer.py
```
This process might take a few minutes. Then run
```bash
python pretokenize_to_indices.py
python pretokenize_ass_to_indices.py
```
Then run
```bash
python train.py
```
You can detach from tmux with `Ctrl + B`, release both, then `D`. You can now safely `exit` the VM, and it will keep running. You can check back after a few hours, by SSH-ing again, and attaching back to your tmux session. To view GPU usage, use
```bash
watch -n 1 nvidia-smi
```
When training is done, download `./ming-gpt/base`.

For fine-tuning, first change learning rate to `3e-5` in `config.py`, then continue training with
```bash
python fine_tune.py
```

MAKE SURE TO DELETE YOUR INSTANCE AFTER USAGE. THEY WILL KEEP INCURRING FEES.

## Inference
Run
```bash
python infer.py
```
You can import the `get_response` function from `infer.py` into your own code.
