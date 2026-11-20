# Zero-Shot Survey Completion with a Tabular Foundation Model

Fondazione Bruno Kessler · SimulacraBench Final · Codabench submission **1061774**

The organizer imports `predict(frame, schema)` from `submission/main.py`. It runs one frozen NVIDIA Kumo Tabular Large classifier with four native views per nonconstant question on all three instruments. The submission directory holds the original wrapper, configuration, pinned Python requirements, vendored SDM snapshot and original third-party notices. The weight is fetched separately. All paths below are relative to the root of this release.

## Rebuild the submitted ZIP

Requires Python 3.10+ and a filesystem supporting hard links. Packaging uses the standard library and does not load the model. Place the upstream classifier at exactly `submission/kumo-large.pt`. From the release root run

```sh
curl --fail --location --retry 3 --output submission/kumo-large.pt https://huggingface.co/nvidia/Kumo-Tabular/resolve/3520742f7f3b937a2751266cea22d7fb6686882c/large/classifier.pt
python3 rebuild.py KUMO_LARGE_FINAL_4_VIEWS_1911_v2.zip
```

The builder checks all 166 original member hashes and sizes, uses the original member order and metadata, verifies the complete archive, and creates the named output only if it does not already exist.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| NVIDIA checkpoint | 854850034 | `cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613` |
| Final ZIP | 855852867 | `1b5ecb16eb88a1fe9ba2674e7f2f817d159b769161dcece91437e90cccad8742` |

## Run the organizer's public synthetic example

The organizer's phase-2 grader supplies the frame and schema to the wrapper. The following example uses the UNICEF public synthetic generator. The hosted Test score requires the organizer's private evaluation environment. For inference, use the competition's Python 3.13 environment with a CUDA-compatible PyTorch 2.8.0 installation. Run from the release root after placing the weight. The example pins the organizer's public scorer with visible-marginal normalization.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
git clone https://github.com/SituatedEvals/public.git starter
git -C starter checkout --detach 5c69e17b191e82fe867eea6daf06c8ba8feaf622
python -m pip install -r starter/requirements.txt -r submission/requirements.txt
submission_dir="$PWD/submission"
cd starter
python make_sandbox.py --schema data/unicef.json --out _sandbox
python score.py --phase 2 --data _sandbox --schema data/unicef.json --config config.yml --submission "$submission_dir"
```

For offline evaluation the checkpoint and SDM are local. No public survey panel is needed at runtime. The original config stores model revision `3520742f7f3b937a2751266cea22d7fb6686882c` and upstream implementation revision `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. A Git commit for the team's uploaded wrapper has not been established. The full ZIP digest identifies the submitted implementation.

## Licences

The team releases its own code in the submission wrapper and its reproduction helpers under the [MIT License](LICENSE). The NVIDIA checkpoint uses OpenMDW-1.1. Its terms are in [KUMO_MODEL_LICENSE](submission/KUMO_MODEL_LICENSE). Vendored SDM and its third-party components carry their original notices in [licenses/sdm](submission/licenses/sdm/). The [SituatedEvals MIT notice](submission/LICENSE) covers the starter material.
