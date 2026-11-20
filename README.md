# Zero-Shot Survey Completion with a Tabular Foundation Model

Fondazione Bruno Kessler’s entry for SimulacraBench 2026, Codabench submission #1061774.

We used the same frozen NVIDIA Kumo Tabular Large model for all three surveys. For each question, we use everyone with a complete response as a labelled example. Kumo then predicts the missing answers from each held-out respondent’s visible survey fields. We average logits from four preprocessing views before applying softmax. The entry point is [submission/main.py](submission/main.py).

Our Final score was **0.49** overall, with **0.34** on UNICEF, **0.24** on World Bank and **0.88** on UNHCR. [Test leaderboard](https://www.codabench.org/competitions/17822/#/results-tab)

## Rebuild the submission

From the repository root, download the checkpoint next to `main.py` and run the build script. It requires Python 3.10+ and a filesystem that supports hard links.

```sh
curl -fL --retry 3 -o submission/kumo-large.pt \
  https://huggingface.co/nvidia/Kumo-Tabular/resolve/3520742f7f3b937a2751266cea22d7fb6686882c/large/classifier.pt
python3 rebuild.py KUMO_LARGE_FINAL_4_VIEWS_1911_v2.zip
```

The script checks the checkpoint and every source file, then prints the SHA-256 of the reconstructed ZIP. For our submission it was `1b5ecb16eb88a1fe9ba2674e7f2f817d159b769161dcece91437e90cccad8742`. The checkpoint digest is in [config.json](submission/config.json).

## Run the UNICEF example

This example runs the organizers’ grader on synthetic UNICEF responses. Start at the repository root with the checkpoint in place and Python 3.13 installed. The grader installs the model dependencies from `submission/requirements.txt`.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
git clone https://github.com/SituatedEvals/public.git starter
git -C starter checkout --detach 5c69e17b191e82fe867eea6daf06c8ba8feaf622
cd starter
python -m pip install -r requirements.txt
python make_sandbox.py --schema data/unicef.json --out _sandbox
python score.py --phase 2 --data _sandbox --schema data/unicef.json --submission ../submission
```

## License

Our wrapper and build script use the [MIT License](LICENSE). NVIDIA distributes the checkpoint under [OpenMDW-1.1](submission/KUMO_MODEL_LICENSE). The repository also includes the [SDM notices](submission/licenses/sdm/) and the [starter kit’s MIT notice](submission/LICENSE).
