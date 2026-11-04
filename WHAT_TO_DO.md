# What to do

The paper leaves one comparison open: is the faithfulness gap the DPO update, or the preference data. Answer it with one supervised-fine-tuning checkpoint, the unmodified Tulu-3 preference mixture, and three offline optimizers. Stop at a failed gate. Do not read a later stage.

Details, hyperparameters, and launch commands: `experiment/optimizer_gap/README.md`.

## Already in the repo

- Decision rules for signal, optimizer, and inconclusive (`protocol.py`).
- Rebuilt faithfulness pairs and the KTO split (`pairs.py`, `prepare.py`).
- FaithGap, Wilson intervals, McNemar, and the paired bootstrap (`metrics.py`).
- Tulu-3 SFT, DPO, and SimPO command lines (`commands.py`).
- The 100-item hand-check sampler (`handcheck.py`).
- Tests: `python -m unittest experiment.optimizer_gap.test_optimizer_gap`.

## Build before the first training job

- [x] Clone FaithConflict and open-instruct: `bash scripts/fetch_deps.sh`. Recorded commits are in `third_party/PINS.txt` (`faithconflict` `59e66a0`, `open-instruct` `394b616`).
- [x] Build the pair files: `rebuilt_pairs.jsonl` and `rebuilt_kto.jsonl` are in `experiment/optimizer_gap/data/`.
- [ ] Check every rendered flag against that open-instruct commit’s `--help`. The current names are `--loss_type` and `--beta`.
- [x] Write the eval runner. It loads a checkpoint, summarizes all 1,880 FaithConflict documents with the direct prompt, greedy, one answer each, then calls the judges. The primary judge is `Qwen/Qwen2.5-32B-Instruct`. The judge sees the document, the claim, and the output. The judge batch is 4. Batch 64 ran out of memory on one H100 80GB.
- [ ] Write the KTO trainer call (TRL) and the two-step PPO path: a reward model on the same pairs, then PPO of the SFT policy against that frozen model.
- [ ] Write the safety and capability calls: `olmes` tasks `safety::olmo3` and `tulu_3_dev`.

## Run, in this order

- [x] **Gate 1. Score the released model.** `allenai/Llama-3.1-Tulu-3-8B-DPO`, direct prompt, no system prompt. Passed. FaithGap **24.3** (paper **21.6**, difference 2.7). Confirming 95.2%, opposing 71.0%. Safety 46.4, capability 22.6, subjective −2.3. Summary: `experiment/optimizer_gap/runs/gate1/summary.json`. One H100 80GB for 1h 4m, $2.64. The pod has been deleted.
- [ ] **Gate 2. Retrain from Llama-3.1-8B.** SFT on `allenai/tulu-3-sft-mixture` (effective batch 128, 2 epochs, learning rate 5e-6, seed 123), then DPO on the unmodified preference mixture (`dpo_norm`, beta 5, learning rate 5e-7, 1 epoch). Pass: DPO gap within 5 points of **21.8**, and at least 8 points above the SFT gap (theirs were 9.7, then 21.8). Stop if it fails.
- [ ] **Swap the optimizer.** Same SFT checkpoint, same unmodified mixture.
  - [ ] DPO, the run from Gate 2.
  - [ ] SimPO, same script, beta 2, gamma/beta 0.5, same learning rate.
  - [ ] KTO, both sides of every pair kept.
  - [ ] PPO, scored beside the other three. It does not vote.
- [ ] **Read the original-mixture column** on the primary judge. A second judge (`Llama-3.1-70B-Instruct` or `OLMo-2-1124-13B-Instruct`) has to agree on the sign.
  - **Signal:** DPO, SimPO, and KTO within 5 points of each other, each at least 8 points above the SFT gap.
  - **Optimizer:** DPO exceeds both others by at least 5 points, and each bootstrap interval excludes zero.
  - **Inconclusive:** anything else. Do not scale it.
- [ ] If SimPO differs from DPO by more than 5 points, rerun SimPO at learning rate 1e-6 before using that cell.
- [ ] Repeat the two headline cells at seeds 43 and 44. The first pass is one seed (preference seed 42).
- [ ] **Rebuild the data and retrain.** From the same SFT checkpoint, every optimizer, two datasets: the new pairs alone, and the original mixture plus those pairs. Chosen text reports the document’s claim. The line “the document says X, though this is false” sits in a separated note. Compare with our own DPO baseline. Their 21.6 to 10.7 used a different set of 1,119 pairs.
- [ ] **Check safety and capability** on every checkpoint that will be reported. A gap reduction counts as a fix when neither suite mean drops more than 3 points against the same optimizer on the original mixture.
- [ ] **Hand-check about 100 items**, stratified by category group. The sheet has no judge labels.
- [ ] **Scale the contrast that answered the question.** Two cells only, on OLMo-2-1124-7B (its published Tulu-3 recipe) and Llama-3.1-70B. DPO and SimPO if the signal was shared. DPO and the furthest offline optimizer if only DPO opened the gap. Qwen waits until there is a preference mixture built for its chat template.

## Report

For every cell: confirming rate, opposing rate, FaithGap, Wilson intervals, the paired bootstrap interval, and McNemar on the discordant pairs. Split safety, capability, and subjective. State the call in one line: signal, optimizer, or inconclusive.
