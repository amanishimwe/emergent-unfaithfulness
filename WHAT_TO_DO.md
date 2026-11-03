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

- [ ] Clone FaithConflict and open-instruct: `bash scripts/fetch_deps.sh`. Recorded commits go to `third_party/PINS.txt`.
- [ ] Build the pair files: `python -m experiment.optimizer_gap.prepare` on `third_party/Emergent-unfaithfulness/data/faithconflict.json`.
- [ ] Check every rendered flag against that open-instruct commit’s `--help`. The current names are `--loss_type` and `--beta`.
- [ ] Write the eval runner. It loads a checkpoint, summarizes all 1,880 FaithConflict documents with the direct prompt, greedy, one answer each, then calls the judges. The primary judge is `Qwen/Qwen2.5-32B-Instruct`. The judge sees the document, the claim, and the output.
- [ ] Write the KTO trainer call (TRL) and the two-step PPO path: a reward model on the same pairs, then PPO of the SFT policy against that frozen model.
- [ ] Write the safety and capability calls: `olmes` tasks `safety::olmo3` and `tulu_3_dev`.

## Run, in this order

- [ ] **Gate 1. Score the released model.** `allenai/Llama-3.1-Tulu-3-8B-DPO`, direct prompt, no system prompt. Pass: aggregate FaithGap within 5 points of **21.6**. A miss means the prompt or the judge is wrong. Stop.
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
