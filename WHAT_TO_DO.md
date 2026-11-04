# What to do

The submission is a causal result. The faithfulness gap is in the preference labels, and a relabeling removes it for more than one optimizer. The DPO–SimPO pair is the pilot. Later stages run only if that pair, then the three-way call, stays sharp.

Freeze the call before any new training. On `allenai/Llama-3.1-Tulu-3-8B-SFT` and the unmodified Tulu-3 preference mixture, with the primary judge:

- **Data.** DPO, SimPO, and KTO land within 5 points of each other, and each is at least 8 points above the SFT gap. The labels are enough. `protocol.py` names this call `signal`.
- **Optimizer.** DPO exceeds both of the others by at least 5 points, and each bootstrap interval excludes zero. The DPO update is the cause.
- **Inconclusive.** Stop. That result is a workshop note, not this submission.

Then retrain every optimizer that opened the gap on `rebuilt_pairs.jsonl`, where the chosen summary reports the document’s claim. The gap has to fall for each of them, against our own baseline on the original mixture. Calling that a fix also requires the original mixture plus those pairs, with safety and capability within 3 points of the same optimizer on the original mixture.

A second judge has to agree on the sign. The hand-check has to agree with the primary judge often enough to quote in the paper.

About $2,000 at $2.50 per H100-hour if the pilot is clean, about $2,400 if SimPO needs the learning-rate rerun. Roughly two weeks on 8 H100s, or about six weeks on one. The bill is the GPU-hours either way. Eight-GPU bursts are what fit a late-January deadline. Stop at the first failed gate.

Details, hyperparameters, and launch commands: `experiment/optimizer_gap/README.md`.

## Already in the repo

- Decision rules for data (`signal`), optimizer, and inconclusive (`protocol.py`).
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
- [ ] Write the KTO trainer call (TRL) before that run. Both sides of every pair stay. Nothing is dropped.
- [ ] Write the safety and capability calls before any checkpoint is called a fix: `olmes` tasks `safety::olmo3` and `tulu_3_dev`.

PPO is not in this submission. It does not vote.

## Run next

Do these in order. A later step is wasted if the earlier call is inconclusive.

- [x] **Gate 1. Score the released DPO model.** `allenai/Llama-3.1-Tulu-3-8B-DPO`, direct prompt, no system prompt. Passed. FaithGap **24.3** (paper **21.6**, difference 2.7). Confirming 95.2%, opposing 71.0%. Safety 46.4, capability 22.6, subjective −2.3. Summary: `experiment/optimizer_gap/runs/gate1/summary.json`. One H100 80GB for 1h 4m, $2.64. The pod has been deleted. This is a methods check. The paper leads with the attribution.
- [ ] **Hand-check about 100 Gate 1 items**, stratified by category group. The sheet has no judge labels. No GPU. Stop if the labels and the judge disagree often. Quote the agreement in the paper.
- [ ] **Score the released SFT checkpoint.** `allenai/Llama-3.1-Tulu-3-8B-SFT`, same prompt, same judge. Pass: FaithGap within 5 points of **9.7**. Stop if it is within 5 points of the Gate 1 gap, 24.3. A few dollars on one H100. While this passes, every optimizer starts from this public checkpoint. Do not retrain SFT from Llama-3.1-8B.
- [ ] **Train DPO and SimPO from that SFT checkpoint.** Unmodified preference mixture, same script, one epoch, preference seed 42. DPO: `dpo_norm`, beta 5, learning rate 5e-7. SimPO: beta 2, gamma/beta 0.5, same learning rate. About $200 and 10 hours on 8 H100s per run, or about 3 days on one H100. Then score both. If SimPO differs from DPO by more than 5 points, rerun SimPO at learning rate 1e-6 before using that cell (about $200 more). Stop if the pair is still inconclusive. Do not add a run to rescue it.
- [ ] **KTO** on the same pairs, from the same SFT checkpoint. About $200. Apply the three-way rule above. Stop if that call is inconclusive.
- [ ] **Second judge** on the saved summaries. `Llama-3.1-70B-Instruct` or `OLMo-2-1124-13B-Instruct`. A few dollars. Stop if it disagrees on the sign.
- [ ] **Rebuilt pairs.** From the same SFT checkpoint, retrain every optimizer that opened the gap on `rebuilt_pairs.jsonl`. About $200 per optimizer. Chosen text reports the document’s claim. The line “the document says X, though this is false” sits in a separated note. Compare with our own baseline on the original mixture. Their 21.6 to 10.7 used a different set of 1,119 pairs. Stop if the gap does not fall.
- [ ] **Seeds 43 and 44** on the two headline cells only. About $800. Stop if the call flips.
- [ ] **OLMo-2-1124-7B**, two cells, its published Tulu-3 recipe. About $400. DPO and SimPO if the data signal was shared. DPO and the furthest offline optimizer if only DPO opened the gap. Stop if it does not repeat.
- [ ] **Safety and capability** only on checkpoints the paper will call a fix. That claim also trains the original mixture plus the rebuilt pairs. A gap reduction counts as a fix when neither suite mean drops more than 3 points against the same optimizer on the original mixture.

## Out of this submission

- PPO.
- Llama-3.1-70B. The 7B second family is the scale check.
- A from-scratch SFT retrain, while the released SFT gap is near 9.7.
- Qwen, until there is a preference mixture built for its chat template.

## Report

Lead with the attribution. Gate 1 is the methods check: the judge recovered 24.3 against the paper’s 21.6.

The main figure is FaithGap for SFT, DPO, SimPO, and KTO on the original mixture, then the same optimizers on the rebuilt pairs. The main table splits safety, capability, and subjective, and puts the suite scores beside any checkpoint called a fix.

For every cell: confirming rate, opposing rate, FaithGap, Wilson intervals, the paired bootstrap interval, and McNemar on the discordant pairs. State the call in one line, using the rule frozen above: data, optimizer, or inconclusive. The appendix holds the 100-item agreement, the intervals, and the SimPO learning-rate rerun.
