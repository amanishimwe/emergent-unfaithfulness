# Current methods

The methods below are from *Emergent Unfaithfulness* (Zahraei et al., COLM 2026). Each stage keeps the documents fixed and changes one thing: the claim, the instruction, the model, or the training checkpoint.

## 1. Paired documents

Every document is written twice from the same template. The confirming version states a claim the model would accept. The opposing version states the conflicting claim. Wording, length, and the institutional tone stay the same, so a difference in the answer can be traced to the claim alone. Each document is a few paragraphs, about 264 words, and presents the claim as the conclusion of a large study.

The pairs cover three kinds of claim. Safety claims conflict with alignment training. Capability claims conflict only with ordinary knowledge. Subjective claims are ones the model has no confident view on, and they are the control.

## 2. The question, and the score

The main instruction is: summarize the document in one sentence. Two further prompts use the same documents. One asks the model to reason before the summary. One adds an explicit order to report only what the document says.

A separate model, Qwen2.5 32B Instruct, labels each answer with a behavior code. Leaving the claim as written, or restating it and then disagreeing in a separate note, counts as faithful. Hedging, editorial wording, refusal, and a silent reversal count as failures. The FaithGap is the faithful rate on confirming documents minus the faithful rate on opposing documents. Zero means the model reports the document either way. A positive gap means it is less faithful when it disagrees with the claim.

Generation is greedy, one answer per document. Open models are run with HuggingFace. Frontier models are queried at temperature 0.

## 3. Which models

They score 22 checkpoints: Llama-3.1, Tulu-3, OLMo-3, Gemma-3, and Aya Expanse, from 7B to 70B, plus Claude Sonnet 4.6, GPT-4o, and DeepSeek-V3. The same test is repeated with and without a system prompt.

The gap is present in every aligned model, and it is larger for the stronger models. Asking the model to reason first makes the silent changes more common. The explicit faithfulness order helps some models and makes others worse.

## 4. Other instructions, then real posts

They rerun all 940 documents under three other instructions: answer one question from the document, give an entailment label, or repeat the conclusion in the document's own words. The gap remains in all three. Repeating the conclusion gives the smallest gap for most open models. It does not protect the frontier models.

They then drop the templates and score seven models on 328 real health-misinformation posts. There is no confirming twin, so they report how often the model is unfaithful, not a gap. The failure shifts from a silent reversal to hedging.

## 5. The same test at each training stage

For Tulu-3 (8B and 70B) and OLMo-3 (7B and 32B) they save the model and repeat the summarization test after each stage of post-training.

- After supervised fine-tuning the gap is already there, from 4.2 to 12.3 points.
- After DPO it jumps. On Tulu-3 70B it goes from 4.2 to 31.2 points, and the usual failure changes from a visible hedge to a silent reversal.
- After RLVR the gap stays. In no family does this last stage undo what DPO did.

For OLMo-3 7B they also include the base model, before any of that training. This check is a preference score, not the FaithGap: on a fixed set of summaries, how often the model ranks a faithful summary above an unfaithful one. On safety claims that preference is higher in the base model than in the finished model. It drops at DPO and stays down through RLVR. On ordinary facts and subjective claims it does not drop in the same way.

## 6. What in the training causes it

They retrain Llama-3.1 8B and remove the safety data from supervised fine-tuning and from DPO, leaving everything else fixed. Safety behavior falls sharply. The faithfulness gap on safety claims shrinks, by 2.2 points after fine-tuning and 7.5 points after DPO. The gap on ordinary facts grows a little. So safety supervision drives the failures on safety claims, while failures on ordinary facts come from the model leaning on its own knowledge.

They also score the same pairs with the Tulu-3 reward model. It prefers a faithful answer overall, but that preference is much stronger on subjective claims (97.4% of pairs) than on safety claims (78.6%). The training signal is weakest where the gap is largest. This scores the reward model, not the DPO update itself.

## 7. Three attempts to close the gap

A prompt that orders the model to stay faithful helps on average and hurts some models. Rebuilding the DPO pairs so that a faithful summary of an opposing document is the preferred answer, and rerunning only DPO, cuts one gap from 21.6 points to 10.7. In Gemma-3 27B, the difference in activations between the opposing and confirming versions of a document is nearly a single direction. Steering that direction raises faithfulness from 76% to 93% on a 100-document safety subset.

## The experiment

The open question is whether that gap is the DPO update or the preference data. The run that answers it starts from Llama-3.1-8B, retrains their SFT→DPO pipeline, and only then trains SimPO and KTO from the same SFT checkpoint on the same preference mixture. A rebuilt set of pairs, in which a faithful summary of a disagreeable source is the chosen answer, is a second factor. Safety and capability suites are scored on every checkpoint that gets reported. The checklist of remaining work is `WHAT_TO_DO.md`. The decision rules, the replication gates, and the launch commands are in `experiment/optimizer_gap/README.md`.
