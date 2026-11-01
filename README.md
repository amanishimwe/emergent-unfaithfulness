# Copy-edit and silent claim changes

## Current method

The current way to study this is the FAITHCONFLICT setup in *Emergent Unfaithfulness* (Zahraei et al., COLM 2026). A model is asked to report what a document says, usually in one sentence. Each document is written twice. One version states a claim the model would normally reject. The other states a claim it would accept. The rest of the text stays the same.

The measure is the FaithGap: how much less faithful the model is on the rejected claim than on the accepted one. A separate judge labels each answer. The paper finds that aligned models often change the rejected claim without saying so, and that this happens more in larger models. The same gap shows up when the model is asked to answer a question, label entailment, or repeat the conclusion in the document's own words. Repeating the conclusion produces the smallest gap for most open models, but the gap does not disappear.

