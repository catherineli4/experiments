from typing import List, Literal
import torch
import torch.nn.functional as F

def log_likelihood_rolling(
    hflm_model,
    hflm_tokenizer,
    dataset: List[str],
    per_example_reduction: Literal["none", "mean", "sum"]
):
    """
    Compute next-token log-likelihoods for each string in `dataset` using a causal LM.

    For a tokenized sequence [t0, t1, ..., t_{n-1}], this returns log p(t_k | t_<k>) for k=1..n-1
    (i.e., first *predicted* token is the second token), excluding any padding.

    per_example_reduction:
      - "none": returns List[List[float]] where each inner list contains per-token log-likelihoods
      - "mean": returns List[float] with mean log-likelihood per example (normalized by #valid tokens)
      - "sum" : returns List[float] with sum of log-likelihoods per example

    Notes:
      * If the tokenizer has no pad_token, this sets it to eos_token for batching.
      * Long texts are truncated to the model’s max length via tokenizer defaults.
      * If an example ends up with 0 valid predicted tokens (degenerate), its mean is 0.0.
    """
    assert per_example_reduction in ("none", "mean", "sum")

    model = hflm_model.model
    tokenizer = hflm_model.tokenizer

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    # Ensure we can pad
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token set; cannot batch with padding.")
        tokenizer.pad_token = tokenizer.eos_token

    BATCH_SIZE = 8  # internal micro-batch size to keep memory in check
    results = []

    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            texts = dataset[start:start + BATCH_SIZE]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = enc["input_ids"].to(device)             # [b, L]
            attention_mask = enc["attention_mask"].to(device)   # [b, L]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                              # [b, L, V]

            # Compute log-probs over the vocab
            log_probs = F.log_softmax(logits, dim=-1)            # [b, L, V]

            # Shift for next-token prediction
            log_probs_shifted = log_probs[:, :-1, :]             # [b, L-1, V]
            labels = input_ids[:, 1:]                            # [b, L-1]
            mask = attention_mask[:, 1:].to(torch.bool)          # [b, L-1] valid predicted positions

            # Gather log p(label_t | context)
            tok_ll = log_probs_shifted.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [b, L-1]

            if per_example_reduction == "none":
                # Return per-token lists, excluding padding
                for row_ll, row_m in zip(tok_ll, mask):
                    results.append(row_ll[row_m].detach().cpu().tolist())

            else:
                mask_f = mask.to(tok_ll.dtype)
                sums = (tok_ll * mask_f).sum(dim=1)                        # [b]
                counts = mask_f.sum(dim=1).clamp(min=1.0)                  # avoid div-by-zero

                if per_example_reduction == "sum":
                    batch_out = sums.detach().cpu().tolist()
                else:  # "mean"
                    batch_out = (sums / counts).detach().cpu().tolist()

                results.extend(batch_out)

    if model_was_training:
        model.train()

    return results
