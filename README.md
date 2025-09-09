# generated_tokens = tokenized.to(model.device)
# attention_mask = torch.ones_like(generated_tokens, device=model.device)


import torch
import torch.nn.functional as F

# tokenized is a torch.Tensor of shape [1, seq_len]
input_ids = tokenized.to(model.device)
attention_mask = torch.ones_like(input_ids, device=model.device)

max_new_tokens = 200  # number of tokens to generate

for _ in range(max_new_tokens):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # logits for the last token

    # Optional: sampling instead of argmax for natural text
    probs = F.softmax(logits / 0.8, dim=-1)  # temperature=0.8
    next_token = torch.multinomial(probs, num_samples=1)

    # Append new token
    input_ids = torch.cat([input_ids, next_token], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=model.device)], dim=1)

# Decode the generated text
output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(output_text)

Easy Explanation

Start with input tokens

tokenized is your starting text turned into numbers (tensor).

attention_mask tells the model which tokens are real words and which are padding (here, all are real).

Loop for N steps (max_new_tokens)
Each loop = generate 1 new token.

Feed current tokens into the model → get predictions (logits).

Look only at the last token’s logits → that’s the model’s guess for the next word.

Choose the next token

F.softmax(logits / 0.8) → turns logits into probabilities (temperature controls creativity).

torch.multinomial → randomly picks the next token based on probabilities.

(If you used argmax, it would always pick the most likely token = more boring, less natural.)

Append the new token

Add it to input_ids so the model can see it next time.

Update attention_mask to include the new token.

Repeat until you reach max_new_tokens.

Decode back to text

Convert token IDs → human-readable text with tokenizer.decode(...).

🤔 Why do this?

Sometimes your model doesn’t support .generate().

Or you want more control (e.g., custom sampling, top-k, temperature).

This manual loop is exactly what happens under the hood of .generate(), but you can tweak it.

👉 Think of it like:

.generate() = autopilot mode.

Manual loop = you’re the pilot, you decide how the next token is chosen.
