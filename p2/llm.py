"""LLM explicador local (Qwen2.5-1.5B-Instruct, Apache-2.0) para p2.explainer.explain.

Solo redacta: la coherencia la decide p2.verifier (módulo aparte). Decodificación greedy -> reproducible.
"""
SYSTEM = ("Eres un asistente que resume las decisiones de un usuario sobre variaciones de una imagen. "
          "Responde únicamente con el JSON que se te pide, sin texto adicional.")


def chat_ids(tok, prompt, device):
    """Aplica la plantilla de chat; transformers 5 devuelve un dict por defecto, 4.x un tensor: se aceptan ambos."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return {k: v.to(device) for k, v in enc.items()}


def load_llm(cfg, device="cuda"):
    """Devuelve (función str->str, nombre con revisión). Descarga los pesos la primera vez (~3.1 GB)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c = cfg["llm"]
    tok = AutoTokenizer.from_pretrained(c["repo"], revision=c["revision"])
    model = AutoModelForCausalLM.from_pretrained(c["repo"], revision=c["revision"], dtype=torch.float16).to(device).eval()

    def generate(prompt):
        enc = chat_ids(tok, prompt, device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=c["max_new_tokens"], do_sample=False,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate, f"{c['repo']}@{c['revision'][:8]}"
