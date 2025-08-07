# OpenAI-Style ML Debugging Practice (v1)

**Format**: 5 self-contained, runnable Python scripts. Each script contains:
- A **buggy implementation** to fix (clearly marked).
- A small **test harness** with asserts you should make pass.

**Timebox**: Aim for 60–90 minutes total (10–20 minutes per exercise).

**How to use**
1. Create a fresh virtualenv with PyTorch installed (CPU is fine).
2. Run each exercise directly, e.g.:
   ```
   python 01_rmsnorm.py
   ```
   You'll see failing asserts until you fix the bugs.
3. Edit the buggy code (look for `# BUG

**Exercises**
1. `01_rmsnorm.py` — Numerics & dtype: implement RMSNorm correctly (upcasting, epsilon, broadcasting).
2. `02_attention_causal.py` — Scaled dot-product attention: scaling, masking, and softmax dim.
3. `03_kv_cache.py` — Incremental decoding: maintain K/V cache and match full-seq attention.
4. `04_cross_entropy_shift.py` — Language modeling loss: next-token shift, flattening, padding mask.
5. `05_rope.py` — Rotary position embeddings: correct pairing, angles, and application to q/k.

**Tip**: Keep fixes minimal; don't refactor beyond what's needed to pass tests.

Good luck!
