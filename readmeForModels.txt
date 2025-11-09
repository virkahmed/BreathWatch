# BreathWatch 🫁 (research prototype)

On-device nightly **cough** and **wheeze** analysis from smartphone audio.
**Not a medical device. Not a diagnosis.** For research/education only.

---

## TL;DR

- We turn WAVs into **1-second log-Mel spectrogram tiles** (80 mels).
- We train two tiny CNNs:
  1) **Cough multitask head** (COUGHVID + ICBHI negatives)  
     - **Head A:** `cough` vs `no_cough`  
     - **Head B:** multi-label attributes: `wet`, `wheezing(self-report)`, `stridor`, `choking`, `congestion`  
     - Attributes use **weak labels** from COUGHVID JSON with **masking** (missing labels don’t count as negatives).
  2) **Wheeze head** (ICBHI)  
     - `wheeze` vs `other` to estimate **wheeze_time_pct** (share of the night with wheeze-like frames).
- Inference merges cough scores into **events** (start/end/confidence), tags events with attributes, and reports nightly **coughs/hr**, **attribute prevalence**, and **wheeze_time_pct**.

---

## What the models predict

### Per 1-second window
- `p_cough` ∈ [0,1]
- Attribute probs (sigmoid): `p_wet`, `p_wheezing_selfreport`, `p_stridor`, `p_choking`, `p_congestion`
- `p_wheeze` (from the separate ICBHI model)

### Per cough event (post-processing)
- `{ start_ms, end_ms, confidence }`
- Event **tags** (e.g., WET/STRIDOR/… if ≥40% tiles inside the event exceed 0.5)
- Quality gating (optional SNR check)

### Nightly summary
- Coughs/hour, bout count/length
- Attribute prevalence (% of tiles ≥ 0.5 for each attribute)
- **Wheeze_time_pct** = % of tiles ≥ 0.6 on the ICBHI wheeze head

> **Note:** “self-reported wheezing” in COUGHVID ≠ acoustic wheeze from ICBHI; we keep both.

---

## Repo layout

