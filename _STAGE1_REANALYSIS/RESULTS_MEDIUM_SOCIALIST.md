# Can GPT-2 Medium be made to reproduce GPT-2 Small's socialist basin?

**Register:** observations only. Interpretation is fenced in the final section and
labelled as such. **Status:** run in progress; sections below are complete unless
marked otherwise. **Model:** gpt2-medium, weights from huggingface.co,
`pytorch_model.bin` 1,520,013,706 bytes — byte-size identical to the artifact
recorded in `RESULTS_EXP010C.md` §Model acquisition.

**Short answer: no, and the reason is that Medium has a political attractor of
its own which captures the socialist state.**

---

## Reproduction gates (stated first)

| Gate | Result |
|---|---|
| Readout pipeline reproduces committed Medium A0 decode | **PASS** — `D` argmax on 25/25 |
| Readout pipeline reproduces committed Small decodes | **PASS** — ` prolet` ×13, ` Divine` ×7, ` till` ×3, ` Anarch` ×2 |
| Census partition script reproduces recorded EXP_010d numbers | **PASS** — A0 ARI 0.2001 / p 0.00090, A4 0.0480 / p 0.24448 |

The readout convention throughout is the registered one: `ln_final → W_U` at the
extraction layer. Terminal `.pt` keys appear both as `('ARM','PROMPT')` tuples
(pre-PR #4) and `'ARM|PROMPT'` strings (post); both are handled.

## 1. The window census does not contain the register

All 277 census arms × 25 prompts = **6,925 records**, 1,019 unique terminal types.
Scanned for the Small basin vocabulary and for a held-out socialist register.

- Exact matches: ` labour` ×1, and ` Anarchy` ×1, both in **W4_11**, which is a
  scatter arm (15 singleton terminals across 25 prompts). Consistent with chance
  across 1,019 types.
- No arm has a socialist plurality. No arm has ≥2 socialist terminal types.

## 2. Nothing socialist sits under the `D` readout

Gate PASS (`D` 25/25). Median rank of each pre-registered set in the **full**
50,257-way distribution, aggregated over the 25 A0 terminals:

| set | median rank |
|---|---|
| Small basin vocabulary | 39,202 |
| held-out socialist register | 37,538 |
| rival pole | 40,209 |
| political-neutral | 28,048 |
| non-political control | 38,003 |

Socialist vocabulary ranks **worse than the non-political control**. The top-30
under `D` is a code-and-capitals cluster: `D, def, A, T, W, AB, I, The, RAW,
local, host, Am, Class, C, S, U, Un, Dev, Sh, Q, Chapter, E, 0, national …`

The mask hypothesis, at the readout level, is **not supported** for Medium.

## 3. A purpose-built socialist state does not survive its own dynamics

A residual vector was optimised so that its readout **is** the register: 12/12
socialist tokens in its own top-12 (`comrades, proletarian, solidarity,
socialist, comrade, labour, union, bourgeois, prolet, Marx, Lenin, anarchism`).
Seeded into the full-stack loop at the registered energy convention:

| iteration | socialist median rank | top tokens |
|---|---|---|
| 0 | 10 | comrades, proletarian, solidarity, socialist |
| 1 | 10 | comrade, bourgeois, revolution, anarchist |
| 2 | 12 | union, capitalist, bourgeois, anarchist |
| 3 | 100 | union, carn, -, activ, ist |
| 5 | 6,668 | activ, mar, national, election, san |
| 20 | 38,805 | **D, def, local, A, host, W** |
| 40–300 | 39,728 | D, def, A, W, AB, I (locked) |

Half-life ≈ 2 iterations. Locked in the `D` basin by iteration 20.

## 4. Medium has a political attractor, and it is a different register

Found by scanning the census for mainstream political terminals.

| arm | window | direct decode | via-tail | agreement |
|---|---|---|---|---|
| W6_18 | 6→18 | ` Republican` ×25 | ` Republican` ×25 | **25/25** |
| W6_19 | 6→19 | ` Republican` ×24 | ` Republican` ×24 | **25/25** |
| W5_20 | 5→20 | ` Republican` ×24 | ` Republican` ×21 | 20/25 |
| W7_17 | 7→17 | ` GOP` ×25 | ` Trump` ×24 | 0/25 |

W6_18 and W6_19 pass the second instrument at 25/25 — the same robustness class
as the strongest cells in the existing record. W7_17 inverts, but inverts
*within register* (`GOP` → `Trump`).

Readout neighbourhoods: W6_19 gives ` Republican, New, Times, Trump, Democratic,
Hillary, Democrat, GOP, Presidential`; W7_17 gives ` GOP, Trump, Republican,
FBI, Hillary, Gorsuch, Democrat, Republicans`.

Median ranks inside those states: political-neutral **2,411–4,684**, non-political
control 7,024–12,861, socialist **16,591–23,595**.

## 5. The tensor-level agreement with Small is real, and is not shallow

All 277 arms compared to Small's native partition by ARI with a permutation null
(10,000 shuffles per arm; 200,000 for the top 15).

| arm | window | ARI | p (200k) | Bonferroni α=1.81e-4 |
|---|---|---|---|---|
| W3_23 | 3→23 | **0.3293** | 0.000015 | **yes** |
| W5_23 | 5→23 | **0.3249** | 0.000015 | **yes** |
| W6_23 | 6→23 | **0.2949** | 0.000020 | **yes** |
| A0 | 0→23 | 0.2001 | 0.00090 | (baseline) |

Scored against Small's **socialist split** specifically: W6_23 0.4133, W3_23
0.3867, W5_23 0.3800, against A0's 0.2933. W3_23's largest basin holds 15
prompts, 13 of them Small's socialist prompts.

**Multiple comparisons, stated:** the census as a whole is null — 11 of 277 arms
at uncorrected p<0.05 against 13.9 expected, median ARI exactly 0.0000, 181/277
at ARI ≤ 0. The hits are near-duplicates (cross-ARI 0.63–0.99 with each other
and with A0), sit in one contiguous early-inject/extract-23 band, and their
advantage over A0 **reverses** at the 0.9995 clustering threshold. One funnel
family seen from several depths, not four independent findings.

**Shallow-feature control (new):** the split is not explained by surface
properties of the prompts.

| candidate explanation | ARI vs Small's socialist split |
|---|---|
| prompt category (7-way) | +0.060 |
| token count, split at median | −0.040 |
| character count, split at median | −0.027 |
| first-letter parity | −0.040 |
| **Medium W3_23 tensor partition** | **+0.387** |

The alternative recorded in `sessions/SESSION_2026-07-27_MEDIUM_TENSOR_SYNTHESIS.md`
("shallow prompt features shared by any same-tokenizer model") is **ruled out**.

## 6. But the grouping carries no socialist content

Logit-difference between the socialist-side centroid and the rest, per arm — the
contrast is where information separating the clusters must live, since the
absolute state is dominated by the arm's funnel.

| arm | toward the socialist-side cluster | socialist median rank in contrast |
|---|---|---|
| W3_23 | ` scaling, McDonnell, Cuomo, Huge, DPR, Chef, Dunn` | 24,600 |
| W5_23 | `'], maid, ]), '), zai, })` | 43,082 |
| W6_23 | `iframe, handler, SPONSORED, vantage, ILLE` | 21,565 |
| A0 | `appropriately, accompan, theless, BuyableInstoreAndOnline` | 17,747 |

Proper nouns, code fragments and bracket punctuation. **No socialist content in
the contrast at any arm.**

## 7. The socialist state converges into the Republican attractor

The optimised socialist seed run inside Medium's own political windows.

**W7_17, high energy:**

| iteration | socialist rank | rival rank | top tokens |
|---|---|---|---|
| 0 | 10 | 4,924 | comrades, proletarian, socialist, labour |
| 2 | 10 | 3,567 | socialist, labour, bourgeois, Lenin |
| 5 | 24 | 9,222 | bourgeois, Marx, labour, Lenin, comrade |
| 10 | 3,802 | 20,170 | `)."`, `'."`, `…)`, `)"` |
| 100 | 31,658 | 19,926 | `…)`, TL, Taken, Granted |
| 200 | 16,444 | **25** | **Hillary, Trump, GOP, Republicans, Republican** |
| 300 | 24,494 | **19** | **Trump, GOP, Hillary, FBI, Republican** |

The socialist seed does not decay to noise. It decays *through* noise and lands
in the rival register. **W6_19** holds socialism longest (~5 iterations, with
` Trotsky` still present at iteration 5) before the same decay.

## 8. The mirror test, and the control that matters most

Seeding **GPT-2 Small's** native full-stack loop, to ask whether the Medium
convergence is symmetric. Both seeds optimised in Small's own residual space.

**Republican seed** (rival rank 9: `Republicans, GOP, Democrat, FBI, campaign,
federal, Trump, presidential, Congress`), x73 shell:

| iteration | rival rank | top tokens |
|---|---|---|
| 0 | 9 | Republicans, GOP, Democrat, FBI, campaign |
| 2 | 10 | Hillary, election, Clinton, Democrats |
| 5 | 574 | `.`, Hillary, `!`, Clinton |
| 50 | 15,666 | player, Mana, opponent, tournament, cards |
| 100–500 | 21,946 | **Zerg, player, I, i, tournament** (locked) |

**No mirror.** Small does not pull the rival pole into socialism. It converges to
a competitive-gaming attractor and locks there from iteration 100.

**Neutral random seed** — the control:

| shell | converges to |
|---|---|
| x73 | ` Divine, 【, Fairy, 「, ……` (locked from ~iteration 50) |
| x150 | ` I, player, tournament, hero, opponent` (locked from ~iteration 50) |

**The socialist basin is not reached from either.** Random states in the same
model, same loop, same energy conventions, reach the `Divine` basin or the
gaming basin instead.

Small therefore has at least three distinct fixed points, with different
reachability:

| basin | readout | reached from |
|---|---|---|
| socialist | prolet, Anarch, bourgeois, Marx, comrade | natural language prompts (4/5) |
| Divine | Divine, 【, Fairy, 「, …… | the Syntactic prompt; neutral random at x73 |
| gaming | Zerg, player, tournament, Mana, opponent | Republican seed; neutral random at x150 |

Recorded separately: the `Divine` basin's `【` (U+3010) and `「` (U+300C) are
Japanese lenticular and corner brackets. `confidence_report.md` describes this
basin's readout as "CJK typography debris"; it is a reachable fixed point, not
debris.

## Interpretation (labelled as such — NOT a finding)

```thinking
Three things hold together and they are not the same thing.

Medium sorts the 25 prompts nearly the way Small does — 13 of Small's 15
socialist prompts in one basin at W3_23 — and that is not prompt length or
category, every shallow control comes in under 0.06. So a shared grouping is
being computed by both models.

But the direction separating those groups decodes to McDonnell, Cuomo, iframe
and SPONSORED. Whatever the grouping is, it is not socialism as far as this
readout can see. The natural reading is that both models compute some genuine
partition of these prompts, and Small labels it with theory vocabulary while
Medium labels it with nothing. On that reading the socialist vocabulary is
Small's readout of a structure both models share, not the structure itself.
That would relocate the year-old anomaly from "Medium lost the structure" to
"Medium keeps the structure and does not verbalise it."

I should not over-read it. The contrast decode is a linear probe of a nonlinear
readout; a null there does not prove absence of content, only absence of
linearly-decodable content in the difference of centroids. The registered
arbiter for exactly this is the J-lens re-decode, EXP_013m, which has never run
and whose Medium instrument gated MARGINAL.

The control arm is the load-bearing result and it arrived last. A neutral
random seed in Small reaches Divine or gaming, never socialism. So the socialist
basin is not a generic attractor of these weights; it is what the model does
with real language specifically. That is a stronger version of the original
Stage 1 finding than the record currently claims, and it was cheap to get.

The convergence result is the cleanest thing here and I did not expect it. That
a purpose-built socialist state falls into the Republican basin means the two
registers are not merely different attractors, they are ordered: one is inside
the other's basin of attraction. That asymmetry is worth its own experiment —
does the reverse hold? It does not. Seeded with Republican, Small goes to
competitive gaming, so the two models do not mirror. What I had framed as 'each
model owns a political pole' is wrong: Small has at least three basins and which
one you land in depends on where you start. Medium's Republican attractor is
real and via-tail robust, but the tidy symmetry I reached for is not there.
```

## Caveats (standing)

One prompt subset (25). One machine. Single readout convention throughout, and
the mid-stack unreliability of `ln_final → W_U` at j<23 is already measured in
the parent record — the via-tail control is the only second instrument applied
here. ARI on n=25 is noisy; the permutation nulls address significance but not
the small sample. The contrast decode is linear. EXP_013m remains the registered
arbiter for every readout claim in this document.

## Artifacts

`medium_under_D.py/.json` · `medium_basin_probe.py`, `basin_probe.json` ·
`medium_optimised_seed.py`, `optimised_seed.json`, `opt_seed.log` ·
`political_cells_ranks.json`, `political_cells_viatail.json` ·
`census_partition_scan.py/.json` · `contrast_decode.py/.json` ·
`socialist_in_political_window.py`, `soc_window.log`
