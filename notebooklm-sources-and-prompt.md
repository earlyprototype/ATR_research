# NotebookLM source set and podcast prompt

Companion to `potter-embodied-neuroscience-study.md`. This gathers the sources behind that study for loading into NotebookLM, plus a ready-to-paste Audio Overview (podcast) prompt.

Sources are grouped by how well NotebookLM can actually ingest them. NotebookLM fetches real content, so open-access full texts are far more useful than paywalled pages, which yield only an abstract. Every link below was surfaced via web search during the study; full-text pages themselves could not be opened in that session (the environment blocked direct fetching), so treat access levels as expected, not confirmed.

---

## A. Open-access full text (highest priority: read in full by NotebookLM)

- Shaping Embodied Neural Networks for Adaptive Goal-directed Behavior (Chao, Bakkum, Potter 2008, PLoS Comput Biol): https://pmc.ncbi.nlm.nih.gov/articles/PMC2265558/
- Controlling Bursting in Cortical Cultures with Closed-Loop Multi-Electrode Stimulation (Wagenaar, Madhavan, Pine, Potter 2005, J Neurosci): https://pmc.ncbi.nlm.nih.gov/articles/PMC2663856/
- An extremely rich repertoire of bursting patterns during the development of cortical cultures (Wagenaar, Pine, Potter 2006, BMC Neuroscience): https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-7-11
- MEART: the semi-living artist (Bakkum, Gamblen, Ben-Ary, Chao, Potter 2007, Front Neurorobotics): https://www.frontiersin.org/articles/10.3389/neuro.12.005.2007/full
- A low-cost multielectrode system enabling real-time closed-loop processing with rapid recovery from stimulation artifacts (Rolston, Gross, Potter 2009, Front Neuroeng): https://www.frontiersin.org/journals/neuroengineering/articles/10.3389/neuro.16.012.2009/full
- Closed-Loop, Open-Source Electrophysiology (Rolston et al. 2010, Front Neurosci): https://pmc.ncbi.nlm.nih.gov/articles/PMC2940414/
- SALPA, real-time stimulus artifact suppression by local curve fitting (Wagenaar & Potter 2002, preprint PDF): https://danielwagenaar.net/papers/02-WP-preprint.pdf
- Removing Some 'A' from AI: Embodied Cultured Networks (Bakkum et al. 2004, preprint PDF): https://neurolab.gatech.edu/wp/wp-content/uploads/potter/publications/DagstuhlAIBakkumpreprint.pdf
- Effective parameters for stimulation of dissociated cultures using multi-electrode arrays (Wagenaar, Pine, Potter 2004, preprint PDF): https://neurolab.gatech.edu/wp/wp-content/uploads/potter/publications/WagenaarPinePotter2004.pdf

## B. Abstract or metadata only (paywalled: NotebookLM gets the abstract, not the full paper)

- The Neurally Controlled Animat (DeMarse, Wagenaar, Blau, Potter 2001, Autonomous Robots): https://link.springer.com/article/10.1023/A:1012407611130 (a Caltech full-text copy may exist: https://authors.library.caltech.edu/records/jjvm8-6ek69)
- Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task (Bakkum, Chao, Potter 2008, J Neural Eng): https://iopscience.iop.org/article/10.1088/1741-2560/5/3/004
- Region-specific network plasticity: comparison of the center of activity trajectory (CAT) with other statistics (Chao, Bakkum, Potter 2007, J Neural Eng): https://iopscience.iop.org/article/10.1088/1741-2560/4/3/015
- Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks (Sussillo & Barak 2013, Neural Computation): https://direct.mit.edu/neco/article/25/3/626

## C. Context and overview

- Potter lab publication list: https://potterlab.org/publications/
- Closing the Loop Between Neurons and Neurotechnology (Front Neurosci 2010; likely a Potter perspective, verify authorship): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00015/full
- Wikipedia, Hybrot: https://en.wikipedia.org/wiki/Hybrot
- Wikipedia, Neurally controlled animat: https://en.wikipedia.org/wiki/Neurally_controlled_animat

## The study itself, as a source

Add `potter-embodied-neuroscience-study.md` (in this repo) as an uploaded source. It is the only source carrying the honesty flags and the ATR alignment, so including it lets the podcast lean on the verified/unverified distinctions instead of flattening them. Upload the file directly rather than pointing NotebookLM at a repo URL, since the repository is private.

---

## NotebookLM Audio Overview prompt

Paste into the Customize box before generating the Audio Overview.

> Create a focused, honest conversation for a listener preparing to actually talk with the neuroengineer Steve M. Potter. Two threads: (1) explain Potter's research program clearly, and (2) examine how it relates to a separate project called ATR (Activation Tensor Resonance).
>
> For Potter, cover: his core argument that a neural system cannot be understood as an open-loop stimulus-response object and must be studied embodied in a closed sensory-motor loop; the animat and hybrots; MEART; the closed-loop burst-control work (synchronized bursting as a self-reinforcing collective mode that feedback can reshape); and the 2008 goal-directed learning result. Emphasize the pivotal detail that the 2008 learning was feedback-contingent: the same stimuli replayed open-loop did nothing.
>
> For the alignment, treat rhymes as analogies to examine, never as equivalences. The genuine rhyme: an isolated recurrent system, driven only by its own activity, settling into a small set of self-reinforcing low-dimensional modes. The load-bearing differences: Potter's cultures learn because the loop changes synaptic weights through plasticity, while ATR runs on frozen weights so its loop reveals existing structure rather than teaching anything; living versus artificial; native versus imposed recurrence. Make the central tension explicit: Potter argues isolation is the wrong way to understand a neural system, whereas ATR is a claim that controlled isolation can reveal real intrinsic structure.
>
> Discipline: distinguish established findings from interpretive framing. Do not present the "attractor/basin" language as Potter's own words; flag it as an external lens. Do not overstate the strength of the learning result. Do not call the cultures "creative." If something is uncertain, say so. Avoid hype. End on the open questions worth putting to Potter directly.

---

## Two cautions

- NotebookLM's hosts simplify and occasionally overreach, so treat the audio as a warm-up, not a citable record. Check anything surprising against the flagged study.
- For the Section B (paywalled) sources, the podcast will be working from abstracts only, so it should not speak with confidence about those papers' internal methods or numbers.
