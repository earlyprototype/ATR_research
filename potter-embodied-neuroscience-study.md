# Steve M. Potter: Embodied, Closed-Loop Neuroscience

### A sourced study, oriented toward alignment with the Activation Tensor Resonance (ATR) project

Prepared as background for a substantive conversation with Steve M. Potter. The goal is veracity over completeness. Where a claim could not be checked, that is stated rather than smoothed over.

---

## 0. How to read the status flags, and how the sources were gathered (read this first)

Every substantive claim below carries a status flag. The user's original scheme had three levels. One of them, "[verified: primary source read]", I do not use, and here is why.

In this working session, direct fetching of full-text pages was blocked. Every attempt to open a journal page, a PubMed record, a PubMed Central full text, a lab homepage, or even a structured academic API (Crossref, Europe PMC, Semantic Scholar, NCBI E-utilities, OpenAlex) returned an HTTP 403 at the outbound proxy, both through the fetch tool and through command-line curl. Only allowlisted hosts (package registries, GitHub) were reachable that way. What did work, and worked well, was web search: it returned bibliographic records and, in many cases, the paper's own abstract text and short full-text passages as surfaced by the search index. I mined that channel hard, and most claims below are now confirmed at abstract level rather than mere metadata.

So the flags used here are:

- **[verified: secondary source]** means the claim was confirmed against search-surfaced material: a publisher or PubMed record, or abstract or full-text-snippet text quoted by the search index. It was NOT read in the full-text primary paper. For bibliographic facts (who, where, when, pages) this is strong. For findings, it reflects the abstract or an indexed passage, not the full method, results, and statistics.
- **[unverified: background knowledge]** means I am asserting it from general knowledge, without a source confirmed this session. Treat these as prompts to check, not as established fact.

I do not use a "primary source read" flag anywhere, because I did not read a primary source in full: full-text access was blocked. Section 8 lists exactly what remained unverified. The practical implication: I can vouch for each paper's existence, authorship, venue, and headline claim, and often for specific numbers stated in the abstract, but not for the fine detail of methods, controls, or statistics.

---

## 1. The core argument: a neural system is not a stimulus-response object

Potter's central methodological claim can be stated in one sentence, in something close to his own framing.

For most of the twentieth century, neuroscience studied the nervous system in an "open loop": present a stimulus, record the response, repeat. Potter's position is that this linear "stimulate then record the response" approach is inadequate for understanding how neural systems actually work, because in a living animal the brain is never a passive responder. It sits inside a continuous sensory-motor loop: brain to body to environment and back to brain. Its own outputs change the world, and the changed world returns as its next input. Learning, and the ability to predict the consequences of one's own actions, depend on that loop being closed. [verified: secondary source, from the abstracts and descriptions of Potter, Wagenaar & DeMarse 2006 and the closed-loop framing on the lab pages and in the 2010 "Closing the Loop Between Neurons and Neurotechnology" piece]

The operational move that follows is "embodiment". Potter and colleagues argue, in the "Closing the Loop" chapter, that "to learn, a system must have a body to behave with and an environment in which to behave", and that by "re-embodying" a dissociated cultured network (giving it an artificial body and world, and feeding its activity back to it as consequence) network function can be mapped onto behaviour. [verified: secondary source, from the abstract of Potter, Wagenaar & DeMarse 2006]

Three clarifications matter for accuracy:

- The word "embodiment" here is literal and engineered, not metaphorical. It means an actual closed feedback loop between a specific culture of neurons and a specific body (simulated or robotic) in a specific environment, implemented in hardware and software. [verified: secondary source]
- The argument is a claim about method (how you should study a neural system) as much as a claim about biology. The object of study is the loop, not the network in isolation. [verified: secondary source, consistent with the lab's self-description and the group's programmatic titles]
- It is not only rhetoric. The 2008 learning experiment (Section 2.4) provides the empirical teeth: the same "effective" training stimuli, when replayed open-loop rather than contingent on the network's ongoing performance, produced neither the plasticity nor the behaviour. Closing the loop was not decorative; it was the mechanism. [verified: secondary source, from the abstract of Bakkum, Chao & Potter 2008]

This is the thread that most directly meets the ATR project, and I return to it in Section 6. Note now a tension that becomes the most interesting thing to raise with him: his polemic is that you must embody a network to understand it, whereas ATR deliberately does the opposite, sealing a network off from any body or world and looping it onto itself. That mirror-image relationship is a feature to examine, not a coincidence to gloss.

---

## 2. The empirical program: what each experiment actually demonstrated

Potter's lab did not argue for embodiment in the abstract. It built systems. Here is the program, with what each one actually showed kept separate from what it is sometimes claimed to show. A useful piece of his vocabulary: a **hybrot** (his coinage) is a hybrid of living neurons and a robot, a culture on a multi-electrode array whose activity drives a robotic or simulated body and which receives that body's sensory situation back as electrical stimulation. An **animat** is the special case where the body is simulated rather than physical. [verified: secondary source]

### 2.1 The neurally controlled animat (the origin experiment)

**Citation.** DeMarse, T. B., Wagenaar, D. A., Blau, A. W., & Potter, S. M. (2001). "The Neurally Controlled Animat: Biological Brains Acting with Simulated Bodies." *Autonomous Robots*, 11, 305 to 310. [verified: secondary source, publisher and PubMed metadata]

**Correction to the brief.** The lead named this as "DeMarse and Potter". The paper has four authors: DeMarse, Wagenaar, Blau, and Potter. The date (2001) and journal (*Autonomous Robots*) are correct. [verified: secondary source]

**Where.** This work was done at Caltech (Division of Biology), before Potter moved to Georgia Tech. [verified: secondary source; Caltech affiliation appears in the indexed record and a Caltech library copy]

**What it demonstrated.** A living network of dissociated rat cortical neurons, grown on a multi-electrode array (an MEA is a small chip with a grid of electrodes that can both record from and stimulate the cells on it), was interfaced two-way to a computer-generated animat in a virtual world. The culture's distributed activity was read out to drive the animat, and information about the animat's situation was fed back to the culture as electrical stimulation. The stated aim was to study how information is processed and encoded in a living network by watching a network and its behaviour together. [verified: secondary source]

**What it did NOT demonstrate.** The 2001 paper is best read as a proof of principle: it showed the closed loop could be built and run. The target behaviours often described for this paradigm (approach a target, avoid a wall, without colliding) were goals of the hybrot program that developed over subsequent years, not robust results established in 2001. It is easy to over-read the animat as "a brain in a dish that learned to control a body"; the honest 2001 claim is narrower, and the strong learning claims come later (Section 2.4). [verified: secondary source for the proof-of-principle reading; the precise strength of any behavioural claim in the full results was not read]

### 2.2 MEART, the semi-living artist

**Citation.** Bakkum, D. J., Gamblen, P. M., Ben-Ary, G., Chao, Z. C., & Potter, S. M. (2007). "MEART: the semi-living artist." *Frontiers in Neurorobotics*, 1 (2007), DOI 10.3389/neuro.12.005.2007. [verified: secondary source, Frontiers and PubMed metadata; the exact article number was not separately confirmed]

**What it was.** MEART was a collaboration between SymbioticA (an art-science lab in Australia) and the Potter lab in Atlanta, with the artist Guy Ben-Ary. A pneumatically actuated robotic arm made drawings, driven by a living network of rat cortical neurons on an MEA, running as a real-time closed-loop system: the culture behaved (via the arm) and received electrical stimulation as feedback on that behaviour. The culture and the robot were often on different continents, linked over the internet. [verified: secondary source, from the abstract and project descriptions]

**What it demonstrated, honestly.** MEART is two things at once. As science, it was another instance of the embodied-culture paradigm, used (per the abstract) to study the network mechanisms that produce adaptive, goal-directed behaviour. As public engagement and bio-art, it was a vehicle for discussion about neural interfaces, creativity, and biotechnology. It is not evidence that the culture is "creative" or "an artist"; that framing is deliberately provocative and belongs to the art side of the project. Keeping that line clear is part of taking the work seriously. [verified: secondary source for the description; the interpretive caution is mine, flagged]

### 2.3 Closed-loop electrophysiology and stimulus-artifact suppression (the enabling technology)

Everything above depends on solving one hard engineering problem: on a single MEA you want to stimulate and record at the same time, but a stimulus is on the order of volts while a neural signal is on the order of tens of microvolts, roughly a hundred-thousand-fold difference. The stimulus saturates the recording electronics, sometimes blinding them for up to a second, which is fatal if you want to see the network's immediate response and close a fast loop. [verified: secondary source, from the Rolston et al. and NeuroRighter descriptions]

Three contributions matter:

- **SALPA.** Wagenaar, D. A., & Potter, S. M. (2002). "Real-time multi-channel stimulus artifact suppression by local curve fitting." *Journal of Neuroscience Methods*, 120, 113 to 120. SALPA (Suppression of Artifact by Local Approximation) fits local cubic polynomials to the artifact and subtracts them, flattening the baseline so spikes can again be detected by voltage thresholding. It cut the post-stimulus blind period by about an order of magnitude, to under 2 ms. [verified: secondary source for citation, pages, method, and the "under 2 ms" figure]

- **A low-cost real-time closed-loop system.** Rolston, J. D., Gross, R. E., & Potter, S. M. (2009). "A low-cost multielectrode system for data acquisition enabling real-time closed-loop processing with rapid recovery from stimulation artifacts." *Frontiers in Neuroengineering*, 2:12 (DOI 10.3389/neuro.16.012.2009). [verified: secondary source, including volume and article number]

- **NeuroRighter, the open-source platform.** The lab's closed-loop hardware and software line was named NeuroRighter, with a real-time SALPA implementation able to recover an action potential within about 1 ms of a stimulus on an adjacent electrode. Associated papers include "Closed-Loop, Open-Source Electrophysiology" (*Frontiers in Neuroscience*, 2010) and, from the same platform, "Closed-Loop, Multichannel Experimentation Using the Open-Source NeuroRighter Electrophysiology Platform" (*Frontiers in Neural Circuits*, 2012, led by Newman and colleagues). [verified: secondary source for the platform name, the ~1 ms figure, and the venues; exact volumes, article numbers, and the full author lists were not all confirmed]

**What this demonstrated.** Not a finding about brains, but the instruments that made the findings possible: systems that record, decide, and stimulate fast enough and cleanly enough to run a genuine closed loop on living tissue. This is a real and often under-credited part of the contribution. [verified: secondary source for existence; the significance judgment is mine]

### 2.4 Goal-directed learning in an embodied cortical network (the load-bearing learning result)

**Citation.** Bakkum, D. J., Chao, Z. C., & Potter, S. M. (2008). "Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task." *Journal of Neural Engineering*, 5(3), 310 to 323. [verified: secondary source, IOP and PubMed metadata]

**Correction to the brief.** The lead placed this in "PLoS ONE, roughly 2008". The year and authors are right, but the journal is the *Journal of Neural Engineering*, not PLoS ONE. There is a closely related companion in a PLoS journal, listed next. [verified: secondary source]

**The companion and the metric.** Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2008), "Shaping Embodied Neural Networks for Adaptive Goal-directed Behavior", *PLoS Computational Biology*, 4(3), e1000042 (open access, PMC2265558), is the modelling companion: it embodied a **simulated** network through a sensory-motor loop and used an adaptive training algorithm exploiting spike-timing-dependent plasticity (STDP, the rule by which the relative timing of two neurons' spikes strengthens or weakens the synapse between them). The "behaviour" readout in both papers is the **Center of Activity (CA)**, the activity-weighted spatial centroid of firing across the electrode array; its path over time is the **Center of Activity Trajectory (CAT)**, introduced in Chao, Bakkum & Potter (2007), "Region-specific network plasticity in simulated and living cortical networks: comparison of the center of activity trajectory (CAT) with other statistics", *Journal of Neural Engineering*, 4(3). Goal-directed behaviour means steering the CA toward a target region. [verified: secondary source for the CA/CAT metric, STDP, the simulated-network nature of the PLoS paper, and the 2007 citation]

**What the 2008 experiment demonstrated (from the abstract and indexed passages).** A living neocortical network learned, within tens of minutes, to modulate its own dynamics to reach pre-determined activity states, driven by patterned training stimuli through the MEA. The method needed no prior map of the network's functional connectivity; effective training sequences were discovered and refined continuously from real-time feedback on performance. The short-term response to training became "engraved", so progressively fewer training stimuli were needed. After two hours of training, plasticity remained significantly above baseline for about 80 minutes. [verified: secondary source, from the abstract]

**The single most important detail.** A sequence of training stimuli that had been effective did NOT induce significant plasticity or the desired behaviour when simply replayed to the network open-loop, once it was no longer contingent on feedback. In other words, it was the closed-loop contingency, not the stimuli themselves, that drove the learning. This is the empirical core of the whole embodiment argument, and it is the sharpest point of contact and contrast with ATR (Section 6). [verified: secondary source, from the abstract]

**What to hold lightly.** "Learned" and "goal-directed" are load-bearing, contested words in this field. The demonstrated effect is that closed-loop, feedback-contingent patterned stimulation could steer a network toward target activity states faster over time, an effect consistent with activity-dependent plasticity and abolished when the contingency was removed. How large, how reliable, and across how many cultures the effect held are exactly the details that live in the full results, which were not read this session. Worth asking him directly. [verified: secondary source for the shape of the claim; the caution is mine]

### 2.5 Burst control (detailed in Section 3)

**Citation.** Wagenaar, D. A., Madhavan, R., Pine, J., & Potter, S. M. (2005). "Controlling Bursting in Cortical Cultures with Closed-Loop Multi-Electrode Stimulation." *Journal of Neuroscience*, 25(3), 680 to 688. [verified: secondary source, journal and PubMed metadata]

**Correction to the brief.** The lead named this "Wagenaar, Pine, Potter". The burst-control paper has four authors, including Radhika Madhavan. The three-author "Wagenaar, Pine, Potter" combination is correct for two other papers (Section 3). [verified: secondary source]

I treat this result in its own section because it is the clearest window onto the dynamical-systems substance and the closest technical rhyme with ATR.

---

## 3. The dynamical-systems substance: synchronized bursting and how feedback reshapes it

This is the richest technical thread and the one to get exactly right.

### 3.1 The phenomenon

Dissociate cortical neurons from a rodent embryo, grow them at reasonable density on an MEA, and they wire themselves back up and begin to fire. A dominant mode of that firing is the "network burst" or "globally synchronized burst": most of the network falls silent, then almost all of it fires together in a brief, intense volley, then it quiets again, over and over. In dense dissociated cultures this synchronized bursting is a major mode of activity, and, unlike in an intact brain, it persists as a dominant pattern for the lifetime of the culture, reported as up to about two years. [verified: secondary source, from the abstract of Wagenaar, Madhavan, Pine & Potter 2005]

### 3.2 The proposed cause

Wagenaar, Madhavan, Pine, and Potter hypothesised that this persistence is caused by the lack of input from other brain areas. In an intact brain, a cortical region is bathed in afferent input (signals arriving from elsewhere). A dish has none. Their reasoning: replace the missing afferents with electrical stimulation and see whether the runaway synchronization can be tamed. [verified: secondary source, from the abstract]

### 3.3 The intervention and the result

The abstract-level result is specific:

- Slow stimulation through a single electrode actually increased burstiness, because it entrained bursts (it paced them rather than dispersing them).
- Rapid stimulation reduced burstiness.
- The strongest control came from two moves together: distributing the stimuli across several electrodes, and continuously fine-tuning stimulus strength with closed-loop feedback. That combination greatly enhanced burst control. [verified: secondary source, from the abstract of Wagenaar, Madhavan, Pine & Potter 2005]

The plain reading: the synchronized-burst mode is not a fixed fact about the tissue. It is a dynamical regime the network falls into for want of structured input, and appropriately shaped feedback can push the network out of it into a more dispersed, less globally synchronized regime.

### 3.4 The two related "Wagenaar, Pine, Potter" papers

- Wagenaar, D. A., Pine, J., & Potter, S. M. (2004). "Effective parameters for stimulation of dissociated cultures using multi-electrode arrays." *Journal of Neuroscience Methods*, 138, 27 to 37. The methods groundwork: what stimulation reliably drives these cultures. [verified: secondary source]
- Wagenaar, D. A., Pine, J., & Potter, S. M. (2006). "An extremely rich repertoire of bursting patterns during the development of cortical cultures." *BMC Neuroscience*, 7:11 (open access). They followed 58 cultures of varying density (about 3,000 to 50,000 neurons on areas of roughly 30 to 75 square millimetres) over the first five weeks of development. Two headline findings: bursting is not one stereotyped pattern but a wide, culture-specific, developmentally shifting repertoire; and plating density strongly shaped development, with dense cultures beginning to burst earlier and (from stimulation responses) growing axons faster. [verified: secondary source for the citation, the 58-culture design, the density figures, and the two headlines; the detailed pattern taxonomy was not read in full]

### 3.5 The dynamical-systems reading (framing, flagged as such)

It is fair, and useful for the ATR alignment, to describe synchronized bursting in dynamical-systems language: a high-dimensional system (thousands of neurons) that spontaneously and repeatedly collapses onto a small, self-reinforcing collective mode, that is, low-dimensional behaviour emerging from a high-dimensional substrate, robust enough to recur for the life of the culture, and reshaped when feedback changes the effective input. That description is consistent with the empirical papers.

But provenance matters. The specific vocabulary of "low-dimensional attractor", "basin", and "state-space collapse" is the standard vocabulary of the adjacent theoretical literature (Section 4). I did not confirm this session that Potter's own burst papers formally quantified the dimensionality of the dynamics (for example, by principal component analysis of population activity) or used the word "attractor" for the burst state. Interestingly, the group's later behaviour metric, the Center of Activity Trajectory (Section 2.4), is itself a low-dimensional (two-dimensional) reduction of the population activity, which shows they thought in reduced-dimensional terms, but that is a metric choice, not a formal attractor analysis. So the "self-reinforcing low-dimensional mode" reading is best offered to him as an interpretive lens to test against his own view, not asserted as his lab's stated claim. [verified: secondary source for the phenomenology and the CAT metric; unverified: background knowledge for the attractor framing being his own]

---

## 4. The intellectual neighbourhood: living networks and the dynamical-systems view of computation

Potter's living-network work sits next to, but is not the same as, a theoretical tradition that analyses neural systems as dynamical systems. The shared idea: a recurrent network's behaviour is best understood as motion in a state space shaped by fixed points (states the system tends to sit at), attractors (states it is pulled toward), and the low-dimensional structure that organises the high-dimensional activity.

**The reference point named in the brief.** Sussillo, D., & Barak, O. (2013). "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626 to 649. Their method: take a trained artificial recurrent neural network, find the fixed points and slow points of its dynamics by optimisation, then linearise around those points to reverse-engineer what the network is doing. The governing insight is exactly the one that makes Potter's cultures interesting dynamically: high-dimensional recurrent networks often organise their computation on a low-dimensional set of states. [verified: secondary source, from the abstract and metadata]

**How it relates to Potter, and how it does not.** The relationship is thematic and by analogy, not lineage. Sussillo and Barak analyse artificial networks with known, differentiable weights, which is what makes fixed-point-finding tractable. Potter's networks are living tissue whose synaptic weights are neither known nor static. The two share a vocabulary and a hypothesis (low-dimensional dynamics in recurrent systems) while working on opposite kinds of object. Potter's contribution to this neighbourhood is empirical and instrumental: real recurrent tissue, a real closed loop, real feedback reshaping a real collective mode. The formal state-space analysis is mostly done by others, on models. [unverified: background knowledge, as a characterisation of the division of labour]

**Adjacent traditions worth naming, because they are the shared water:**

- Attractor networks and content-addressable memory (the Hopfield-network idea, 1982, that a recurrent network stores patterns as attractors). [unverified: background knowledge]
- Population and neural-manifold analyses in systems neuroscience (associated with, among others, Churchland, Shenoy, and Sussillo), treating population activity as trajectories on low-dimensional manifolds. [unverified: background knowledge]
- Reservoir computing and FORCE learning (Sussillo & Abbott, 2009), where a fixed or lightly trained recurrent network's rich intrinsic dynamics are read out for computation. This one is a particularly apt cousin, because it, like ATR, exploits the intrinsic dynamics of a network that is not fully trained. [unverified: background knowledge for the specific attribution]
- Small-circuit dynamical neuroscience in the tradition of Eve Marder's work on the crustacean stomatogastric ganglion, which established that a fixed small circuit can produce many dynamical modes depending on modulation, a living-tissue cousin of the "one network, many attractors" idea. [unverified: background knowledge]

The honest summary: Potter is not primarily a dynamical-systems theorist. He built the living, closed-loop instruments that the dynamical-systems view of recurrent computation can be tested against. The vocabulary is shared; the methods are complementary rather than identical.

---

## 5. Lab identity and biography (for conversational grounding)

- Potter led the Laboratory for Neuroengineering (the Neurolab), associated with the Coulter Department of Biomedical Engineering at Georgia Tech (jointly with Emory University School of Medicine). His maintained web presence includes potterlab.org, potterlab.gatech.edu, and neurolab.gatech.edu. [verified: secondary source, from lab-page titles, domains, and the Georgia Tech and Emory affiliations that appear in the group's papers; the pages themselves were blocked this session]
- The animat work began at Caltech (around 1999 to 2001); the embodied-culture program matured after his move to Georgia Tech. He coined the term "hybrot". [verified: secondary source for the Caltech origin and the hybrot coinage; the exact move date is background knowledge]
- He is described as retired or as holding an associate or adjunct role. [verified: secondary source, from profile metadata; the precise current title and retirement year were not confirmed]

If precision on titles, dates, or the exact lab name matters in conversation, treat the above as approximate and confirm with him.

---

## 6. Alignment map: ATR and Potter's program

This section maps the user's project, Activation Tensor Resonance (ATR), against Potter's work. The discipline: present genuine structural rhymes as analogies to examine, and be at least as clear about where they break as about where they hold. ATR's own description is taken as given from the user; it is an exploratory art-science process that produced reproducible, independently reviewed findings, not an established method, and nothing below upgrades that status.

A one-line reminder of ATR: take a small open-weight language model, feed its internal activation (the residual stream) back in as its own next input, rescale to constant energy, and iterate hundreds of times. The text dissolves and the state settles into a small number of attractors (about five semantic basins in GPT-2 Small, four thematically coherent). A corpus or bias fingerprint explanation was tested and refuted. A noise control showed random inputs also converge, but to different, meaningless basins, so the semantic basins are a property of the landscape as visited from where language lives, not of the model alone. Attractor structure differs across models. One basin turned out to be an exact two-state cycle sustained by a single network component.

### 6.1 Where the two genuinely rhyme (analogies to examine, not equivalences)

1. **An isolated recurrent system driven only by its own activity.** Potter's dissociated culture is cut off from the afferent input a real cortex would receive; in the burst-control work, its "world" is only what is looped back to it. ATR seals a language model off from any external prompt and loops its own internal state back in. Both ask the same shaped question: what does a recurrent system do when the only thing driving it is itself? This is the strongest rhyme, and it is real. [rhyme; grounded in verified descriptions of both]

2. **Collapse onto a small set of self-reinforcing modes.** The culture repeatedly collapses onto the synchronized-burst mode. ATR collapses onto roughly five basins in GPT-2 Small. In both, a high-dimensional system driven by its own activity settles into a low-dimensional set of stable configurations. [rhyme]

3. **Feedback reshapes the modes.** In the culture, rapid, distributed, closed-loop stimulation pushes the network out of the global-burst regime. In ATR, the rescale-to-constant-energy step plus iteration is the operation that drives the state into (and holds it in) an attractor. In both, the specific feedback rule is not incidental; it determines which regime you land in. [rhyme]

4. **The attractor structure is a property of the landscape, and depends on where you start.** ATR's noise control is the sharp version: random starts converge, but to different, meaningless basins, so the meaningful basins reflect the landscape as entered from the region "where language lives". This has a genuine cousin in Potter's world: whether a stimulus entrained or dispersed bursting depended on its structure and site, and different cultures settle into different bursting repertoires. Both point to attractor structure as system-and-initial-condition specific, not universal. The most philosophically interesting rhyme, and worth examining slowly with him. [rhyme; the culture side is my synthesis across the 2005 and 2006 papers, flagged]

5. **System-specific attractor structure across instances.** ATR finds different structure across models (GPT-2 Small several basins, GPT-2 Medium one, Pythia different again). Potter's cultures show a rich, culture-to-culture repertoire (the 2006 paper's central point). Both resist a single universal answer; the modes are a property of the particular network. [rhyme]

### 6.2 Where the two do NOT align (the load-bearing disanalogies)

1. **Plasticity versus a frozen landscape, and feedback-contingency versus raw recirculation. This is the biggest one, and the 2008 result makes it precise.** Potter's whole program is about learning: the closed loop changes the culture's synaptic weights through activity-dependent plasticity, and, crucially, that learning was contingent on feedback. The exact same stimuli replayed open-loop did nothing. ATR runs on frozen weights and has no external contingency at all: it recirculates the system's own state, rescaled, with nothing evaluating or gating it. So the very mechanism that made Potter's loop a teacher (feedback-contingent plasticity) is precisely what ATR lacks. Potter's loop changes the network; ATR's loop leaves the network untouched and merely traverses its fixed landscape. Any sentence that lets ATR sound like it "trains" the model, or that lets Potter's cultures sound like they merely "reveal" pre-existing structure, is wrong in both directions. [disanalogy; grounded in the verified 2008 feedback-contingency finding]

2. **ATR rhymes with Potter's isolated dish, not with his embodiment thesis.** This is subtle and, I think, the single most useful thing to bring to him. Potter's headline argument is that a network must be embodied (given a body and a world, with its activity returning as consequence) to be understood. ATR has no body, no environment, and no consequence: it is pure introspection, the system's own internal state fed back rescaled. So ATR's true analog in his corpus is the un-embodied, self-driven bursting culture (his effective control condition), not the animat, the hybrot, or MEART. The Alvin Lucier lineage makes this precise: "I Am Sitting in a Room" reveals the fixed resonant modes of a room by recirculation, which maps onto revealing intrinsic dynamical modes of a fixed network, not onto learning through a world. The two projects are, in intent, near mirror images: he closes the loop through a world to make a network behave and learn; ATR closes the loop through nothing but the network to make a frozen network show its resting modes. [disanalogy and framing; the interpretive claim is mine, flagged]

3. **Native recurrence versus imposed recurrence.** Potter's culture is physically, natively recurrent: real neurons with real recurrent synapses. A transformer language model is not natively recurrent in its weights; ATR manufactures the recurrence by looping activations across iterations. So "recurrent system" means something structurally different on each side. Worth stating so the rhyme is not overclaimed. [disanalogy; unverified: background knowledge on the architectural point, standard]

4. **Living versus artificial, in-vitro versus in-silico.** One side is wet tissue with metabolism, development, real noise, and finite lifespan (recall the 2006 paper watched cultures develop over weeks); the other is largely deterministic computation. Time, noise, and variability mean different things on each side. [disanalogy]

5. **Scale, and the ambiguity of "size".** Potter's cultures ran from a few thousand up to about 50,000 neurons. GPT-2 Small is about 124 million parameters over 12 layers. These are not commensurable counts (a synapse is not a parameter, a neuron is not a unit), so "which is bigger" is not even well posed. Note the scale, do not equate it. [disanalogy; the culture figures are verified secondary, the GPT-2 figure is background knowledge]

### 6.3 The shape of the alignment, stated once, plainly

ATR and Potter's burst-control work rhyme because both take a recurrent network, cut it off from the world, drive it only with its own activity through a specific feedback rule, and watch it fall into a small set of self-reinforcing low-dimensional modes whose structure is specific to the particular network and to where you start. They diverge because his loop changes the network (feedback-contingent plasticity, learning, embodiment through a world) while ATR's loop leaves the network frozen and merely exposes it. The most productive thing to put to Potter is not the similarity but the tension: his life's argument is that isolation is the wrong way to understand a neural system, and ATR is a claim that controlled isolation can reveal something real about a network's intrinsic structure. He will have a view, and that is the conversation worth having.

---

## 7. Open questions to raise with Potter

1. Did your lab ever formally quantify the dimensionality of the bursting dynamics (for example, PCA of population activity), and would you call the synchronized-burst state an attractor in the dynamical-systems sense, or is that other people's language laid over your phenomenon? (You clearly thought in reduced dimensions with the Center of Activity Trajectory; how far does that go?)
2. In the burst-control work, how much did the starting condition (which electrodes, which culture) determine which regime the network fell into? Is there a living-tissue analog to a basin of attraction that depends on where you enter the state space?
3. Your 2008 result showed that replaying effective stimuli open-loop did nothing: learning needed the feedback contingency. Given that, do you think there is anything real to learn about a network from studying it deliberately un-embodied and self-driven, as in the bare bursting culture, or as in feeding a frozen artificial network its own activity?
4. How strong, in your own assessment, was the "learning" in the 2008 task, and where would you place the line between activity-dependent plasticity and learning proper?
5. If you had to name the intrinsic modes a culture "wants" to fall into, absent any structured input, what would you say they are, and how many are there?
6. Given ATR (a frozen artificial network looped onto itself, settling into a few semantic attractors with no feedback contingency), do you see that as adjacent to your work, orthogonal to it, or a category error, and why?

---

## 8. What could not be verified this session (read before relying on any detail)

**Retrieval status, stated fully.** Web search was reachable and, used thoroughly, productive: it returned bibliographic records and abstract or short full-text passages from PubMed, publisher pages (Springer, IOP, Frontiers, BMC, MIT Press), Semantic Scholar, ResearchGate, and the lab's own page listings. Direct full-text fetching was blocked for every host tried, HTML pages and structured APIs alike (PubMed, PubMed Central, jneurosci.org, Frontiers, BMC, IOP, Wikipedia, potterlab.org, neurolab.gatech.edu, Daniel Wagenaar's site, and the Crossref, Europe PMC, Semantic Scholar, NCBI E-utilities, and OpenAlex APIs): all returned HTTP 403 at the outbound proxy, via both the fetch tool and command-line curl. Only allowlisted infrastructure hosts were reachable that way. So no full-text primary source was read this session; everything here is at abstract, indexed-passage, or metadata resolution.

**Specific items not confirmed, to check before use:**

1. **No primary full text was read.** Methods, controls, effect sizes, and statistics were not verified. This most affects the strength and reliability of the 2008 learning result (including how many cultures showed it), the fine structure of the 2005 burst results, and the pattern taxonomy of the 2006 "rich repertoire" paper.
2. **Some exact bibliographic minutiae** remain unconfirmed: the MEART article number; the full author lists and exact volumes/article numbers of the 2010 *Frontiers in Neuroscience* and 2012 *Frontiers in Neural Circuits* NeuroRighter papers. The SALPA (120:113 to 120), Rolston 2009 (Front. Neuroeng. 2:12), animat (Auton. Robots 11:305 to 310), burst-control (J. Neurosci. 25(3):680 to 688), Bakkum 2008 (J. Neural Eng. 5(3):310 to 323), Chao 2008 (PLoS Comput. Biol. 4(3):e1000042), and Sussillo & Barak (Neural Comput. 25(3):626 to 649) citations were each confirmed at the level stated.
3. **The dynamical-systems framing of bursting** (attractor, basin, low-dimensional mode) was NOT confirmed to be language Potter's own burst papers use. It is presented as an interpretive lens from the adjacent literature, to test with him. The Center of Activity Trajectory is a genuine low-dimensional reduction his lab did use.
4. **Lab identity specifics** (the exact formal name of the Laboratory for Neuroengineering, Potter's precise current title, his retirement year, and the Caltech-to-Georgia-Tech move date) were not read from the lab pages, which were blocked. The Caltech origin of the animat, the Georgia Tech and Emory affiliations, and the hybrot coinage are confirmed at secondary level.
5. **The adjacent-tradition attributions in Section 4** (Hopfield 1982, Sussillo & Abbott 2009 FORCE, Marder's stomatogastric work, neural-manifold analyses) are background knowledge, included to sketch shared vocabulary, and were not individually verified this session.
6. **The 2010 "Closing the Loop Between Neurons and Neurotechnology"** piece (*Frontiers in Neuroscience*) is cited as a likely statement of Potter's philosophy based on its title and venue; its authorship and content were not confirmed in full. Verify before quoting it as his.

**Corrections to the brief, restated for prominence:** the goal-directed learning paper (Bakkum, Chao, Potter, 2008) is in the *Journal of Neural Engineering*, not PLoS ONE; a related companion (Chao, Bakkum, Potter, 2008) is in *PLoS Computational Biology* and used a simulated network. Two papers named with three authors in the brief actually have four: the animat (add Wagenaar and Blau) and burst control (add Madhavan).

---

## Annotated bibliography

Each entry carries a source-status mark. "[verified: secondary source]" means confirmed via search-surfaced metadata, abstract, or indexed passage, not full text read. Full-text reading was blocked this session (Section 8).

1. **DeMarse, T. B., Wagenaar, D. A., Blau, A. W., & Potter, S. M. (2001).** "The Neurally Controlled Animat: Biological Brains Acting with Simulated Bodies." *Autonomous Robots*, 11, 305 to 310. The origin experiment: a rat cortical culture on an MEA interfaced two-way to a simulated animal. Proof of principle for the closed loop. Done at Caltech. [verified: secondary source]

2. **Wagenaar, D. A., & Potter, S. M. (2002).** "Real-time multi-channel stimulus artifact suppression by local curve fitting." *Journal of Neuroscience Methods*, 120, 113 to 120. Introduces SALPA (local cubic-polynomial fitting), which cut the post-stimulus blind period by roughly an order of magnitude, to under 2 ms, enabling fast closed loops. [verified: secondary source]

3. **Bakkum, D. J., Shkolnik, A. C., Ben-Ary, G., Gamblen, P., DeMarse, T. B., & Potter, S. M. (2004).** "Removing Some 'A' from AI: Embodied Cultured Networks." In *Embodied Artificial Intelligence* (Dagstuhl seminar, July 2003; revised selected papers), Springer LNAI (DOI 10.1007/978-3-540-27833-7_10). A programmatic statement of the embodied-culture philosophy. [verified: secondary source; author list confirmed via search]

4. **Wagenaar, D. A., Pine, J., & Potter, S. M. (2004).** "Effective parameters for stimulation of dissociated cultures using multi-electrode arrays." *Journal of Neuroscience Methods*, 138, 27 to 37. Methods groundwork on what stimulation reliably drives these cultures. [verified: secondary source]

5. **Wagenaar, D. A., Madhavan, R., Pine, J., & Potter, S. M. (2005).** "Controlling Bursting in Cortical Cultures with Closed-Loop Multi-Electrode Stimulation." *Journal of Neuroscience*, 25(3), 680 to 688. The key dynamical result: synchronized bursting persists for want of afferent input; rapid, distributed, closed-loop stimulation reduces it. Central to Section 3. [verified: secondary source]

6. **Potter, S. M., Wagenaar, D. A., & DeMarse, T. B. (2006).** "Closing the Loop: Stimulation Feedback Systems for Embodied MEA Cultures." In Taketani, M., & Baudry, M. (eds), *Advances in Network Electrophysiology: Using Multi-Electrode Arrays*. Springer, Boston. States the embodiment thesis: to learn, a system needs a body and an environment; re-embodying cultures maps network function onto behaviour. Best single source for the core argument in his own framing. [verified: secondary source]

7. **Wagenaar, D. A., Pine, J., & Potter, S. M. (2006).** "An extremely rich repertoire of bursting patterns during the development of cortical cultures." *BMC Neuroscience*, 7:11 (open access). 58 cultures, about 3,000 to 50,000 neurons, first five weeks; bursting is a wide, culture-specific, developmentally shifting repertoire, and density strongly shapes development. Supports the "system-specific attractor structure" rhyme. [verified: secondary source]

8. **Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2007).** "Region-specific network plasticity in simulated and living cortical networks: comparison of the center of activity trajectory (CAT) with other statistics." *Journal of Neural Engineering*, 4(3). Introduces the Center of Activity Trajectory, the low-dimensional behaviour metric used in the learning work. [verified: secondary source]

9. **Bakkum, D. J., Gamblen, P. M., Ben-Ary, G., Chao, Z. C., & Potter, S. M. (2007).** "MEART: the semi-living artist." *Frontiers in Neurorobotics*, 1 (2007). The robotic-arm drawing system driven by a cultured network, with SymbioticA and Guy Ben-Ary. Science and bio-art at once; not evidence of machine creativity. [verified: secondary source]

10. **Bakkum, D. J., Chao, Z. C., & Potter, S. M. (2008).** "Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task." *Journal of Neural Engineering*, 5(3), 310 to 323. The learning result: feedback-contingent patterned stimulation steers a network's Center of Activity toward a target within tens of minutes, the effect becoming "engraved" and lasting about 80 minutes after two hours of training, and abolished when stimuli were replayed open-loop. Corrects the brief's "PLoS ONE" attribution. [verified: secondary source]

11. **Chao, Z. C., Bakkum, D. J., & Potter, S. M. (2008).** "Shaping Embodied Neural Networks for Adaptive Goal-directed Behavior." *PLoS Computational Biology*, 4(3), e1000042 (open access, PMC2265558). Modelling companion to entry 10, using a simulated network and a spike-timing-dependent-plasticity-based training rule. [verified: secondary source]

12. **Rolston, J. D., Gross, R. E., & Potter, S. M. (2009).** "A low-cost multielectrode system for data acquisition enabling real-time closed-loop processing with rapid recovery from stimulation artifacts." *Frontiers in Neuroengineering*, 2:12. Part of the NeuroRighter closed-loop platform line (see also *Frontiers in Neuroscience* 2010 and *Frontiers in Neural Circuits* 2012). The instruments behind the science. [verified: secondary source for this entry; the 2010 and 2012 companions' full details were not all confirmed]

13. **Sussillo, D., & Barak, O. (2013).** "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626 to 649. Not Potter's work. The reference point for the dynamical-systems reading: find fixed and slow points in trained RNNs and linearise to reverse-engineer them. Adjacent tradition, shared vocabulary (Section 4). [verified: secondary source]

*Cited more tentatively:* "Closing the Loop Between Neurons and Neurotechnology" (*Frontiers in Neuroscience*, 2010), a likely statement of Potter's philosophy identified by title and venue; authorship and content not confirmed in full. [verified: secondary source for existence; unverified for authorship]

---

*Prepared 2026-07-21. Retrieval channel: web search only; full-text fetching (HTML pages and structured APIs) was blocked by the environment network policy (HTTP 403 on all non-allowlisted hosts). Treat every claim at the resolution of its status flag.*
