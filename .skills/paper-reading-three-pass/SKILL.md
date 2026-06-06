---
name: paper-reading-three-pass
description: Use this skill when reading an academic paper PDF and producing an integrated Markdown deep-reading note using a senior algorithm engineer's three-pass reading method internally: quick screening, structured full pass, and deep reconstruction.
---

# Integrated Three-Pass Paper Reading

Use this workflow to turn a paper PDF into a useful Markdown deep-reading note. The three passes are an internal reading method: first decide whether the paper deserves attention, then understand its structure, then reconstruct the author's research process deeply enough to reuse or critique it.

The final Markdown should not mechanically expose the three passes as separate `Pass 1`, `Pass 2`, and `Pass 3` sections unless the user explicitly asks for that format. Instead, synthesize the information from different reading depths into one organized document with natural sections such as problem, contribution, method, evidence, ablations, limitations, reproducibility, and follow-up ideas.

## Inputs

- A paper PDF path.
- Optional user context: research direction, reading goal, target depth, and whether they want implementation details.

If no PDF is available, ask for the PDF path before producing the paper summary.

## Workflow

### 1. Pass One: Quick Screening

Spend the first pass on selection and orientation.

Read:
- Title
- Abstract
- Conclusion
- Key figures and tables in the method and experiment sections

Extract:
- What problem the paper claims to solve
- Whether it is relevant to the user's goal
- The headline method or idea
- The main empirical result, including concrete numbers when present
- A first quality signal: clear problem, convincing result, plausible method, and relevance

Decision:
- `Stop`: unrelated, low quality, or only worth knowing at a surface level
- `Continue`: relevant enough for a full structured read

### 2. Pass Two: Structured Full Read

Read from beginning to end, but skip dense proof details, derivations, and low-level formula manipulation unless they are central.

Focus on:
- What each section contributes to the argument
- The full method pipeline and how its components connect
- Every important figure and table: axes, labels, comparison methods, metrics, and gap size
- Which baselines are used and whether they are fair
- Which prior works are foundational, directly improved upon, or required background
- What remains unclear after this pass

Decision:
- `Stop at understanding`: enough to know the problem, method, and result
- `Read prerequisites`: if the paper is too hard because key cited work is missing
- `Deep read`: if the paper is central to the user's work or needs implementation/research follow-up

### 3. Pass Three: Deep Reconstruction

Read until the paper can be mentally reconstructed.

For each important paragraph, ask:
- Why is this sentence here?
- What problem is the author handling at this point?
- If I were doing this research, what would I try here?
- Why did the author choose this method instead of alternatives?
- How exactly would I reproduce this method or experiment?
- Can the stated limitation become a next research idea?

Deep-read outputs should include:
- Algorithm/pipeline reconstruction
- Experiment reconstruction
- Assumptions and hidden dependencies
- Failure modes and limitations
- Reproducibility checklist
- Potential improvements or follow-up ideas

## Markdown Output Template

```markdown
# Paper Summary: <paper title>

## Reading Verdict
- Depth reached: Quick screening / Structured full read / Deep reconstruction
- Decision: Stop / Read prerequisites / Keep as reference / Implement / Build on it
- Relevance: <why this paper matters or does not matter for the user's goal>

## Metadata
- Title:
- Authors:
- Venue / year:
- PDF:

## One-Sentence Contribution
<State the core contribution in one precise sentence.>

## Why This Paper Matters
<What problem is being solved, why it is important, and where it sits in the research or engineering landscape. Integrate the quick-screening judgment here instead of writing a separate Pass 1 section.>

## Core Idea
<The main method or insight. Avoid vague restatement.>

## Method Reconstruction
<Explain the pipeline, model, algorithm, or theoretical construction step by step.>

## Experiments and Evidence
<Summarize datasets, metrics, baselines, ablations, and the most important numbers.>

## Ablations and What Actually Matters
<If the paper contains ablations or controlled comparisons, explain what they prove and what they do not prove. Omit this section if not applicable.>

## Key Figures and Tables
| Item | What it shows | Why it matters |
| --- | --- | --- |
| Figure/Table N |  |  |

## Strengths
- 

## Limitations and Risks
- 

## Reproducibility Notes
- Code/data availability:
- Required implementation details:
- Missing details:
- Estimated reproduction difficulty:

## Questions After Reading
- 

## Follow-Up Ideas
- 

## Important Prior Work to Chase
| Citation | Why it matters |
| --- | --- |
|  |  |
```

## Quality Rules

- Use the three-pass method to decide reading depth, but make the final document feel like one coherent study note rather than a reading log.
- Do not create visible sections named `Pass 1`, `Pass 2`, or `Pass 3` unless the user explicitly requests a pass-by-pass reading record.
- The `Reading Verdict` may briefly state the depth reached, but the body should be organized around the paper's ideas, evidence, and reuse value.
- Preserve concrete numbers, datasets, metrics, and baseline names when available.
- Do not invent missing metadata or results. Mark them as `not found in extracted text`.
- Tie claims back to paper sections, pages, figures, or tables when possible.
- If the PDF extraction loses figures or tables, say so and infer only from captions/text.
- Keep the quick-screening judgment concise inside `Reading Verdict` and `Why This Paper Matters`. Only produce deep-reconstruction detail when the paper warrants it or the user asks for depth.
