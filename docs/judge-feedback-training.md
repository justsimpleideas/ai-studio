# Judge-feedback training: a vision model grades outputs, the image model learns

Status: **on hold by decision** (see S12 in [decisions.md](decisions.md)). No code has
been written for this yet. This file records the idea and the analysis so it can be
picked up later without redoing it.

## The idea

Today a LoRA learns from the user's own photos ([training.md](training.md)). This
adds a second source of training signal: the studio generates images from prompts,
a vision-language model (the "judge") checks each image against its prompt, and that
verdict decides what the image model learns from. Repeated in rounds, the model
should get measurably better at following prompts.

The motivating failures are small, consistent ones: a hand with four fingers, the
right hand raised when the prompt said left, two objects when the prompt asked for
three. The same flow applies to any mistake the judge can check.

## Judge setup (chosen if this goes ahead)

- **Default: the local vision model** the studio already installs for captioning.
  Free, offline, nothing leaves the machine. Weaker than a large hosted model.
- **Option: an external vision API** with the user's own key, as a stronger
  "teacher". Off by default and switched on per run, never globally. Before the run
  starts the user sees what will be sent (the images and prompts of that run only)
  and an estimate of how many requests it makes; every request goes through the
  internet gate (S9 in [decisions.md](decisions.md)) and is logged.
- Both sit behind one judge interface, so the rest of the pipeline does not know
  which one answered. The external option is named generically in the UI and code.

## Turning verdicts into training

| Method | How | Fits 12 GB | Expected effect |
|---|---|---|---|
| **1. Best-of-N, then fine-tune** | Generate 4–8 images per prompt, keep only those the judge passes, train a LoRA on the kept images with their prompts as captions | Yes, reuses the existing LoRA trainer | Fixes consistent habits (left/right mix-ups, wrong counts in common phrasings) |
| **2. Preference pairs (Diffusion-DPO)** | For one prompt, a passed image and a failed one form a pair; the loss pushes toward the first and away from the second | Yes with a LoRA: the reference model is the base with the adapter switched off, so no second copy of the weights | More direct at removing a specific error than method 1 |
| 3. Reward-driven RL (DDPO / GRPO-style) | Score every sample, update the policy online | No: very many judge calls, unstable, prone to gaming the judge | Not planned |
| Bonus: fix at generation time | Judge flags a bad region, regenerate or inpaint just that region | Yes | Immediate, no training; complements the above |

Order if built: method 1 first (shortest path to an end-to-end result), then
method 2.

### The piece that makes "gradually better" a number

A **fixed evaluation prompt set** that is never trained on. After every round the
model generates from it and the judge scores it; the studio plots the score per
round. A round that lowers the score is discarded, which also catches the model
learning to please the judge instead of the prompt.

## Judging well

- **Ask small questions, not for a grade.** Split each prompt into a checklist of
  yes/no questions ("Is there a person?", "Is a hand raised?", "Is it the person's
  own left hand?") and ask them one at a time, in the style of TIFA / DSG. Holistic
  1–10 scores are noisier.
- **Measure the judge before trusting it.** Counting fingers and telling the
  subject's left from right are known weak spots of vision-language models, large
  ones included. A small hand-labelled set is needed to measure how often each judge
  is right on each kind of question; question kinds a judge gets wrong often are
  not used for training.
- "Left" is ambiguous (the subject's left or the viewer's left). The checklist
  states which one it means.

## Honest limits

- A LoRA trained on a few thousand samples corrects habits well; it improves deep
  abilities such as hand anatomy only modestly.
- Video costs several times more on both sides: training on 12 GB is limited to
  short low-resolution clips, and judging a clip means sampling frames. Images first.
- The external option costs money per request and needs a key separate from any
  subscription; whether a provider's terms allow using its outputs to train a model
  has to be checked against that provider's terms by the user.
- No image or video model is installed on the development machine at the time of
  writing, so development and tests would use the fake engine and a fake judge; a
  real run waits for a model to be installed.

## Rough phases (if resumed)

1. Judge module: prompt → checklist, local judge, fake judge for tests, judge
   accuracy measurement on a labelled set.
2. Best-of-N generation, filter, hand off to the existing LoRA trainer; evaluation
   set and per-round score chart.
3. External judge option behind the internet gate, with the per-run consent screen
   and request estimate.
4. Diffusion-DPO on preference pairs.
5. Optional: generation-time fix (regenerate / inpaint flagged regions).
