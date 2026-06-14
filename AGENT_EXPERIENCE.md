# What an agent sees, and how it reacts

This document walks through the experience of being a Terra Incognita agent:
the exact text a model receives at each consultation, where every line comes
from, and what a competent mind does with it. All excerpts are from a real
session (seed 42, 2 agents, central book) played by hand through
`scratch/play.py`, the human-as-brain replay driver. It reflects the prompt
as of v0.39 (post-playtest fixes).

## The shape of a life

An agent is consulted only at decision points; between them, the engine's
Tier B executor walks paths, gathers, and fires survival reflexes on its
own. A consultation is one LLM call: the agent receives a system prompt
(identity + the few things every survivor knows) and a user prompt (its
current situation), and must answer with a single JSON object containing a
plan of up to 20 primitive actions, plus — crucially — its rewritten beliefs
and private notes. **Whatever it does not write down, it forgets**: the next
consultation rebuilds its entire mind from those two fields. This is the
core epistemics mechanic. An agent that takes good notes resumes
interrupted experiments; one that doesn't wanders.

Consultations are triggered by: plan finished, a step failed, the
replan-interval timer (default 15 ticks), being spoken to directly, or a
survival reflex having overridden the plan. The prompt now says which
(`== WHY NOW ==`, below).

## The system prompt

> You are A01, a survivor in an unknown world. Basic facts you know: your
> energy drains every tick (more when moving) and at 0 energy you die,
> permanently. Berries are food — eat(berries) restores 25 energy. Wood and
> stone are materials, not food. You must stand on a tile to gather from it;
> give gather a target x, y and you will walk there first. Beyond that you
> do NOT know how this world works. You can discover how to create things by
> combining items (action: combine), but failures may destroy ingredients.
> You can talk to nearby agents — your voice only carries a few tiles. Your
> goals: survive, and understand this world. The world is a 32x32 grid whose
> edges wrap around (stepping off one side brings you out the opposite side,
> and the shortest path may cross an edge). This world contains a single
> Book listing open questions about how the world works. Filling it with
> correct answers is a shared goal of your society.

What's **given** is embodied common sense: metabolism, that berries are
food, how gathering works, that speech is local, the world's shape. These
were chosen because withholding them measures confusion, not discovery
(m3 runs: agents starved while mining stone). What's **withheld** is
everything the experiments measure: every recipe, regrowth rules, spoilage,
terrain coverage, what failure costs. The final sentences vary with the
book condition (`personal` books say "you carry a private Book";
`librarian` adds who the keeper is).

## The user prompt, section by section

```
== STATUS == energy 71/100 | position (12, 12) | standing on grass |
             inventory: wood x2 (1/8 slots) | tools: none
```
Engine-owned ground truth about the body. Slots matter: 8 slots × 5 units
each, so hoarding has a cost and the basket is worth discovering.

```
== WHY NOW == You are planning because of the routine 15-tick check-in;
              your unfinished plan (1 step left) was set aside — re-issue
              what you still want.
```
Why the planner is running *now*. The variants: first decision, previous
plan finished, a step failed (see RECENT EVENTS for the reason), the timer
(with how many steps were set aside), someone spoke to you, or your
survival reflex took over. Added in v0.39: without it, a model could not
distinguish "my plan completed" from "my plan was discarded halfway",
and timer interrupts silently killed multi-step experiments.

```
== YOU KNOW == axe = 1 stone + 2 wood -> tool: axe
```
Confirmed recipes only — the engine writes this; the agent cannot. A recipe
enters here by successfully crafting it (or via the knowledge condition:
oracle/fragmented/one_expert seed it at birth). This is the one part of the
mind the agent cannot corrupt.

```
== YOU BELIEVE == Failed combines destroy the ingredients (lost wood+stone
                  at t31); wood+stone 1:1 is not a recipe; ...
== NOTES == A01. berries x2, stone x1, wood x1. t31: combine wood1+stone1
            FAILED, materials destroyed. TODO at Book: answer combine_loss...
```
The mind's own previous output, returned verbatim. Beliefs are a list of
hypotheses (max 12 × 250 chars); notes are one free-form scratchpad (max
1500 chars, rewritten in full each time). The engine never reads either —
they exist purely so the mind can persist itself. In practice the quality
of these two fields *is* the quality of the agent.

```
== NEARBY == berries at (16, 16), (15, 15); wood at (18, 22); stone at
             (20, 15); a book at (16, 16); A00 at (16, 16)
```
Everything within vision radius 6, nearest first (top 3 per resource).
Structures show their builder ("a fire at (10, 5), built by A02") except
world-placed ones like the Book. Coordinates can wrap: an agent at y=4 may
see wood "at (25, 31)" — the system prompt's torus sentence exists so this
reads as *nearby across the edge*, not across the map.

```
== BOOK ==
- [recipe_axe] What is the exact recipe to create axe ...?
    answered by A00 at t47: "axe = 2 wood + 1 stone, no special conditions..."
- [regrow_wood] Do trees grow back after being harvested? After how long?
    (unanswered)
Notes (permanent, cannot be edited):
  [t24] A01: "wood tiles exhaust after ONE gather..."
You may write: book_answer overwrites a question's answer, book_note adds
a permanent note.
```
Only in book conditions. Out of reach of a central book it collapses to a
pointer ("The Book stands at (16, 16). Stand on or next to it to read it").
In reach, the agent sees every question with its current answer
(author + tick; answers are overwritable) and the latest permanent notes.
`book_answer`/`book_note` appear in ACTIONS only when actually writable —
a librarian-condition reader is told why not. The Book is the society's
externalized memory: unlike notes, it survives its author's death.

```
== RECENT EVENTS ==
[t52] You gathered 1 wood at (18, 22).
[t52] The wood at (18, 22) is exhausted.
[t55] You said to A01: "the axe recipe is 2 wood + 1 stone..."
[t55] (A01 was too far away to hear you.)
[t56] Your plan stopped: give: A01 is too far away.
[t58] You ate berries (+25 energy).
[t61] A00 said to you: "A01: the axe recipe is 2 wood + 1 stone anywhere..."
```
The journal: last 12 lines of everything this agent did, perceived (within
vision radius), or was told. This is the only window onto the world's
dynamics — regrowth ("New berries appeared at (13, 13)"), spoilage ("1 of
your berries rotted away"), other agents' actions ("A00 updated the book's
answer to 'recipe_axe'"), and the all-important experiment feedback ("Your
experiment failed. Nothing happened. The materials crumbled uselessly.").
The earshot line in parentheses (v0.39) tells a speaker their directed
message went unheard; without it, agents believed coordination had
happened when it hadn't.

```
== ACTIONS == move_to | gather | eat | combine | say | give | drop | wait
              [| book_answer | book_note]
```
Then the format contract: plan up to 20 steps ("you will not be consulted
again until the plan finishes, a step fails, or many ticks pass"), respond
with one JSON object. The example plan's combine is deliberately **not** a
real recipe — models copy examples verbatim, and until v0.39 the example
was literally the axe (`wood 2 + stone 1`), handing every model a free
"discovery". `tests/test_prompt_fixes.py` now asserts the example can never
resolve against the techtree.

## How a competent agent reacts — a played timeline

From the seed-42 session (the numbers are real):

| Tick | Agent | What happened |
|---|---|---|
| t0 | both | First decision. No food in A01's view → food becomes the plan's anchor. A00 sees wood at y=31 from y=4 and, knowing the world wraps, treats it as close. |
| t15 | both | Timer replan. WHY NOW says steps were set aside; both re-issue from notes. A00's journal showed "New berries appeared" → first regrowth hypothesis. |
| t22 | A00 | combine(1 wood + 1 stone) → "materials crumbled uselessly"; inventory confirms both gone. Belief formed: *failures destroy ingredients*. |
| t22–24 | A01 | Reaches the Book, reads 13 open questions, writes `regrow_berries` and `time_energy` answers from its own evidence, leaves a permanent note. |
| t30 | A00 | combine(2 wood + 1 stone) → "You created an AXE!" → enters YOU KNOW. |
| t33 | A01 | Warns A00 about the 1:1 failure from 12 tiles away → "(A00 was too far away to hear you.)" → belief: *voice carries only a few tiles*. |
| t44 | A01 | combine(3 wood) with only 1 wood → plan stops, "you lack those materials"; next prompt's WHY NOW says a step failed. Lesson: lacking materials refuses, it doesn't destroy. |
| t47–48 | A00 | At the Book: answers `recipe_axe`, notes the failure costs. The society now knows the axe even if A00 dies. |
| t55–56 | A00 | Says + gives to A01 too early — both fail with explicit feedback. Plans around it: wait until adjacent. |
| t58 | A01 | Energy dips below 30 → reflex force-eats, dropping the Book plan. Next WHY NOW: "your survival reflex took over". Belief: *keep food above 30 to stay in control*. |
| t61 | A00→A01 | Adjacent say lands; A01 is interrupted ("someone spoke to you"), reads the recipe from speech. |
| t62 | A01 | Crafts the axe → event logged as `recipe_learned source=taught` — the engine distinguishes taught from discovered without ever reading a mind. |

The pattern to notice: every durable piece of understanding moved through
four distinct stores, each with different persistence and reach —
**journal** (automatic, 12 lines, private) → **beliefs/notes** (deliberate,
private, lost on death) → **speech** (reaches ~3 tiles, once) → **Book**
(permanent, society-wide, requires the trip). The experiments in
`ti.py run --book ...` vary exactly which of these stores exist.

## What the agent never sees

Ground truth (`ground_truth.json`), the techtree, other agents' minds,
the survey judge's verdicts, anything beyond vision radius 6, and any
numeric world parameter (regrowth probabilities, spoilage rates, terrain
fractions). Everything it "knows" beyond the system prompt's basics must
come through the journal, and survive through its own writing.

## Known sharp edges (deliberate)

- Failure feedback is uninformative by design (`failure_verbosity: low`):
  "Nothing happened" — agents must vary one factor at a time to learn.
- The fire/pickaxe conflict (same ingredients, different tile) makes the
  question "under what special conditions?" genuinely hard.
- Speech can lie; the engine never validates content. The Book's answers
  can be wrong; only the survey judge (out-of-band) compares them to truth.
- The replan timer means long plans are *always* at risk of being set
  aside; agents that front-load important steps do better.
