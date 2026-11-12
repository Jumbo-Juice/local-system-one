# local-system-one

A small local AI that makes each decision with one quick multiple-choice look, exercised by game
demos and by a fraud app on PaySim and IEEE-CIS, and told to newcomers as one story.

## Language

### The engine

**System One model**:
An LLM used to answer a multiple-choice question in one forward pass, without writing any text.
_Avoid_: Jev (that is TypeSafe's commercial model, which this project does not reproduce)

**Decision**:
One question put to the model: the state, an instruction and a list of lettered options.
_Avoid_: query, prompt

**Option**:
One lettered answer in a decision (A, B, C, …).
_Avoid_: choice, label (the letter is the label; the option is what it stands for)

**One look**:
The single forward pass in which the model reads a decision and scores every option.
_Avoid_: inference step, generation

**Score**:
How strongly the model leans to an option in one look. Scores rank options; they are not
calibrated probabilities.
_Avoid_: probability, confidence

### Fraud app

**Action**:
What the fraud app does with a payment: **approve**, **review** or **decline**.
_Avoid_: "send for a check", "hold" (for review); "block" (for decline)

**Review**:
The action that hands a payment to a fraud analyst, at a fixed fee and within a limited budget per
window.

**Rule filter**:
The hand-written first step that approves payment types which never had fraud, before the model
sees anything.
_Avoid_: pre-filter, gate

**Window**:
A time-ordered block of payments from one period (500 on PaySim, 1,000 on IEEE-CIS), the unit
every setup is tested on.
_Avoid_: batch, test set

**Cost**:
The money a setup loses on a window: fraud approved, genuine customers declined, and review fees.
_Avoid_: loss, penalty, score

**Setup**:
One way of deciding a whole window, such as rules only, logistic regression, or the hybrid of rule
filter and model.
_Avoid_: method, pipeline, configuration

**Comparison**:
Two setups put side by side on the same windows, named in the order the question was
registered: the first setup, then the second.
_Avoid_: "better vs other", "winner vs loser" (the order is the question's, not the result's)

**Paired difference**:
The first setup's cost minus the second's, window by window, averaged over the windows. Negative
means the first costs less; it can be positive.
_Avoid_: "the cheaper minus the dearer", saving (a saving is always positive; this is not)

**Distinguishable**:
Said of a comparison whose 95% window-bootstrap interval of the paired difference excludes 0.
The only word for this result; a comparison that fails it is "not distinguishable".
_Avoid_: significant, "no clear difference", "may cost less"

### The story

**Story**:
The scroll page that puts the whole project's findings, games and fraud together, to one question,
for a newcomer reading alone.
_Avoid_: site, landing page, deck (the deck is the talk's slides), viewer, console

**The question**:
"Can a model that never writes be trusted to decide?" The one question the story asks.

**Trial**:
The question narrowed to one thing ("trusted to be quick?", "trusted with a whole game?"), with its
evidence and one stamp.
_Avoid_: test (the final test and the test windows are eval runs), chapter, experiment

**Stamp**:
The answer a trial ends on: **yes**, **no**, **only with help** or **can't tell**.
_Avoid_: verdict (a comparison's verdict is distinguishable or not), grade, result

**Help**:
What was added around the model or to it before it decided well: structure (tiers of decisions,
option lists, asking in both orders, tournaments, the cost rule) or training.
_Avoid_: scaffolding, tricks, fixes

**Can't tell**:
The stamp for a trial whose comparison is not distinguishable. It is not a "no".
_Avoid_: tie, draw, "no difference"

## Relationships

- The **Story** asks **the question** in several **Trials**; each **Trial** ends on one **Stamp**.
- A **Trial** stamped **only with help** names the **Help** it needed.
- A **Trial** that rests on a **Comparison** that is not **Distinguishable** is stamped **can't tell**.
- A **Window** is decided by one **Setup**; each payment in it gets one **Action**.
- In the hybrid **Setup**, the **Rule filter** decides first; the model decides the rest in one
  **One look** per order of the **Options**.
