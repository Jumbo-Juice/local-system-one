# A primer on the statistics behind the project

For the presenters, to read before questions. The new professor knows econometrics and teaches
scikit-learn and PyTorch. These eight topics are the statistics ideas behind what the deck and
paper show. For each: the idea, a small worked example, where it appears, and how to answer if he
asks.

How to read the numbers. Numbers marked "from the deck" are the project's real numbers (they come
from `slides/data.js`, and the paper uses the same values). Numbers marked "made up" are small
examples written for this primer to show an idea; they are not results. Slide numbers are the
numbers on the slides (03 is the third slide). Paper sections are named as in the paper's headings.

Mental model in one line: a model gives a score, a rule turns the score into an action, and we
judge the action in dollars on windows of payments, with an interval to say how sure we are.

---

## 1. Logistic regression: odds, and what a weight means

**The idea.** Logistic regression adds up weighted facts into one score, then squashes the score
into a probability between 0 and 1. Odds are a way of writing a probability: odds = p / (1 - p). A
probability of 0.5 is odds of 1 (even); 0.2 is odds of 0.25 (one to four). The useful fact: adding
a weight to the score multiplies the odds by e to the power of that weight, whatever the other facts
are. That is the whole meaning of a weight.

**Worked example (from the deck, slide 06, made-up weights).** Start at -7.00. The payment is a
transfer, so add 1.50. The log of the amount is 12.63, times 0.35, adds 4.42. It is not at night and
not a round amount, so those add 0. The score is -1.08. The probability is p = 1 / (1 + e^1.08) =
0.25. The odds are 0.25 / 0.75, about 0.34. Now the weight 1.50 on "is a transfer": it multiplies
the odds by e^1.50 = 4.48. A transfer has about 4.5 times the odds of fraud of a non-transfer with
the same other facts. It does not multiply the probability by 4.5: with these made-up numbers, odds
0.34 become 0.34 x 4.48 = 1.52, which is a probability of 0.60, not 0.25 x 4.5.

Sign and size: a positive weight raises the odds of fraud, a negative one lowers them, zero does
nothing. Weights are only comparable when the facts are on comparable scales (see topic 3).

**Where it appears.**
- Deck: slide 06 (the table of facts, the sigmoid, and the odds box).
- Paper: Methods, "Logistic regression".
- It is also the baseline every other method is compared with (slide 05 to slide 11; paper:
  Methods, "Logistic regression"; Results, "The final test").

**If he asks.**
- "Is this the logit model?" Yes, the same idea: a linear score through a sigmoid, fitted by
  maximum likelihood. We use it as a baseline, not as the point of the project.
- "Why not read the weights as effects?" The weights here are made up on the slide, and on real
  data the facts are correlated, so a weight shows what the model uses, not what causes fraud.

---

## 2. Softmax as the multinomial logit

**The idea.** Softmax turns several raw scores (logits) into probabilities that add up to one:
take e to the power of each score, divide each by the sum of them all. Logistic regression is the
two-answer case. With three answers it is the multinomial logit. Adding the same number to every
logit changes nothing, so a logit only matters relative to the others.

**Worked example (from the deck, slides 08 and 09: the real PaySim transfer of 305,369.89 USD to a
customer, untrained Qwen2.5-1.5B).** The model gives one score to each of three letters. Shifted so
the top is 0, the logits for A (approve), B (review), C (decline) are -3.41, -3.41 and 0.00. Take e
to each: 0.03, 0.03 and 1.00; the sum is 1.07. Divide: A = 0.03, B = 0.03, C = 0.94. So the model
says decline with 94%. The shift does not matter: the slide's logits are the model's log-probabilities moved so the top
is 0, and softmax ignores adding one constant to all three scores (the deck says so).

Then slide 09 asks the same question with the options in reverse order, and the model puts 0.54 on
approve, 0.05 on review and 0.42 on decline: it follows the letter's position (the last letter
both times), not its meaning. Averaging the two orders gives approve 0.28, review 0.04, decline
0.68.

**Where it appears.**
- Deck: slide 08 (logits and softmax), slide 09 (the two orders and the average).
- Paper: Methods, "The language model as a decision model" and "From probabilities to an action".
  The paper may use "softmax (multinomial logistic)" once.

**If he asks.**
- "Where is the intercept?" There is none: the language model's three letter scores are the
  logits directly. Only differences between them matter.
- "Why average two orders?" The probabilities depend on which letter holds which answer, so we
  cancel the position effect by asking both ways (slide 14; paper: Biases and Limitations,
  "Option-order bias").

---

## 3. Standardisation and normalisation: scaling changes what a weight seems to say

**The idea.** A weight is "change in score per one unit of the fact". Change the unit and the
weight changes, with no change in the model. Standardising a fact means subtracting its mean and
dividing by its standard deviation, so the new unit is "one standard deviation". Normalising
usually means squeezing a fact into a fixed range such as 0 to 1. Both are only a change of
units.

**Worked example: two identical graphs (made up).** One fact: x = the number of payments the
customer made in the last day, with values 0, 2, 4, 6, 8. Its mean is 4 and its standard deviation
is 2.83.

- Model on the raw fact: score = -3 + 0.5 x.
- Standardise first: z = (x - 4) / 2.83. The same model is now score = -1 + 1.41 z.

| x (raw) | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| z (standardised) | -1.41 | -0.71 | 0.00 | 0.71 | 1.41 |
| score, raw model | -3.0 | -2.0 | -1.0 | 0.0 | 1.0 |
| score, standardised model | -3.0 | -2.0 | -1.0 | 0.0 | 1.0 |

Plot score against x, and score against z: two straight lines, the same shape, the same points,
the same fit, the same predictions. The only differences are the labels on the horizontal axis and
the weight: 0.5 in one, 1.41 in the other. Read naively, the second says the fact is nearly three
times as important. It is not: 0.5 is per payment, 1.41 is per standard deviation of payments, and
2.83 payments make one standard deviation (0.5 x 2.83 = 1.41). On the odds scale: one more payment
multiplies the odds by e^0.5 = 1.65; one standard deviation more multiplies them by e^1.41 = 4.1.
Same model, two ways of saying it.

Why it matters beyond reading: when several facts are on different scales (an amount in dollars
next to a yes or no), the raw weights cannot be compared, so people standardise first. Standardised
weights are comparable in the sense of "per typical spread". Also, a penalty on weights (the
default in scikit-learn's logistic regression) punishes a big weight, so it treats a fact differently
depending on its units; standardising makes the penalty fair. And fitting is faster and steadier.

**Where it appears.**
- Deck: not as a visible idea. Slide 06's speaker notes say the real PaySim logistic regression uses
  standardised facts, and that the weights on the slide are made up.
- Paper: Methods, "Logistic regression" (the weights are read through the odds).
- Where it would come up if he asks: any weight table, ours or his.

**If he asks.**
- "Are your weights standardised?" The real models standardise the facts before fitting; the weights
  on slide 06 are made up and were not fitted, only to show the arithmetic.
- "Does standardising change the predictions?" No. It changes the weights' meaning, not the fit.
- "Normalising versus standardising?" Normalising squeezes into a range; standardising centres
  and divides by the spread. For a linear model either gives the same predictions, with different-looking weights.

---

## 4. Expected loss: deciding by cost, not accuracy

**The idea.** Accuracy counts every mistake as one mistake. But missing a fraud of 300,000 dollars
and declining a 40-dollar customer are not the same size of mistake. So we put a price on each
action and choose the one with the lowest expected cost: probability times what it costs if wrong.
This is expected loss, as in decision theory, and the "price" is the loss function.

**Worked example 1: accuracy (made up).** 1,000 payments, 10 of them fraud. A model that approves
everything is right on 990, 99% accurate, and it loses every fraud. Accuracy rewards doing nothing.

**Worked example 2: the real figures (from the deck, slide 03).** On the final test, approving
every payment is 97% accurate (fraud is 630 of 20,000 payments) and it costs 5,233 USD per window,
the most of any setup. Logistic regression with the rule costs 2,106 USD per window.

**Worked example 3: the rule (from the deck, slide 05; the score is made up).** The cost of each
action for one payment, given a score p:
approve = p x amount; decline = (1 - p) x friction; review = fee + lambda.
For the PaySim transfer of 305,369.89 USD with p of about 0.2535 (the made-up score from slide 06, shown rounded as 0.25) and lambda = 10 (the deck's illustration value; so a review costs the 10 USD fee plus 10 = 20):
approve costs 77,427.42, decline 22,794.25, review 20.00. The cheapest is review. For a 40 USD
payment with the same p: approve 10.14, decline 2.99, review 20.00. The cheapest is decline. Same
score, different action, because the amount decides. Two fixed cut-offs on p cannot do that.

The prices are assumptions: friction is 10% of the amount at least 1 USD, and a review costs
10 USD (at most 5% of payments).

**Where it appears.**
- Deck: slide 03 (accuracy against cost and the cost formula), slide 05 (the rule), slide 14
  (the prices are assumed).
- Paper: Data and Protocol, "Judging by dollars"; Methods, "The cost rule and the customer line";
  Biases and Limitations, "Assumed costs".

**If he asks.**
- "Where do the prices come from?" We chose them; they are not measured. Change them and the
  verdicts can move (slide 14: a flat price per customer declined makes the rule's saving
  not distinguishable from 30 USD upward). That is the main limitation, and we say so.
- "Is this a proper loss function?" It is a hand-priced loss, and a decision rule on top of
  predicted probabilities: Bayes decision with unequal costs.

---

## 5. Confidence intervals and the bootstrap

**The idea.** A mean from a sample is not the true mean; it would be a bit different on another
sample. An interval says how far it could plausibly move. The bootstrap gets one without a formula:
resample the data you have, with replacement, many times, take the mean of each resample, and keep
the middle 95% of those means. Here the data resampled are windows, not payments, because
payments next to each other in time are not independent and one large fraud can dominate a window.

**Worked example (made up).** Six windows, cost difference between two setups, in dollars:
-40, -10, -30, 20, -50, -5. Mean = -19.2. Draw six windows with replacement, for example
-10, -10, 20, -50, -40, -10: the mean is -16.7. Repeat 10,000 times (the project uses 10,000
resamples). The 2.5th and 97.5th percentiles of the 10,000 means are the interval. The interval is
wide when windows disagree and narrow when they agree.

**Worked example (from the deck, slide 11).** Final test, 20 windows. Primary comparison: the language
model reading the tree score minus logistic regression, both with the rule: -163 USD per window,
95% interval -462 to 238. The interval contains 0, so we call the two "not distinguishable". We
do not say "no difference"; we say the data cannot tell. The mean looks sizeable, the interval says it is not safe.

**Where it appears.**
- Deck: slide 05 ("distinguishable" defined), slide 11 (the interval bars), slide 12.
- Paper: Data and Protocol, "Paired differences and intervals"; Results, "The final test".

**If he asks.**
- "Why not a t-test?" The paper uses the bootstrap (resampling windows, 10,000 times) and does not
  report a t-test. Our own reading (Inferred): window costs are skewed by a few large frauds and
  there are only 20 windows on the final test, so a formula that assumes a bell shape is
  harder to trust. Do not claim we compared the two.
- "Why 'distinguishable', not 'significant'?" We chose a plain word on purpose: the interval stays
  on one side of zero or it does not. No p-values appear.
- "Is a 95% interval a 95% chance the truth is inside?" Strictly, no: it is the range that would
  contain the truth in 95% of repeated samples. In plain words: how far this number could move.

---

## 6. Paired differences

**The idea.** When two methods are tested on the same cases, compare them case by case, then
average the differences. Shared ups and downs (a hard window hurts both) cancel in the
difference, so the interval is much tighter than comparing the two averages separately.

**Worked example (made up).** Cost per window, three windows:

| window | method A | method B | A minus B |
|---|---|---|---|
| 1 | 1,000 | 1,010 | -10 |
| 2 | 3,000 | 3,020 | -20 |
| 3 | 500 | 505 | -5 |

Looked at separately, A costs 1,000 to 3,000 and B 1,010 to 3,020: the spreads swamp the gap. In
pairs, A is cheaper in all three windows by a small, steady amount: mean -11.7, a clear
and consistent saving. Pairing used the fact that both methods see the same windows.

**Where it appears.**
- Deck: slide 03 (windows), slide 05 (the rule cost less in 18 of 20 earlier test windows, a paired count), slide 11
  (every row is "cost minus logistic regression's", both deciding the same 20 windows).
- Paper: Data and Protocol, "Split by time, windows, dev and the final test" and "Paired
  differences and intervals".

**If he asks.**
- "Why compare in windows?" Every method decides the same windows, so the comparison is paired
  at window level, and the window is the unit for the bootstrap.
- "Why the order, first minus second?" So the sign is always the same: negative means the first is cheaper.

---

## 7. Selection bias and specification search: why dev flatters, why we pre-registered

**The idea.** If you try many options and keep the one that looks best on some data, that data
flatters it: part of its edge is luck, and the luck will not come back on new data. The more
options tried, the bigger the flattery. This is selection bias (in econometrics, specification
search). Two defences: keep a final test nobody tuned on, and write down what you will test before you
see its results (pre-registration).

**Worked example (made up).** Ten coins, all fair. Flip each ten times, pick the one with the most
heads: perhaps 8 heads out of 10. Claim "this coin lands heads 80% of the time"; flip it ten more
times and you get about 5. The best of ten was chosen on luck. Dev is the first ten flips; the final
test is the second.

**Worked example (from the deck, slide 12).** Boosted trees minus logistic regression, both with the rule.
On dev (88 windows): -295 USD per window, interval -411 to -173, distinguishable: it looked like a
clear win. On new windows (10): -97 USD per window, interval -385 to 182, not distinguishable. Same
models, same rule. Why: the rule's review price was picked on the trees' dev scores, and dev rows
sit close to the rows the trees learned from.

How the final test is protected: its 20 windows were reserved before this work began, and the
setups, comparisons and the rule for choosing a winner were written down and committed before any
setup scored a final-test row; each setup was scored once. Anything asked after seeing results is
labelled a "read-out", not a test result. This limits the bias, it does not remove it (the paper
says so).

**Where it appears.**
- Deck: slide 12 (two verdicts), slide 04 (fit, dev, test and the final test).
- Paper: Data and Protocol, "Split by time, windows, dev and the final test"; Biases and
  Limitations, "Optimism from choosing on dev".

**If he asks.**
- "Is the final test not tuned on at all?" The cut-offs, lambda, the prompt and which setups went
  forward were all chosen on dev; the final test was scored once. A small procedure difference
  remains: lambda for the language-model setups came from more dev windows than logistic regression's.
- "Why not cross-validation?" Time order matters here: we split by time and never shuffle, so
  no model learns from the future, which random folds would allow.
- "How do you know you did not peek?" The paper states the setups, comparisons and the rule for choosing a winner were committed before any setup scored a final-test row, and each was scored once. We cannot prove a negative beyond that record.

---

## 8. Class imbalance and base rates

**The idea.** Fraud is rare. When 97 in 100 cases are genuine, a model that always says "genuine"
is 97% right, and an alarm that is right "most of the time" can still be wrong most of the time it
rings. The base rate (the share of fraud) sets what accuracy means and how many false alarms to
expect.

**Worked example (made up).** 1,000 payments, 10 fraud. A detector catches 8 of 10 frauds and wrongly
flags 40 of the 990 genuine ones. Of 48 flags, 8 are real: precision is 17%. It catches 80% of the
fraud, yet five in six flags are false alarms. Good recall with low precision is the normal
situation when fraud is rare.

**Worked example (from the deck, slides 03 and 04).** IEEE-CIS final test: 630 frauds in 20,000 payments
(3.15%), so "approve everything" is 97% accurate and it is also the most expensive setup. PaySim is
rarer still: about one fraud in 775 payments. On the final test, logistic regression with the rule
stopped 384 frauds, declined 1,051 good customers and sent 652 more good customers to review. Counting every flag (declined or reviewed) as an alarm, precision is 384 / (384 + 1,051 + 652) = 0.18.

So the project does not use accuracy. It prices the decision (topic 4) and reports, next to every
cost, how many good customers were declined and how many frauds were stopped.

**Where it appears.**
- Deck: slide 03 (the trivial model), slide 04 (fraud share of each dataset).
- Paper: Data and Protocol, "Two datasets, two cycles" and "Judging by dollars".

**If he asks.**
- "Did you rebalance (oversample, class weights)?" No class weighting or oversampling appears in
  the fraud code (Observed: searched for it). The imbalance is handled in the decision: the rule
  prices each action. One language-model setup did weight each training row by what a mistake on
  it costs; on dev it did not help (paper: Results, "IEEE-CIS on dev"). Do not claim more.
- "Why not AUC?" AUC is reported in the paper's text under Results, "The final test" (not in a table), but it ranks payments and ignores amounts and
  prices, which is why the dollar cost is the headline.
