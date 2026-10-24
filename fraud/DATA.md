# PaySim: what the data looks like (balances excluded)

Source: Kaggle `sriharshaeedala/financial-fraud-detection-dataset`, `Synthetic_Financial_datasets_log.csv`
(493,534,783 bytes), licence CC BY-SA 4.0 (Documented: [dataset page](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)). PaySim itself is by
E. A. Lopez-Rojas, A. Elmir and S. Axelsson. Provenance: the results up to 2026-10-05 were computed on a
copy downloaded on 2026-10-03 under PaySim's original file name; it is the same file (byte size
Observed equal; checked identical by the owner on 2026-10-05). Build the cache
with `python -m fraud.data`. Everything below is **Observed** on that file unless labelled
otherwise. The four balance columns are never read (`fraud/CLAUDE.md`).

## Size and fraud by type

| type | rows | fraud | fraud share |
|---|---:|---:|---:|
| CASH_IN | 1,399,284 | 0 | 0 |
| CASH_OUT | 2,237,500 | 4,116 | 0.184% |
| DEBIT | 41,432 | 0 | 0 |
| PAYMENT | 2,151,495 | 0 | 0 |
| TRANSFER | 532,909 | 4,097 | 0.769% |
| **all** | **6,362,620** | **8,213** | **0.129%** |

- **Fraud-free types: CASH_IN, DEBIT, PAYMENT**, in the whole file and in the training steps
  (1–300). These are what the hybrid's rule filter approves.
- PAYMENT always goes to a merchant (`M…`); every other type goes to a customer (`C…`).
- `isFlaggedFraud` is set on 16 rows, all fraud. PaySim's own rule is not used as a signal: 16 of
  8,213 frauds.

## Time

`step` is the hour, 1–743 (31 days). Legitimate volume collapses after step ~400, while fraud
keeps a steady ~11 rows an hour:

| steps | rows | fraud | fraud share |
|---|---:|---:|---:|
| 1–100 | 1,059,516 | 1,190 | 0.112% |
| 101–200 | 1,322,666 | 1,064 | 0.080% |
| 201–300 | 1,691,468 | 1,147 | 0.068% |
| 301–400 | 1,713,380 | 1,076 | 0.063% |
| 401–500 | 274,777 | 1,084 | 0.395% |
| 501–600 | 197,240 | 1,052 | 0.533% |
| 601–700 | 92,152 | 1,110 | 1.205% |
| 701–743 | 11,421 | 490 | 4.290% |

**Implementation choice:** train = steps 1–300, test = steps 301–743. The test period's natural
fraud rate is higher and changes a lot over time. That is why each window reports its cost both
as sampled and reweighted to the natural rate of its own time span.

## Accounts

- **Origin accounts almost never repeat:** 6,353,307 distinct senders; 0.15% send more than once.
  A per-account tier would have nothing to work on (the PRD non-goal holds).
- Receivers repeat more: 2,722,362 distinct; 16.9% receive more than once.
- Only 1,769 ids appear as both a sender and a receiver.

## The fraud pattern

- A fraud TRANSFER is almost always followed **in the next row** by a fraud CASH_OUT of **the same
  amount** in the same hour: 4,075 of 4,097 fraud transfers.
- The CASH_OUT's sender is **never** the TRANSFER's receiver (0 of 4,075). The ids do not link the
  two halves; only the amount and the hour do. **Inferred:** a PaySim generator artifact (fresh
  ids for each side), not how real mule accounts look.
- Fraud amounts are capped at 10,000,000 per transaction (287 fraud rows sit exactly at the cap;
  2,920 legit rows also do).

## Signals by class (whole file; signals use earlier rows only)

Share of rows where the signal is above zero:

| signal | TRANSFER fraud | TRANSFER legit | CASH_OUT fraud | CASH_OUT legit |
|---|---:|---:|---:|---:|
| receiver received before (`dest_in_before`) | 0.1% | 88.5% | 73.7% | 85.4% |
| receiver got a TRANSFER before | 0.0% | 56.0% | 30.9% | 43.7% |
| same amount earlier this hour | 1.5% | 0.6% | **99.1%** | **0.0%** |
| round amount (multiple of 1,000) | 3.5% | 0.6% | 3.8% | 0.0% |
| sender sent before | 0.2% | 0.1% | 0.3% | 0.1% |

Amount quantiles (q10 / q50 / q90): fraud TRANSFER 38k / 446k / 4.57M; legit TRANSFER 87k / 487k /
1.77M; fraud CASH_OUT 37k / 436k / 4.45M; legit CASH_OUT 30k / 147k / 354k (q99 573k).

**Inferred:** two signals nearly separate fraud. A TRANSFER to a receiver that has never received
money is fraud 4.2% of the time in training, and almost every fraud TRANSFER is one. A CASH_OUT of
an amount already moved this hour is fraud 69% of the time in training and catches 99% of fraud
cash-outs. Hand-written rules can use both. The model has to read them from the text to keep up,
which makes the "beat rules-only" bar hard.

## Rule cells on the training steps (1–300)

Cost columns: "approve" = the fraud amount let through if every row in the cell is approved;
"decline" = 10% of the legit amount if every row is declined.

| cell | rows | fraud | fraud share | approve | decline |
|---|---:|---:|---:|---:|---:|
| CASH_OUT, same amount this hour | 2,423 | 1,682 | 69.4% | 2,267M | 5.9M |
| CASH_OUT, no same amount | 1,442,708 | 28 | 0.002% | 5.5M | 25,223M |
| TRANSFER to a new receiver | 40,129 | 1,686 | 4.2% | 2,275M | 2,524M |
| TRANSFER to a known receiver | 295,284 | 5 | 0.002% | 4.7M | 20,731M |
| TRANSFER, round amount | 39 | 37 | 94.9% | 370M | 0.1M |
| TRANSFER, same amount this hour | 67 | 16 | 23.9% | 100.5M | 0.7M |
