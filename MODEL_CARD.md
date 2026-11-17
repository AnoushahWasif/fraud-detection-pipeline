# Model card: fraud-scoring

## Model details

- **What it is:** gradient boosting (scikit-learn `HistGradientBoostingClassifier`, class-weighted, 300 rounds, 15 leaves, learning rate 0.05) behind a `StandardScaler`, in an `imblearn` pipeline.
- **Output:** a score from 0 to 1 per transaction and an alert flag at threshold 0.8126. The threshold was chosen to minimise expected cost on validation.
- **Version:** identified by the SHA-256 of the saved model (the API returns its first 12 characters). Every saved model carries:
  - the data and config hashes;
  - the git commit, and whether the tree had uncommitted changes;
  - library versions, seed and features.
- **Loading:** only from hash-checked bytes. A different scikit-learn version is refused.

## Intended use

- Ranking card transactions for **human review** in a queue with fixed capacity, about 1 review per 1,000 transactions.
- A portfolio demonstration of a traceable, leakage-tested ML pipeline.

## Out of scope

- **Blocking payments or customers automatically.** The service only scores. Around 30% of frauds are missed, and every alert needs a person.
- **Any data other than this 2013 dataset.** The features are PCA components specific to it, so the model can't score real transactions from elsewhere.
- Decisions about individuals of any kind.

## Data

OpenML dataset 1597 version 1 (`creditcard`): 284,807 card transactions by European cardholders over two days in September 2013, 492 of them fraud (0.17%). It was collected by Worldline and the ULB Machine Learning Group (Dal Pozzolo et al., 2015). OpenML lists it as public, and the original Kaggle copy is listed under the Open Data Commons DbCL 1.0. It's downloaded by a script, pinned by SHA-256, and never committed or re-hosted.

| Split (by time) | Transactions | Frauds | Hours |
|---|---|---|---|
| Train | 199,364 | 384 | 0 to 36.9 |
| Validation | 42,722 | 56 | 36.9 to 42.0 |
| Test (scored once) | 42,721 | 52 | 42.0 to 48.0 |

## Cost model

- A missed fraud costs its amount.
- A false alarm costs a €5 review (configurable).
- Costs are reported per 1,000 transactions.

## Results

| | Validation | Test (once) |
|---|---|---|
| PR-AUC | 0.847 | 0.730 (95% CI 0.605 to 0.838) |
| Recall at 1 alert per 1,000 | 73% | 71% (57% to 80%) |
| Frauds caught at the threshold | 43 of 56 | 37 of 52 |
| Alerts per 1,000 | 1.05 | 1.10 |
| Expected cost per 1,000 | €9.18 | €63.32 (€14.87 to €134.68) |
| Cost with no model | €196.57 | €144.40 |

**On validation the model cut expected losses by 95%; on test by 56%.** The ranking held roughly, with recall about the same. The money didn't: four large frauds (€187, €358, €634 and €1,097) scored below the threshold in the test period. With about 50 frauds per split, a handful of large ones decides the cost, which is why the test interval runs from €15 to €135.

## Limitations

- **Two days of data from 2013.** Fraud patterns change, and the test split covers one evening.
- **Few frauds to learn from or check against:** 384 in training, about 50 each in validation and test. Every figure above has wide uncertainty.
- **The threshold is tuned to validation's frauds.** A different mix of fraud sizes moves the best threshold.
- **Anonymised features.** The model can't be explained in terms of real transaction fields.
- **The cost model leaves out what a false alarm costs the customer.** It only counts the review cost.

## Ethical considerations

A false alarm delays a real customer's payment, and a missed fraud costs money and trust. The service is built so that a person decides: it returns a score and an alert, never a block. Its logs and error responses contain no transaction values.

## How to reproduce

```bash
uv sync --extra tracking
make all              # data -> baseline -> comparison -> final model (about 7 minutes)
uv run python -m fraud_pipeline.holdout   # refuses to run while reports/holdout.json exists
```
