# Data dictionary

The file `训练基因信息_features.xlsx` contains anonymized industrial DNA synthesis records.

| Column | Description |
|---|---|
| `ID` | Anonymized record identifier. |
| `Sequence` | Target DNA sequence. |
| `Length` | Target sequence length in base pairs. |
| `Duration` | Total synthesis duration in days; used as the regression target / practical proxy for synthesis difficulty. |
| `Vienna Structure` | Predicted single-stranded DNA secondary structure in dot-bracket notation. |
| `MFE (kcal/mol)` | Predicted minimum free energy of the folded structure. |
| `Max Stem Length` | Maximum predicted stem length. |
| `Avg Stem Length` | Mean predicted stem length. |
| `Total Loops` | Number of predicted loop regions. |
| `Max Loop Size` | Maximum predicted loop size. |
| `GU Pairs` | Input-column name used in the source workbook; mapped to `GT_pairs` in the analysis notebook. |
| `Pairing Rate` | Fraction of nucleotides involved in predicted base pairing. |
| `Stem Count` | Number of predicted stems. |

## Notes

- The notebook maps the source column `GU Pairs` to the internal feature name `GT_pairs`.
- Sequence-derived primary features are computed in `model.ipynb`.
- Records are filtered during quality control before final model development.
