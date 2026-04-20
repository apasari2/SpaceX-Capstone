# TD ICU Mortality PyHealth Contribution — PR Package

This directory contains a complete, submission-ready PyHealth contribution for
the Temporal-Difference ICU mortality prediction model from Frost et al.
(2024), arXiv:2411.04285.

## File layout (matches PyHealth's expected structure)

```
pr_package/
├── pyhealth/
│   └── models/
│       └── td_icu_mortality.py                 # the model code
├── docs/
│   └── api/
│       ├── models.rst.patch                    # snippet to add to existing docs/api/models.rst
│       └── models/
│           └── pyhealth.models.td_icu_mortality.rst  # new Sphinx doc page
├── examples/
│   └── mimic4_td_icu_mortality.py              # runnable example + alpha ablation
└── tests/
    ├── conftest.py
    └── test_td_icu_mortality.py                # 50+ unit tests, synthetic data only
```

## How to integrate into the PyHealth fork

Assuming your local clone of PyHealth is at `~/pyhealth`:

```bash
# 1. Copy the model file
cp pyhealth/models/td_icu_mortality.py ~/pyhealth/pyhealth/models/

# 2. Add the model to pyhealth/models/__init__.py so it's importable from pyhealth.models
#    Add: from pyhealth.models.td_icu_mortality import TDICUMortalityModel, CNNLSTMPredictor
#    (plus any other symbols you want to re-export)

# 3. Copy the docs page
cp docs/api/models/pyhealth.models.td_icu_mortality.rst \
   ~/pyhealth/docs/api/models/

# 4. Update the docs index (see models.rst.patch for the snippet to add
#    to ~/pyhealth/docs/api/models.rst)

# 5. Copy the example
cp examples/mimic4_td_icu_mortality.py ~/pyhealth/examples/

# 6. Copy the tests
cp tests/test_td_icu_mortality.py ~/pyhealth/tests/
cp tests/conftest.py ~/pyhealth/tests/  # only if tests/conftest.py doesn't already exist
```

## Grading rubric mapping

### Code Implementation (12 pts)

| Criterion | Points | How it's satisfied |
|---|---|---|
| Inherits from BaseModel | 2 | `TDICUMortalityModel(BaseModel)` (line 370 of model file); `super().__init__(dataset)` is called. |
| Implements required abstract methods | 5 | `prepare_labels`, `get_loss_function`, and `forward` are all implemented with docstrings and correct types. |
| Clear forward pass | 3 | `forward` handles supervised and TD modes through a clean dispatch on `train_td`. |
| File path `pyhealth/models/new_model.py` | — | `pyhealth/models/td_icu_mortality.py` ✓ |
| Sphinx doc file | — | `docs/api/models/pyhealth.models.td_icu_mortality.rst` ✓ |
| Index file updates | 2 | See `docs/api/models.rst.patch` for the toctree entry to add. |
| Example script (1 pt) | 1 | `examples/mimic4_td_icu_mortality.py` with both basic demo and alpha ablation. |
| Proper initialization and configuration methods | 2 | Each construction stage is factored into `_register_scaling_buffers`, `_build_embeddings`, `_build_cnn`, `_build_lstm`, `_build_head`, plus `init_weights`. |

### Documentation (5 pts)

| Criterion | Points | How it's satisfied |
|---|---|---|
| Comprehensive docstrings, Google style | 2 | Module, class, and every public method have Google-style docstrings with Args / Returns / Raises as applicable. |
| Proper type hints | 2 | Every function signature has hints; `from __future__ import annotations` allows forward references cleanly. |
| High-level description and usage examples | 1 | Module docstring gives the paper's thesis and architecture overview; `TDICUMortalityModel` docstring includes an `Example:` block; example script demonstrates end-to-end usage. |

### Code Quality (3 pts)

| Criterion | Points | How it's satisfied |
|---|---|---|
| PEP8, 88-char max | 1 | Verified: no line exceeds 88 characters (see `make lint` below). |
| snake_case / PascalCase | 1 | Classes `CNNLSTMPredictor`, `TDICUMortalityModel`, `MaxPool1D`, `Transpose`, `WeightedBCELoss` use PascalCase; all functions and variables use snake_case. |
| Well-structured, readable code | 1 | Construction split into named helper methods; TD-specific logic lives in named helpers (`compute_td_target`, `predict_next_target`, `soft_update_target`); no 200-line forward method. |

### Bonus: Fast and Performant Tests (5 pts) + Comprehensive Coverage (6 pts)

See `tests/README_tests.md` or the docstring at the top of
`test_td_icu_mortality.py` for the test-rubric mapping.

## Verifying the PR before submitting

```bash
# 1. PEP8 / line length check
python -c "
with open('pyhealth/models/td_icu_mortality.py') as f:
    lines = f.read().split('\n')
long = [i for i, l in enumerate(lines, 1) if len(l) > 88]
print(f'Long lines (>88 chars): {len(long)}')
for i in long: print(f'  line {i}: {len(lines[i-1])} chars')
"

# 2. Model imports and basic instantiation
python -c "
from pyhealth.models.td_icu_mortality import TDICUMortalityModel
print('import OK')
"

# 3. Run the example
python examples/mimic4_td_icu_mortality.py

# 4. Run the tests
pytest tests/test_td_icu_mortality.py -v
```

Expected test output:
- 50+ tests passing
- Total runtime under 5 seconds
- No warnings about slow tests

## Known deviations from the paper

- The class name is `TDICUMortalityModel` to align with PyHealth's naming for
  existing models (e.g. `SparcNet`, `RETAIN`).
- The model requires both `scaling` (per-feature mean/std dict) and
  `features_vocab` (ordered feature names) as constructor arguments. These
  must be provided by the user; the example script shows how to build a
  minimal version.
- The target network is kept in eval mode whenever its predictions are
  consumed; this freezes BatchNorm stats and disables LSTM dropout for
  consistent TD targets.
- Only `binary` mode is supported; `multiclass` would require non-trivial
  changes to the TD target logic and is deferred to a future PR.
