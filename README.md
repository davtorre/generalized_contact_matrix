# Generalized contact matrices

A small Python module for **generalized contact matrices**: matrices whose element
$R_{ab}$ is the total number of contacts between a demographic group $a$ (participants)
and a group $b$ (contacts), where a group is any combination of features (age, gender,
country, ...). Participant and contact groups can use different features, so the matrix
can be rectangular.

## Layout

```
generalized_contact_matrix/
├── pyproject.toml         # makes src/gcm installable: pip install -e .
├── src/gcm/
│   ├── __init__.py        # from gcm import Domain, Counts, Element, ContactMatrix, heatmap
│   ├── matrix.py          # Domain, Counts, Element, ContactMatrix
│   └── plot.py            # heatmap(): interactive HTML heatmap of a ContactMatrix
├── examples/              # 01_domains_and_counts, 02_building_a_matrix, 03_polymod
├── scripts/
│   └── make_dataset.py    # POLYMOD raw files -> data/processed/polymod_contact_matrix.csv
├── tests/
│   ├── test_matrix.py     # Counts, from_numpy / to_numpy round trips
│   └── test_examples.py   # runs every example
└── data/
    ├── raw/polymod/       # POLYMOD survey files, as downloaded
    └── processed/         # datasets in the format read by ContactMatrix.from_dataframe
```

## Examples

Three short examples in `examples/`, each one idea:

| Example | Shows |
|---|---|
| `01_domains_and_counts.py` | `Domain`, products, indices, fixed domains (catching typos), `Counts` |
| `02_building_a_matrix.py` | the three ways to build a `ContactMatrix` and the two ways out; contacts per participant; `None` for features added later |
| `03_polymod.py` | the real POLYMOD data: mean contacts by country, interactive heatmaps (needs `data/processed/`, see below) |

They are plain scripts (`python examples/01_domains_and_counts.py`) written in the "percent"
format: the `# %%` lines mark cells, so VS Code or Jupyter (with jupytext) open them as
notebooks. Plots are written to `examples/output/`.

## Usage

Install the package in your environment (Python ≥ 3.10; `numpy` and `polars` are installed
if missing). `-e` makes it *editable*: changes to the code here are seen immediately, with no
reinstall. From this folder:

```
pip install -e .
```

or, from another project's environment, `pip install -e /path/to/generalized_contact_matrix`.

```python
import polars as pl
from gcm import ContactMatrix, heatmap

cm = ContactMatrix().from_dataframe(pl.read_csv("data/processed/polymod_contact_matrix.csv"))
R, domain_part, domain_cont = cm.to_numpy()     # (240, 30) matrix + the Domains of its rows/columns
heatmap(cm, "polymod.html", sort_part=["country", "gender", "age"], open_browser=True)
heatmap(cm, "polymod_by_country.html", block_level="country")   # one panel per country
df = cm.to_dataframe()                          # back to the dataset format below
```

### Domains

A `Domain` has `domain_names` (e.g. `['age', 'gender']`) and the ordered `elements` (groups,
as tuples such as `(0, 'F')`) they take; it fixes
the row/column order of the matrix. Domains are collected from the data by default, or can
be given up front, e.g. to get a complete grid or to reject typos with `fixed=True`:

```python
from gcm import Domain

groups = Domain("age", range(0, 75, 5), fixed=True) * Domain("gender", ["F", "M"], fixed=True)
cm = ContactMatrix(domain_part=groups, domain_cont=groups)   # 30 x 30, zeros where no data
cm.domain_part.index((5, "F"))                               # row of the 5-9 year old women
```

### Counts

`Counts` holds one number per group of a `Domain`: the vector $\underline N$ (participants per
group, a census population, ...). `None` means unknown, not zero.

```python
from gcm import Counts

groups = Domain("age", ["0-4", "5-9"]) * Domain("eth", ["hispanic", "white"]) * Domain("gender", ["F", "M"])
N = Counts(groups, [120, 95, 80, 101, 60, 75, 82, 90])      # one value per group, in domain order
N[("5-9", "hispanic", "F")]                                 # 60
N.add(("5-9", "hispanic", "F"), 3)                          # increment; N[...] = value sets
values, domain = N.to_numpy()
```

A `ContactMatrix` keeps its own as `cm.N_part` (rows) and `cm.N_cont` (columns); they live on
the matrix's domains and follow them when `add_element` adds groups or features.

### Building a matrix

| You have | Use |
|---|---|
| domains + a matrix of numbers | `ContactMatrix.from_numpy(R, domain_part, domain_cont, N_part=..., N_cont=...)` |
| a long table (`part_*`, `cnt_*`, `contacts`, ...) | `ContactMatrix().from_dataframe(df)` |
| one cell at a time | `cm.add_element(Element({'age': 0}, {'age': 5}, R, N_part, N_cont))` |

`from_numpy` is the inverse of `to_numpy`: row `i` is `domain_part.elements[i]`, column `j`
is `domain_cont.elements[j]`, and `N_part` / `N_cont` follow the same order (or are given as
`Counts`, looked up group by group).

```python
age = Domain("age", [0, 5, 10])
cm = ContactMatrix.from_numpy([[12, 3, 1], [4, 9, 2], [1, 2, 6]], age, age, N_part=[50, 40, 30])
R, rows, cols = cm.to_numpy()                                # the same matrix back, rows[i] = (age,)
```

### Tests

```
python tests/test_matrix.py        # or, for everything: python -m pytest tests
python tests/test_examples.py      # runs every example
```

The POLYMOD round trip is skipped when `data/processed/` has not been built.

### Dataset format

One row per pair (participant group $a$, contact group $b$):

| column | meaning |
|---|---|
| `part_<feature>` | participant features, e.g. `part_age`, `part_gender`, `part_country` |
| `cnt_<feature>` | contact features, e.g. `cnt_age`, `cnt_gender` |
| `contacts` | $R_{ab}$, total contacts reported by participants in $a$ with people in $b$ |
| `N_part` | number of participants in $a$ |
| `N_cnt` | number of recorded contacts in $b$ ($\sum_a R_{ab}$) |

### Rebuilding the POLYMOD dataset

`data/` is not tracked by git, so on a fresh clone download the raw files first
(7 files, about 8 MB, from [Zenodo record 3746312](https://zenodo.org/records/3746312)):

```
mkdir -p data/raw/polymod
for f in contact_common.csv dictionary.xls hh_common.csv hh_extra.csv \
         participant_common.csv participant_extra.csv sday.csv; do
  curl -fL -o "data/raw/polymod/2008_Mossong_POLYMOD_$f" \
       "https://zenodo.org/records/3746312/files/2008_Mossong_POLYMOD_$f?download=1"
done
```

Then build the dataset into `data/processed/`:

```
python scripts/make_dataset.py                        # age, gender, country x age, gender
python scripts/make_dataset.py --part-features age,daytype --cnt-features age --age-breaks 0,5,18,65 --countries IT,DE
```

See `python scripts/make_dataset.py --help` for the options.

## Data

POLYMOD contact survey: J. Mossong et al., *Social contacts and mixing patterns relevant
to the spread of infectious diseases*, PLoS Medicine 5(3): e74 (2008),
doi:[10.1371/journal.pmed.0050074](https://doi.org/10.1371/journal.pmed.0050074).
Files from the socialmixr-format release on Zenodo:
[record 3746312](https://zenodo.org/records/3746312).

## Acknowledgement of AI assistance

Parts of this repository were written or revised with **Claude** (Anthropic), used
through Claude Code. The design of `ContactMatrix` and most of `src/gcm/matrix.py` are
my own; Claude reviewed it repeatedly and pointed out bugs, which I mostly fixed myself.

What Claude wrote directly:

- `src/gcm/matrix.py`: the fixes I asked for in `add_element` (all checks before any
  change, domain extension of existing keys, tuple keys with `None` as the placeholder
  for features added later) and in `from_dataframe` (the `add` return behaviour, the
  missing-column message).
- `src/gcm/matrix.py`: the `Domain` class, rewritten from my first draft (ordered tuple
  elements, Cartesian product, fixed domains, sorting), its use as `domain_part` /
  `domain_cont` in `ContactMatrix`, `to_dataframe`, and `from_numpy` (from my idea of
  building a matrix directly from domains and numbers), and `to_numpy` returning the
  domains (from my idea).
- `src/gcm/matrix.py`: the `Counts` class (from my idea of a vector of counts over a
  `Domain`) and its use for `N_part` / `N_cont` in `ContactMatrix`.
- `tests/`: the tests.
- `examples/`: the three examples.
- `scripts/make_dataset.py`: the whole POLYMOD pipeline.
- `src/gcm/plot.py`: the whole module.
- The current folder layout, the packaging (`pyproject.toml`, `__init__.py`) and this README.

All of it was reviewed and tested by me before use. Commits that include code written by
Claude carry a `Co-Authored-By: Claude` trailer.
