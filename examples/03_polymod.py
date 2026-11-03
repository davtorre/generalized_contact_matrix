# %% [markdown]
# # 3. POLYMOD
#
# The POLYMOD survey (Mossong et al. 2008) recorded the contacts of 7,000+ participants in 8
# European countries. Here participants are described by age, gender and country, and their
# contacts by age and gender: a 240 × 30 generalized contact matrix.
#
# This example needs the processed dataset. If `data/processed/polymod_contact_matrix.csv`
# is missing, download the raw files and run `python scripts/make_dataset.py` (see README).

# %%
from pathlib import Path

import numpy as np
import polars as pl

from gcm import ContactMatrix, Domain, heatmap

# the project root, whether this runs from the root or from examples/
ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
CSV = ROOT / "data" / "processed" / "polymod_contact_matrix.csv"
OUT = ROOT / "examples" / "output"
if not CSV.exists():
    raise SystemExit(f"{CSV} not found: build it first (see 'Rebuilding the POLYMOD dataset' in the README)")
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load

# %%
cm = ContactMatrix().from_dataframe(pl.read_csv(CSV))
R, rows, cols = cm.to_numpy()
N_part, _ = cm.N_part.to_numpy()

print(rows)
print(cols)
print("R:", R.shape, "|", R.sum(), "contacts reported by", N_part.sum(), "participants")

# %% [markdown]
# ## Mean contacts per participant, by country
#
# The rows are groups (age, gender, country), so selecting a country means selecting the rows
# whose third entry is that country.

# %%
country = rows.domain_names.index("country")
for c in sorted({g[country] for g in rows}):
    i = [k for k, g in enumerate(rows) if g[country] == c]
    print(f"{c}: {R[i].sum() / N_part[i].sum():5.2f} contacts per participant ({N_part[i].sum()} participants)")

# %% [markdown]
# ## Who do 10-14 year olds meet?
#
# Contacts per participant ($C = R / N$) for the row group (10-14, F, IT), by contact age,
# summing the two contact genders.

# %%
a = rows.index((10, "F", "IT"))
C_a = R[a] / N_part[a]
ages = Domain("age", sorted({b[0] for b in cols}))
by_age = {age: sum(C_a[j] for j, b in enumerate(cols) if b[0] == age) for (age,) in ages}
last = max(by_age)                         # the last age bin is open-ended (70+)
for age, c in by_age.items():
    label = f"{age}+" if age == last else f"{age}-{age + 4}"
    print(f"  contacts aged {label:>5}: {c:5.2f} {'#' * round(c * 2)}")

# %% [markdown]
# ## Interactive heatmaps
#
# `heatmap` writes a self-contained HTML page: hover a cell for $R_{ab}$, $M_{ab} = R_{ab} / N_a$
# and both groups; switch between $M$ and $R$ and a log colour scale on the page. With
# `block_level="country"`, it draws one panel per country on a shared colour scale.

# %%
full = heatmap(cm, OUT / "polymod.html", title="POLYMOD contact matrix",
               sort_part=["country", "gender", "age"], sort_cont=["gender", "age"])
blocks = heatmap(cm, OUT / "polymod_by_country.html", title="POLYMOD by country",
                 block_level="country", sort_cont=["gender", "age"])
print("open in a browser:")
print(" ", full)
print(" ", blocks)
