# %% [markdown]
# # 2. Building a contact matrix
#
# A `ContactMatrix` stores $R_{ab}$, the total number of contacts between participant group
# $a$ (rows) and contact group $b$ (columns), together with the domains of both sides and
# their counts `N_part` / `N_cont`. Rows and columns can use *different* features, so the
# matrix can be rectangular.
#
# There are three ways in, and two ways out:
#
# | You have | Use |
# |---|---|
# | domains + a matrix of numbers | `ContactMatrix.from_numpy(R, domain_part, domain_cont)` |
# | one cell at a time | `cm.add_element(Element(...))` |
# | a long table | `ContactMatrix().from_dataframe(df)` |
#
# and back with `to_numpy()` and `to_dataframe()`.

# %%
import numpy as np

from gcm import ContactMatrix, Counts, Domain, Element

# %% [markdown]
# ## From a matrix of numbers
#
# Participants are described by age and gender (6 groups), contacts by age only (3 groups):
# a 6 × 3 matrix. Row `i` is `domain_part[i]` and column `j` is `domain_cont[j]`.

# %%
domain_part = Domain("age", ["0-4", "5-9", "10-14"]) * Domain("gender", ["F", "M"])
domain_cont = Domain("age", ["0-4", "5-9", "10-14"])

R = np.array([[30, 12,  4],      # (0-4, F)
              [28, 10,  5],      # (0-4, M)
              [11, 65, 14],      # (5-9, F)
              [ 9, 70, 12],      # (5-9, M)
              [ 3, 15, 80],      # (10-14, F)
              [ 4, 13, 85]])     # (10-14, M)
N_part = Counts(domain_part, [10, 9, 12, 13, 11, 12])     # participants per row group

cm = ContactMatrix.from_numpy(R, domain_part, domain_cont, N_part=N_part)

# %% [markdown]
# `to_numpy()` gives back the matrix and the domains of its rows and columns, so
# `ContactMatrix.from_numpy(*cm.to_numpy())` rebuilds the same matrix.

# %%
R_back, rows, cols = cm.to_numpy()
print(R_back)
print("rows:", rows.elements)
print("cols:", cols.elements)
print("R for (5-9, M) -> 5-9:", R_back[rows.index(("5-9", "M")), cols.index("5-9")])

# %% [markdown]
# `to_dataframe()` writes one row per (participant group, contact group) pair: the format of
# `data/processed/` and of `from_dataframe`. Unknown counts (here `N_cnt`) are null.

# %%
df = cm.to_dataframe()
print(df.head(6))

# %% [markdown]
# ## Contacts per participant
#
# The contact rate matrix is $C = \mathrm{diag}(\underline N)^{-1} \underline{\underline R}$:
# each row divided by its number of participants.

# %%
N_values, _ = cm.N_part.to_numpy()
C = R_back / N_values[:, None]
print(np.round(C, 2))
print("mean contacts per participant:", round(R_back.sum() / N_values.sum(), 2))

# %% [markdown]
# ## One cell at a time
#
# `add_element` takes the groups as dicts (feature -> value), so the order of the features
# doesn't matter. The domains grow as new groups appear.

# %%
cm2 = ContactMatrix()
cm2.add_element(Element({"age": "0-4"}, {"age": "0-4"}, R=30, N_part=10, N_cont=None))
cm2.add_element(Element({"age": "0-4"}, {"age": "5-9"}, R=12, N_part=10, N_cont=None))
cm2.add_element(Element({"age": "5-9"}, {"age": "5-9"}, R=65, N_part=12, N_cont=None))
print(cm2.domain_part)
print(cm2.to_numpy()[0])

# %% [markdown]
# ### A new feature: `None` means "I don't hold the data"
#
# If an element brings a feature the matrix has never seen (here `gender`), the domain gains
# that feature. The groups stored before have no value for it, so they get `None`: their
# data was recorded without gender.

# %%
cm2.add_element(Element({"age": "5-9", "gender": "F"}, {"age": "0-4"}, R=11, N_part=12, N_cont=None))
print(cm2.domain_part)
print(dict(cm2.N_part))

# %% [markdown]
# ## From a table, and back
#
# `from_dataframe` reads `part_<feature>`, `cnt_<feature>`, `contacts`, `N_part`, `N_cnt`
# columns. Reading the table written above gives the same matrix.

# %%
cm3 = ContactMatrix().from_dataframe(df)
R3, rows3, cols3 = cm3.to_numpy()
print("same matrix:", np.array_equal(R3, R_back), "| same groups:", rows3 == rows and cols3 == cols)
