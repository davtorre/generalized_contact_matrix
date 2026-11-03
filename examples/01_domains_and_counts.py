# %% [markdown]
# # 1. Domains and counts
#
# A contact matrix needs to know *which groups* its rows and columns are. In `gcm` that is a
# **`Domain`**: a list of feature names (`domain_names`) and the ordered groups (`elements`)
# they take. A **`Counts`** is one number per group of a domain, e.g. the vector $\underline N$
# of participants per group.
#
# Run as a script (`python examples/01_domains_and_counts.py`) or open it as a notebook.

# %%
from gcm import Counts, Domain

# %% [markdown]
# ## Building domains
#
# A domain over one feature takes plain values; each becomes a 1-tuple.

# %%
age = Domain("age", ["0-4", "5-9", "10-14"])
gender = Domain("gender", ["F", "M"])
print(age)
print(age.elements)

# %% [markdown]
# The Cartesian product `*` combines features: every age with every gender. The groups are
# tuples with one entry per feature, in a fixed order, and each has an index: row/column `i`
# of a matrix over this domain is `groups[i]`.

# %%
groups = age * gender
print(groups)
print(len(groups), "groups:", groups.elements)
print("index of ('5-9', 'F'):", groups.index(("5-9", "F")))
print("groups[3]:", groups[3])
print("('5-9', 'F') in groups:", ("5-9", "F") in groups)

# %% [markdown]
# ## Fixed domains catch typos
#
# By default a domain grows when a new group shows up. A **fixed** domain refuses instead,
# which turns a typo into an error rather than a silent extra group.

# %%
fixed_gender = Domain("gender", ["F", "M"], fixed=True)
try:
    fixed_gender.add("Femal")
except ValueError as e:
    print("rejected:", e)

# %% [markdown]
# ## Sorting
#
# Groups keep the order in which they were added; `sorted(by=...)` returns a sorted copy.
# Here the groups are sorted by gender first, then by age.

# %%
ages = Domain("age", [10, 0, 5]) * Domain("gender", ["M", "F"])
print(ages.elements)
print(ages.sorted(by=["gender"]).elements)

# %% [markdown]
# ## Counts: one number per group
#
# `Counts(domain, values)` takes one value per group, in domain order. Groups are looked up
# by their tuple. `None` means **unknown**, which is different from zero.

# %%
N = Counts(groups, [120, 95, 80, 101, None, 75])
print(N)
print("N[('5-9', 'F')] =", N[("5-9", "F")])

N.add(("5-9", "F"), 3)                     # increment
N[("0-4", "M")] = 97                       # set
print("after add/set:", dict(N))

print("unknown group count:", N[("10-14", "F")])
print("with a default:     ", N.get(("10-14", "F"), 0))

# %% [markdown]
# `to_numpy()` returns the values and (a copy of) the domain. Because one count is unknown,
# the array is float, with `NaN` in its place.

# %%
values, domain = N.to_numpy()
print(values)
print(domain.elements)
