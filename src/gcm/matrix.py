import numpy as np
import polars as pl
from typing import List, Dict, Iterable
from itertools import product


def _sort_key(value, positions):
    # None last; type name before value so ints and strings are never compared directly
    return tuple((value[p] is None, type(value[p]).__name__, value[p] if value[p] is not None else 0)
                 for p in positions)


class Domain:
    """Named domains and the ordered list of elements (groups) they take.

    Every element is a tuple with one entry per domain name, e.g. (0, 'F') for domain names
    ['age', 'gender']; a single-name domain also accepts plain elements (0 -> (0,)).
    Elements keep the order in which they were added, and each has a fixed index.
    A fixed domain rejects new elements (use it to catch typos such as 'Femal').

        age = Domain('age', range(0, 75, 5))
        gender = Domain('gender', ['F', 'M'])
        groups = age * gender          # domain names ['age', 'gender'], 30 elements (0,'F'), (0,'M'), ...
    """

    def __init__(self, domain_names: List[str] | str = (), elements: Iterable = (), fixed: bool = False):
        self.domain_names: List[str] = [domain_names] if isinstance(domain_names, str) else list(domain_names)
        repeated = {n for n in self.domain_names if self.domain_names.count(n) > 1}
        if repeated:
            raise ValueError(f"repeated domain names: {sorted(repeated)}")
        self._index: Dict[tuple, int] = {}   # element -> position
        self._elements: List[tuple] = []     # position -> element
        self.fixed = False
        for e in elements:
            self.add(e)
        self.fixed = fixed

    def _as_tuple(self, element) -> tuple:
        if len(self.domain_names) == 1 and not isinstance(element, tuple):
            element = (element,)
        if not isinstance(element, tuple) or len(element) != len(self.domain_names):
            raise ValueError(f"{element!r} needs one entry per domain name {self.domain_names}")
        return element

    def add(self, element) -> int:
        """Add an element (no-op if already there) and return its index."""
        element = self._as_tuple(element)
        if element not in self._index:
            if self.fixed:
                raise ValueError(f"{element} is not in the fixed domain {self.domain_names}")
            self._index[element] = len(self._elements)
            self._elements.append(element)
        return self._index[element]

    def index(self, element) -> int:
        return self._index[self._as_tuple(element)]

    @property
    def elements(self) -> List[tuple]:
        return list(self._elements)

    def __getitem__(self, i):
        """domain[i] is the i-th element (row/column i of the matrix); slices give lists."""
        return self._elements[i]

    def extend_domain_names(self, new_names: List[str]):
        """Add domain names; existing elements get None ("any value") for them."""
        if self.fixed:
            raise ValueError(f"cannot add {new_names} to the fixed domain {self.domain_names}")
        clash = [n for n in new_names if n in self.domain_names]
        if clash:
            raise ValueError(f"domain names already present: {clash}")
        self.domain_names.extend(new_names)
        suffix = (None,) * len(new_names)
        self._elements = [e + suffix for e in self._elements]
        self._index = {e: i for i, e in enumerate(self._elements)}

    def sorted(self, by: List[str] | None = None) -> "Domain":
        """A copy with elements sorted by the domain names in `by` (default: all, in order); None last."""
        by = list(by or self.domain_names)
        unknown = [n for n in by if n not in self.domain_names]
        if unknown:
            raise ValueError(f"cannot sort by {unknown}: domain names are {self.domain_names}")
        positions = [self.domain_names.index(n) for n in by] + \
                    [i for i, n in enumerate(self.domain_names) if n not in by]
        return Domain(self.domain_names, sorted(self._elements, key=lambda e: _sort_key(e, positions)), self.fixed)

    def copy(self) -> "Domain":
        return Domain(self.domain_names, self._elements, self.fixed)

    def __mul__(self, other: "Domain") -> "Domain":
        """Cartesian product: domain names concatenated, elements (a + b) for a, b in product."""
        if not isinstance(other, Domain):
            return NotImplemented
        return Domain(self.domain_names + other.domain_names,
                      (a + b for a, b in product(self._elements, other._elements)),
                      fixed=self.fixed and other.fixed)

    def __len__(self):
        return len(self._elements)

    def __iter__(self):
        return iter(self._elements)

    def __contains__(self, element):
        try:
            return self._as_tuple(element) in self._index
        except ValueError:
            return False

    def __eq__(self, other):
        return isinstance(other, Domain) and self.domain_names == other.domain_names and self.elements == other.elements

    def __repr__(self):
        shown = ", ".join(map(repr, self.elements[:4])) + (", ..." if len(self) > 4 else "")
        return (f"Domain(domain_names={self.domain_names}, {len(self)} elements: [{shown}]"
                f"{', fixed' if self.fixed else ''})")

    def pop(self, name : str):
        # Normalize
        name = str(name)
        # Check if name is in the domain:
        if name not in self.domain_names:
            raise ValueError(f"{name} not present in this domain. Domain: {self!r}")
        # Store id
        removed_name_id = self.domain_names.index(name)
        # New domain
        new_domain = self.domain_names.copy()
        removed_name = new_domain.pop(removed_name_id)
        new_elements = [el[:removed_name_id] + el[removed_name_id + 1:] for el in self._elements]
        self.__init__(new_domain, new_elements, self.fixed)

        return removed_name

class Counts:
    """One number per group of a Domain: the vector N (participants, census population, ...).

    Values are stored in the order of the domain's elements; None means unknown (not zero).

        groups = Domain('age', ['0-4', '5-9']) * Domain('gender', ['F', 'M'])
        N = Counts(groups, [120, 95, 80, 101])
        N[('5-9', 'F')]               # 80
        N[('5-9', 'F')] = 82          # set
        N.add(('5-9', 'F'), 3)        # increment -> 85
        values, domain = N.to_numpy()
    """

    def __init__(self, domain : Domain, counts : Iterable | None = None):
        """domain: the groups (copied); counts: one value per element, in domain order (default: unknown)."""
        if not isinstance(domain, Domain):
            raise TypeError(f"domain must be a Domain, not {type(domain).__name__}")
        self.domain = domain.copy()
        self._values = self._check(counts, self.domain)

    @classmethod
    def _shared(cls, domain : Domain, counts : Iterable | None = None) -> "Counts":
        """A Counts that uses `domain` itself (no copy): it follows the domain as it grows.
        Used by ContactMatrix, whose N_part / N_cont live on the matrix's own domains."""
        N = cls.__new__(cls)
        N.domain = domain
        N._values = cls._check(counts, domain)
        return N

    @staticmethod
    def _value(x, where):
        """A count as a plain Python number, or None for unknown (None or NaN)."""
        if x is None:
            return None
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{where}: counts must be numbers, not {x!r}")
        if x != x:                      # NaN
            return None
        if x < 0:
            raise ValueError(f"{where}: counts cannot be negative ({x})")
        return x

    @classmethod
    def _check(cls, counts, domain) -> list:
        if counts is None:
            return [None] * len(domain)
        counts = list(np.asarray(counts, dtype=object).ravel()) if not isinstance(counts, list) else counts
        if len(counts) != len(domain):
            raise ValueError(f"{len(counts)} counts, but the domain has {len(domain)} groups")
        return [cls._value(x, f"group {g}") for x, g in zip(counts, domain)]

    def _sync(self):
        # groups added to a shared domain since the last call start as unknown
        self._values.extend([None] * (len(self.domain) - len(self._values)))

    def _position(self, group, add : bool = False) -> int:
        self._sync()
        if group in self.domain:
            return self.domain.index(group)
        if not add:
            raise KeyError(f"{group} is not a group of the domain {self.domain.domain_names}")
        i = self.domain.add(group)       # raises if the domain is fixed
        self._sync()
        return i

    def __getitem__(self, group):
        return self._values[self._position(group)]

    def __setitem__(self, group, value):
        """Set the count of a group; a group not yet in the domain is added (unless the domain is fixed)."""
        value = self._value(value, f"group {group}")
        self._values[self._position(group, add=True)] = value

    def add(self, group, count):
        """Increment the count of a group (unknown counts as 0); a new group is added to the domain."""
        count = self._value(count, f"group {group}")
        i = self._position(group, add=True)
        if count is not None:
            self._values[i] = (self._values[i] or 0) + count

    def get(self, group, default=None):
        """The count of a group, or `default` if the group is not in the domain or its count is unknown."""
        self._sync()
        if group not in self.domain:
            return default
        value = self._values[self.domain.index(group)]
        return default if value is None else value

    @classmethod
    def from_numpy(cls, counts, domain : Domain) -> "Counts":
        """The inverse of to_numpy (same as Counts(domain, counts))."""
        return cls(domain, counts)

    def to_numpy(self):
        """(values, domain): an int array if every count is a known integer, else float with NaN
        for unknown counts. values[i] is the count of domain[i]; the domain is a copy."""
        self._sync()
        if all(isinstance(v, int) for v in self._values):
            values = np.array(self._values, dtype=int)
        else:
            values = np.array([np.nan if v is None else v for v in self._values], dtype=float)
        return values, self.domain.copy()

    def __len__(self):
        return len(self.domain)

    def __iter__(self):
        """Iterate over (group, count) pairs, in domain order."""
        self._sync()
        return iter(zip(self.domain.elements, self._values))

    def __eq__(self, other):
        if not isinstance(other, Counts):
            return NotImplemented
        self._sync(); other._sync()
        return self.domain == other.domain and self._values == other._values

    def __repr__(self):
        self._sync()
        shown = ", ".join(f"{g!r}: {v}" for g, v in list(zip(self.domain.elements, self._values))[:4])
        return f"Counts({self.domain.domain_names}, {len(self)} groups: {{{shown}{', ...' if len(self) > 4 else ''}}})"


class Element:
    def __init__(self, feature_part : Dict, feature_cont : Dict, R : int, N_part : int, N_cont : int):
        self.features_participant = feature_part
        self.features_contact = feature_cont
        self.total_contacts = R
        self.number_participants = N_part
        self.number_contacts = N_cont

    def __str__(self):
        return f"{self.features_participant} {self.features_contact}: {self.total_contacts}"

class ContactMatrix:

    def add_element(self, element : Element, overwrite : bool = True):

        element_domain_part = [el for el in element.features_participant.keys()]
        element_domain_cont = [el for el in element.features_contact.keys()]
        if set(self.domain_part.domain_names) - set(element_domain_part):
            print(f"Cannot add element {element}: too few features in part.")
            return
        if set(self.domain_cont.domain_names) - set(element_domain_cont):
            print(f"Cannot add element {element}: too few features in cont.")
            return
        # Keep the element's own feature order (a set would make it vary between runs)
        new_elements_part = [el for el in element_domain_part if el not in self.domain_part.domain_names]
        new_elements_cont = [el for el in element_domain_cont if el not in self.domain_cont.domain_names]

        # Run every check before changing anything, so a rejected element leaves the matrix untouched
        if (self.extend_domain_part is False or self.domain_part.fixed) and len(new_elements_part) > 0:
            print(f"New domain element{'s' * (len(new_elements_part) > 1)} found: {new_elements_part}")
            return
        if (self.extend_domain_cont is False or self.domain_cont.fixed) and len(new_elements_cont) > 0:
            print(f"New domain element{'s' * (len(new_elements_cont) > 1)} found: {new_elements_cont}")
            return
        # None is reserved for the "any value" placeholder of features added later
        if None in element.features_participant.values() or None in element.features_contact.values():
            print(f"Cannot add element {element}: feature values cannot be None.")
            return

        # Keys are tuples of feature values, in domain order (new features last)
        part_features = tuple(element.features_participant[f] for f in self.domain_part.domain_names + new_elements_part)
        cont_feature = tuple(element.features_contact[f] for f in self.domain_cont.domain_names + new_elements_cont)
        if self.domain_part.fixed and part_features not in self.domain_part:
            print(f"Cannot add element {element}: {part_features} is not a participant group of the fixed domain.")
            return
        if self.domain_cont.fixed and cont_feature not in self.domain_cont:
            print(f"Cannot add element {element}: {cont_feature} is not a contact group of the fixed domain.")
            return

        # Old keys get None ("any value") for new features, in the domain and in the stored data
        # (N_part / N_cont share the domains, so they follow without renaming)
        if len(new_elements_part) > 0:
            suffix = (None,) * len(new_elements_part)
            self.domain_part.extend_domain_names(new_elements_part)
            self.contacts = {k + suffix: row for k, row in self.contacts.items()}
        if len(new_elements_cont) > 0:
            suffix = (None,) * len(new_elements_cont)
            self.domain_cont.extend_domain_names(new_elements_cont)
            self.contacts = {k: {c + suffix: v for c, v in row.items()} for k, row in self.contacts.items()}

        self.domain_part.add(part_features)
        self.domain_cont.add(cont_feature)

        if part_features not in self.contacts:
            self.contacts[part_features] = {}

        if cont_feature in self.contacts[part_features]:
            print(f"Element {part_features}, {cont_feature} in this contact matrix with value: {self.contacts[part_features][cont_feature]}.")
            if overwrite:
                print("Updating with the new element, and number of participants and contacts")
            else: return

        self.contacts[part_features][cont_feature] = element.total_contacts
        self.N_part[part_features] = element.number_participants
        self.N_cont[cont_feature] = element.number_contacts

    def __init__(self, init_elements : list[Element] | None = None,
                 domain_part : Domain | List[str] | None = None, domain_cont : Domain | List[str] | None = None,
                 extend_domain_part : bool = True, extend_domain_cont : bool = True):
        """domain_part / domain_cont: a Domain (copied, so the original is never modified), or a
        list of domain names (elements are then collected from the added Elements)."""
        self.contacts = {}
        self.extend_domain_part = extend_domain_part
        self.extend_domain_cont = extend_domain_cont
        self.domain_part = self._as_domain(domain_part)
        self.domain_cont = self._as_domain(domain_cont)
        # N of the rows (participants per group) and of the columns, on the matrix's own domains
        self.N_part = Counts._shared(self.domain_part)
        self.N_cont = Counts._shared(self.domain_cont)
        if init_elements:
            for element in init_elements:
                self.add_element(element)


    @staticmethod
    def _as_domain(domain) -> Domain:
        if domain is None:
            return Domain()
        if isinstance(domain, Domain):
            return domain.copy()
        return Domain(list(domain))

    def from_dataframe(self, df : pl.DataFrame, add = True):
        cols = df.columns
        part_columns = [c for c in cols if c.startswith('part_')]
        cont_columns = [c for c in cols if c.startswith('cnt_')]
        if not part_columns:
            print("Dataframe doesn't hold any participant feature.")
            return
        if not cont_columns:
            print("Dataframe doesn't hold any contact feature.")
            return
        if "contacts" not in cols:
            print("Dataframe doesn't hold volume of contacts.")
            return
        missing_N = [c for c in ("N_part", "N_cnt") if c not in cols]
        if missing_N:
            print(f"Dataframe is missing column{'s' * (len(missing_N) > 1)}: {missing_N}")
            return

        elements = []
        for row in df.iter_rows(named=True):
            fp = {c[len('part_'):]: row[c] for c in part_columns}
            fc = {c[len('cnt_'):]: row[c] for c in cont_columns}
            total_contacts = row['contacts']
            N_part = row['N_part']
            N_cnt = row['N_cnt']

            elements.append(Element(fp, fc, total_contacts, N_part, N_cnt))
        if add:
            # Add the rows into this matrix
            for el in elements: self.add_element(el)
            return self
        # Build a separate matrix with the same domain settings, leaving this one untouched
        return ContactMatrix(elements,
                             domain_part=self.domain_part, domain_cont=self.domain_cont,
                             extend_domain_part=self.extend_domain_part,
                             extend_domain_cont=self.extend_domain_cont)

    @classmethod
    def from_numpy(cls, R, domain_part : Domain, domain_cont : Domain, N_part = None, N_cont = None,
                   extend_domain_part : bool = True, extend_domain_cont : bool = True) -> "ContactMatrix":
        """Build a ContactMatrix from a matrix of contacts and the two domains: the inverse of to_numpy.

        R            2D array or nested list, shape (len(domain_part), len(domain_cont)); row i is the
                     participant group domain_part.elements[i], column j the contact group
                     domain_cont.elements[j]
        domain_part  Domain of the rows; domain_cont: Domain of the columns (both are copied)
        N_part       optional numbers of participants: a list/array with one value per row, in the
                     same order (None = unknown), or a Counts holding every participant group
        N_cont       optional numbers of contacts, the same way for the columns

            age = Domain('age', [0, 5, 10])
            cm = ContactMatrix.from_numpy([[12, 3, 1], [4, 9, 2], [1, 2, 6]], age, age, N_part=[50, 40, 30])
        """
        for name, d in (("domain_part", domain_part), ("domain_cont", domain_cont)):
            if not isinstance(d, Domain):
                raise TypeError(f"{name} must be a Domain (it gives the order of the rows/columns), "
                                f"not {type(d).__name__}")
        R = np.asarray(R)
        n_a, n_b = len(domain_part), len(domain_cont)
        if R.ndim != 2 or R.shape != (n_a, n_b):
            raise ValueError(f"R has shape {R.shape}, but the domains need ({n_a}, {n_b}): {n_a} participant "
                             f"groups {domain_part.domain_names} x {n_b} contact groups {domain_cont.domain_names}")
        if not np.issubdtype(R.dtype, np.number):
            raise ValueError(f"R must hold numbers, not {R.dtype}")
        if np.issubdtype(R.dtype, np.floating) and np.isnan(R).any():
            raise ValueError("R contains NaN")
        if (R < 0).any():
            raise ValueError("R contains negative numbers of contacts")

        def counts(N, domain, name):
            """N as a list in domain order; a Counts is looked up group by group."""
            if not isinstance(N, Counts):
                return N
            missing = [g for g in domain if g not in N.domain]
            if missing:
                raise ValueError(f"{name} has no count for {len(missing)} groups, e.g. {missing[0]}")
            return [N[g] for g in domain]

        cm = cls(domain_part=domain_part, domain_cont=domain_cont,
                 extend_domain_part=extend_domain_part, extend_domain_cont=extend_domain_cont)
        for name, N, dom in (("N_part", N_part, cm.domain_part), ("N_cont", N_cont, cm.domain_cont)):
            try:
                setattr(cm, name, Counts._shared(dom, counts(N, dom, name)))
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
        labels_cont = cm.domain_cont.elements
        cm.contacts = {a: {b: R[i, j].item() for j, b in enumerate(labels_cont)}
                       for i, a in enumerate(cm.domain_part.elements)}
        return cm

    def to_numpy(self):
        """R as an array, with the domains of its rows and columns: R, domain_part, domain_cont.

        Row i is the participant group domain_part[i], column j the contact group domain_cont[j];
        the returned domains are copies. Groups without data are rows/columns of zeros.
        ContactMatrix.from_numpy(*cm.to_numpy()) rebuilds the same matrix.
        """
        # integer counts stay int; any non-integer value (e.g. weighted contacts) makes the array float
        integer = all(isinstance(v, (int, np.integer)) for row in self.contacts.values() for v in row.values())
        R = np.zeros(shape = (len(self.domain_part), len(self.domain_cont)), dtype = int if integer else float)
        for fp, row in self.contacts.items():
            a = self.domain_part.index(fp)
            for fc, element in row.items():
                b = self.domain_cont.index(fc)
                R[a,b] = element

        return R, self.domain_part.copy(), self.domain_cont.copy()

    def to_dataframe(self) -> pl.DataFrame:
        """The inverse of from_dataframe: one row per (participant group, contact group) pair of
        the domains, with contacts = 0 where nothing was stored; N_part / N_cnt are null when unknown."""
        rows = []
        for a in self.domain_part:
            row_a = self.contacts.get(a, {})
            part = {f"part_{f}": v for f, v in zip(self.domain_part.domain_names, a)}
            for b in self.domain_cont:
                cont = {f"cnt_{f}": v for f, v in zip(self.domain_cont.domain_names, b)}
                rows.append({**part, **cont, "contacts": row_a.get(b, 0),
                             "N_part": self.N_part.get(a), "N_cnt": self.N_cont.get(b)})
        return pl.DataFrame(rows, infer_schema_length=None)

    def marginalize(self, part: List[str] = (), cont: List[str] = (), null_as_zero: bool = True) -> "ContactMatrix":
        # Normalize input
        part, cont = list(dict.fromkeys(part)), list(dict.fromkeys(cont))
        # If both are empty, just return to save time
        if not part and not cont: return self
        # Validate input
        for side, names, dom in (("participant", part, self.domain_part),
                                 ("contact", cont, self.domain_cont)):
            unknown = [n for n in names if n not in dom.domain_names]
            if unknown:
                raise ValueError(f"Unknown {side} names. {unknown} not available in the domain {dom}")
        # ToDo: implement marginalization on all the features. Implementing the class ALL needed.
        if (len(part) == len(self.domain_part.domain_names)) or (len(cont) == len(self.domain_cont.domain_names)):
            raise ValueError("Marginalizing on all the features is not implemented yet.")
        # Select which column to keep
        keep_part   = [f'part_{n}' for n in self.domain_part.domain_names if n not in part]
        keep_cont   = [f'cnt_{n}'  for n in self.domain_cont.domain_names if n not in cont]
        all_part    = [f'part_{n}' for n in self.domain_part.domain_names]
        all_cnt     = [f'cnt_{n}'  for n in self.domain_cont.domain_names]

        # Helping function
        def total(col, null_as_zero):
            if null_as_zero:
                return pl.col(col).sum().alias(col)
            else:
                return (pl.when(pl.col(col).null_count() == 0)
                            .then(pl.col(col).sum())
                            .otherwise(pl.lit(None)) 
                            .alias(col)
                        )
        # Generate dataframe
        df = self.to_dataframe()
        # Marginalize ContactMatrix via group_by and sum
        R = df.group_by(keep_part + keep_cont).agg(pl.col('contacts').sum())
        # Marginalize Counts
        N_part = df.select(["N_part"] + all_part).unique().group_by(keep_part).agg(total("N_part", null_as_zero)) # De-duplicate, then sum
        N_cont = df.select(["N_cnt"]  + all_cnt).unique().group_by(keep_cont).agg(total("N_cnt", null_as_zero) ) # De-duplicate, then sum

        # Assemble the dataframe
        out_df = R.join(N_part, on = keep_part, how = "left", nulls_equal = True)
        out_df = out_df.join(N_cont, on = keep_cont, how = "left", nulls_equal = True)

        # Pop domain pieces to reduce domain
        new_domain_part = self.domain_part.copy()
        new_domain_cont = self.domain_cont.copy()
        for p in part:
            new_domain_part.pop(p)
        for c in cont:
            new_domain_cont.pop(c)

        new_cm = ContactMatrix(domain_part = new_domain_part, domain_cont=new_domain_cont,
                                extend_domain_part=self.extend_domain_part, extend_domain_cont=self.extend_domain_cont)
        return new_cm.from_dataframe(out_df)
        