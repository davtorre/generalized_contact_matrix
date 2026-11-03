"""Build a generalized contact-matrix dataset from the POLYMOD survey (Mossong et al. 2008).

Raw data: https://zenodo.org/records/3746312 (socialmixr "common" format), expected in
data/raw/polymod/. Run from anywhere:

    python scripts/make_dataset.py
    python scripts/make_dataset.py --part-features age,gender,country --cnt-features age --countries IT,DE

Output: one row per (participant group a, contact group b) with the columns read by
ContactMatrix.from_dataframe:

    part_<feature>...  cnt_<feature>...  contacts  N_part  N_cnt

  contacts  R_ab, total contacts reported by participants in group a with people in group b
  N_part    number of participants in group a (constant for a given a)
  N_cnt     number of recorded contacts in group b, i.e. sum_a R_ab (constant for a given b).
            POLYMOD has no population data; use census counts instead if you need N_b as a
            population size (e.g. for reciprocity).

Every a x b pair is written, with contacts = 0 where nothing was reported, so the matrix is
complete. Participants with zero contacts are kept: they count in N_part.
Age values are the lower bound of the bin (e.g. 5 means [5, 10); the last bin is open-ended).
"""
import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent  # project root
PREFIX = "2008_Mossong_POLYMOD_"

PART_FEATURES = ("age", "gender", "country", "daytype")
CNT_FEATURES = ("age", "gender")
DEFAULT_AGE_BREAKS = list(range(0, 75, 5))  # 0, 5, ..., 70 (70 = 70+)


def read(raw_dir: Path, name: str) -> pl.DataFrame:
    # "" (empty cnt_gender) and "NA" both mean missing
    return pl.read_csv(raw_dir / f"{PREFIX}{name}.csv", null_values=["NA", ""], infer_schema_length=None)


def age_bin(col: str, breaks: list[int]) -> pl.Expr:
    """Map an age to the lower bound of its bin; ages below breaks[0] become null."""
    expr = pl.lit(None, dtype=pl.Int64)
    for b in breaks:  # later (higher) breaks win
        expr = pl.when(pl.col(col) >= b).then(pl.lit(b)).otherwise(expr)
    return expr


def load_participants(raw_dir: Path, breaks: list[int]) -> pl.DataFrame:
    part = read(raw_dir, "participant_common")
    hh = read(raw_dir, "hh_common").select("hh_id", "country")
    sday = read(raw_dir, "sday").select("part_id", "dayofweek")  # 0 = Sunday ... 6 = Saturday
    return (
        part.join(hh, on="hh_id", how="left")
        .join(sday, on="part_id", how="left")
        .select(
            "part_id",
            age_bin("part_age", breaks).alias("part_age"),
            pl.col("part_gender").alias("part_gender"),
            pl.col("country").alias("part_country"),
            pl.when(pl.col("dayofweek").is_in([0, 6])).then(pl.lit("weekend"))
            .when(pl.col("dayofweek").is_not_null()).then(pl.lit("weekday"))
            .alias("part_daytype"),
        )
    )


def load_contacts(raw_dir: Path, breaks: list[int]) -> pl.DataFrame:
    cnt = read(raw_dir, "contact_common")
    lo = pl.min_horizontal("cnt_age_est_min", "cnt_age_est_max")  # some rows have min > max
    hi = pl.max_horizontal("cnt_age_est_min", "cnt_age_est_max")
    return cnt.with_columns(
        # exact age if given, else midpoint of the estimated range
        pl.coalesce("cnt_age_exact", ((lo + hi) // 2)).alias("_age"),
    ).select(
        "part_id",
        age_bin("_age", breaks).alias("cnt_age"),
        pl.col("cnt_gender").alias("cnt_gender"),
    )


def build(raw_dir: Path, part_features: list[str], cnt_features: list[str],
          breaks: list[int], countries: list[str] | None) -> pl.DataFrame:
    pcols = [f"part_{f}" for f in part_features]
    ccols = [f"cnt_{f}" for f in cnt_features]

    part = load_participants(raw_dir, breaks)
    if countries:
        part = part.filter(pl.col("part_country").is_in(countries))
    n0 = part.height
    part = part.drop_nulls(pcols).select("part_id", *pcols)
    print(f"participants: {part.height} kept, {n0 - part.height} dropped (missing {', '.join(part_features)})")

    cnt = load_contacts(raw_dir, breaks).join(part.select("part_id"), on="part_id", how="semi")
    n0 = cnt.height
    cnt = cnt.drop_nulls(ccols)
    print(f"contacts:     {cnt.height} kept, {n0 - cnt.height} dropped (missing {', '.join(cnt_features)})")

    R = (cnt.join(part, on="part_id").group_by(pcols + ccols).agg(pl.len().alias("contacts")))
    N_part = part.group_by(pcols).agg(pl.len().alias("N_part"))

    # contact groups: every combination of category values, so empty groups show up as zeros
    cnt_groups = None
    for c in ccols:
        values = pl.DataFrame({c: breaks}) if c == "cnt_age" else cnt.select(c).unique()
        cnt_groups = values if cnt_groups is None else cnt_groups.join(values, how="cross")

    out = (
        N_part.join(cnt_groups, how="cross")
        .join(R, on=pcols + ccols, how="left")
        .with_columns(pl.col("contacts").fill_null(0))
    )
    N_cnt = out.group_by(ccols).agg(pl.col("contacts").sum().alias("N_cnt"))
    return (
        out.join(N_cnt, on=ccols)
        .select(*pcols, *ccols, "contacts", "N_part", "N_cnt")
        .sort(pcols + ccols)
    )


def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "polymod")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "polymod_contact_matrix.csv")
    ap.add_argument("--part-features", type=csv_list, default=["age", "gender", "country"],
                    help=f"comma-separated, from {PART_FEATURES}")
    ap.add_argument("--cnt-features", type=csv_list, default=["age", "gender"],
                    help=f"comma-separated, from {CNT_FEATURES}")
    ap.add_argument("--age-breaks", type=lambda s: sorted(int(x) for x in csv_list(s)),
                    default=DEFAULT_AGE_BREAKS, help="lower bounds of the age bins, e.g. 0,5,18,65")
    ap.add_argument("--countries", type=csv_list, default=None, help="e.g. IT,DE (default: all)")
    args = ap.parse_args()

    for f in args.part_features:
        if f not in PART_FEATURES:
            ap.error(f"unknown participant feature {f!r}; choose from {PART_FEATURES}")
    for f in args.cnt_features:
        if f not in CNT_FEATURES:
            ap.error(f"unknown contact feature {f!r}; choose from {CNT_FEATURES}")

    df = build(args.raw_dir, args.part_features, args.cnt_features, args.age_breaks, args.countries)
    df.write_csv(args.out)
    print(f"wrote {df.height} rows, {df['contacts'].sum()} contacts -> {args.out}")


if __name__ == "__main__":
    main()
