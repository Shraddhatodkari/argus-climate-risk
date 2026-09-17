"""Real extreme-heat hazard — IMD station climatological normals, not a synthetic
proxy. This closes the gap the V1 demo build left open ("extreme heat / temperature,
if feasible" was not attempted there); it turned out to be feasible.

Source: India Meteorological Department Pune (dsp.imdpune.gov.in) station
climatological statistics — monthly normals per station, including
``mean_of_highest_maximum_temperature_in_the_month_in_c`` (the mean of each year's
single hottest day in that month, at that station — a real extreme-heat indicator,
not a daily average). Retrieved via a public GitHub mirror
(github.com/Vonter/india-climatological-stats) of IMD's own published statistics;
see ``LOCAL_SNAPSHOT_PATH``. The dataset carries a ``district`` column directly from
IMD's own station metadata, so no separate subdivision-mapping assumption is needed
here (unlike rainfall_imd.py) -- resolution is genuinely district-level wherever a
station exists in that district.

Methodology — district heat exposure score (0-100):
1. For the district's IMD station (see ``DISTRICT_TO_STATION_OVERRIDE`` for the one
   case in this demo set with no in-district station), take the calendar month with
   the highest ``mean_of_highest_maximum_temperature_in_the_month_in_c`` -- i.e. the
   district's real climatological peak-heat month.
2. Score = 100 x (that value - HEAT_FLOOR_C) / (HEAT_CEILING_C - HEAT_FLOOR_C),
   clipped to [0, 100]. HEAT_FLOOR_C=30 (a value most of India's normals sit above,
   so it doesn't compress the scale) and HEAT_CEILING_C=48 (close to the highest
   extreme-heat normals seen anywhere in this station dataset, e.g. Bundelkhand/
   central India) are disclosed, documented reference points, not fitted to make any
   particular district look a certain way.

No district in this demo build lacks IMD climatological coverage outright, but
Kanchipuram's own station does not appear in this dataset (Kanchipuram district's
station network is thin in IMD's public normals compilation); Chennai, the nearest
station with real data (~70km away), is used as an explicit, labeled Proxy for that
one district -- never silently presented as Kanchipuram's own reading.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import DATA_DIR
from argus.common.provenance import DataProvenance, DataStatus

LOCAL_SNAPSHOT_PATH = DATA_DIR / "real" / "imd_station_climatological_stats.csv"

HEAT_FLOOR_C = 30.0
HEAT_CEILING_C = 48.0

# Only needed where a district has no station of its own in this compilation.
# Each override is disclosed with the real distance/reason, and the returned score's
# per-district status is PROXY, not OBSERVED, for these districts.
DISTRICT_TO_STATION_OVERRIDE = {
    "Kanchipuram": ("CHENNAI", "TAMIL NADU", "nearest available IMD station (~70km); Kanchipuram itself is not in this compilation"),
}

# Districts that DO have their own station in this compilation, but under a
# differently formatted district label in IMD's own metadata (e.g. Kamrup district
# was bifurcated into Kamrup Metro / Kamrup Rural after this station network was
# set up) -- still a real, in-district station, so still OBSERVED, just resolved by
# the exact label the source data uses.
DISTRICT_LABEL_ALIAS = {
    "Kamrup": "KAMRUP (RURAL)",
}


def load_station_climatology() -> pd.DataFrame:
    return pd.read_csv(LOCAL_SNAPSHOT_PATH)


def compute_district_heat_exposure(districts: pd.DataFrame) -> tuple[pd.DataFrame, DataProvenance]:
    """districts needs 'district' (IMD district-name spelling is upper-cased
    automatically) and 'state'. Returns district, heat_risk, peak_month,
    peak_month_mean_daily_max_c, peak_month_extreme_high_c, station_used, status."""
    retrieved_at = DataProvenance.now_iso()
    climo = load_station_climatology()

    rows = []
    for _, d in districts.iterrows():
        district, state = d["district"], d["state"]
        if district in DISTRICT_TO_STATION_OVERRIDE:
            station_district, station_state, _reason = DISTRICT_TO_STATION_OVERRIDE[district]
            sub = climo[(climo["district"] == station_district) & (climo["state"] == station_state)]
            status = DataStatus.PROXY
        else:
            label = DISTRICT_LABEL_ALIAS.get(district, district.upper())
            sub = climo[(climo["district"] == label) & (climo["state"] == state.upper())]
            status = DataStatus.OBSERVED

        if sub.empty:
            continue

        peak = sub.loc[sub["mean_of_highest_maximum_temperature_in_the_month_in_c"].idxmax()]
        extreme_high = float(peak["mean_of_highest_maximum_temperature_in_the_month_in_c"])
        score = 100.0 * (extreme_high - HEAT_FLOOR_C) / (HEAT_CEILING_C - HEAT_FLOOR_C)
        score = max(0.0, min(100.0, score))

        rows.append(
            {
                "district": district,
                "heat_risk": round(score, 2),
                "peak_month": int(peak["month"]),
                "peak_month_mean_daily_max_c": round(float(peak["mean_daily_maximum_temperature_in_c"]), 1),
                "peak_month_extreme_high_c": round(extreme_high, 1),
                "station_used": peak["station_name"],
                "status": status.value,
            }
        )

    provenance = DataProvenance(
        dataset="IMD station climatological normals (monthly)",
        source_organization="India Meteorological Department, Pune",
        source_url="https://dsp.imdpune.gov.in/",
        publication_date="IMD Pune published climatological normals compilation",
        observation_period="multi-decade station normals (varies by station; not a single fixed period)",
        geographic_resolution="station (district-linked via IMD's own metadata)",
        unit="degrees Celsius",
        methodology=(
            "Real IMD station monthly normals; district score is Derived from the Observed peak-month "
            "extreme-high normal via the linear scaling documented in heat_imd.py's module docstring. "
            "One district (Kanchipuram) uses a Proxy nearest-station substitute -- see status column."
        ),
        status=DataStatus.DERIVED,  # the per-district table's own 'status' column carries the finer Observed/Proxy split
        retrieved_at=retrieved_at,
    )
    return pd.DataFrame(rows), provenance


if __name__ == "__main__":
    demo = pd.DataFrame(
        [
            {"district": "Banda", "state": "Uttar Pradesh"},
            {"district": "Kanchipuram", "state": "Tamil Nadu"},
        ]
    )
    result, prov = compute_district_heat_exposure(demo)
    print(result.to_string(index=False))
    print(prov)
