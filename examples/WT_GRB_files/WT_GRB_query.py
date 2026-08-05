from astroquery.heasarc import Heasarc
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

from astroquery.heasarc import Heasarc
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
from gwosc import datasets


#time conversion
def mjd_to_gps(mjd):
    t = Time(
        mjd,
        format="mjd",
        scale="utc"
    )
    return t.gps

#O4c not available yet
observing_runs = {
    "O1": datasets.run_segment("O1"),
    "O2": datasets.run_segment("O2"),
    "O3a": datasets.run_segment("O3a"),
    "O3b": datasets.run_segment("O3b"),
    "O4a": datasets.run_segment("O4a"),
    "O4b": datasets.run_segment("O4b"),
}


def find_run(gps):
    for run, (start, end) in observing_runs.items():
        if start <= gps <= end:
            return run
    return None


h = Heasarc()

table = h.query_tap("""
SELECT
    name,
    trigger_time,
    ra,
    dec,
    t90
FROM fermigbrst
WHERE t90 < 2
""")


def process_grb_table(table):
    results = []

    for row in table:

        # HEASARC trigger_time is MJD -> GPS
        gps_time = Time(
            row["trigger_time"],
            format="mjd",
            scale="utc"
        ).gps

        # RA/DEC degrees -> radians
        skycoord = SkyCoord(
            ra=row["ra"],
            dec=row["dec"],
            unit="deg"
        )

        results.append({
            "GRB": row["name"],
            "GPS": gps_time,
            "RA_rad": skycoord.ra.radian,
            "DEC_rad": skycoord.dec.radian,
            "T90_s": row["t90"],
        })

    return pd.DataFrame(results)

grb_df = process_grb_table(table)

grb_df["LIGO_run"] = grb_df["GPS"].apply(find_run)

candidate_grbs = grb_df.dropna(subset=["LIGO_run"])


print(candidate_grbs)
print("\nNumber of candidates:", len(candidate_grbs))



# candidate_grbs.to_csv(
#     "Fermi_sGRB_LIGO_candidates.csv",
#     index=False
# )