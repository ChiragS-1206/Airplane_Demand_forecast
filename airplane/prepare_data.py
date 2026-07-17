import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
RAW_CSV = "final_airplane.csv"          # your merged raw T-100 file
OUTPUT_CSV = "route_monthly_data.csv"
MIN_ROUTE_TOTAL_PASSENGERS = 500_000    # same threshold used in the notebook


def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # exact same as notebook's final section: df_clean = df.copy()
    df_clean = df.copy()

    # ---------------------------------------------------------
    # Route-level monthly totals (across all carriers on that route)
    # ---------------------------------------------------------
    print("Building route-level monthly aggregates...")
    route_monthly_all = (
        df_clean.groupby(["ORIGIN", "DEST", "YEAR", "MONTH"])["PASSENGERS"]
        .sum()
        .reset_index()
    )

    # ---------------------------------------------------------
    # Overall (all routes combined) monthly totals
    # ---------------------------------------------------------
    print("Building overall monthly aggregate (ALL routes combined)...")
    overrall_monthly = df_clean.groupby(["YEAR", "MONTH"])["PASSENGERS"].sum().reset_index()
    overrall_monthly["ORIGIN"] = "ALL"
    overrall_monthly["DEST"] = "ALL"

    # ---------------------------------------------------------
    # Combine both
    # ---------------------------------------------------------
    combined_data = pd.concat([route_monthly_all, overrall_monthly], ignore_index=True)
    print(f"Combined shape: {combined_data.shape}")
    print(f"Unique origins: {combined_data['ORIGIN'].nunique()}")

    # ---------------------------------------------------------
    # Filter to active routes only (total passengers > threshold)
    # ---------------------------------------------------------
    print("Filtering to active routes...")
    route_totals = route_monthly_all.groupby(["ORIGIN", "DEST"])["PASSENGERS"].sum().reset_index()
    active_routes = route_totals[route_totals["PASSENGERS"] > MIN_ROUTE_TOTAL_PASSENGERS]
    print(f"Active routes: {active_routes.shape[0]:,}")

    active_routes_pairs = set(zip(active_routes["ORIGIN"], active_routes["DEST"]))

    final_data = combined_data[
        combined_data.apply(
            lambda row: (row["ORIGIN"], row["DEST"]) in active_routes_pairs or row["ORIGIN"] == "ALL",
            axis=1,
        )
    ]
    print(f"Final data shape: {final_data.shape}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved successfully to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
