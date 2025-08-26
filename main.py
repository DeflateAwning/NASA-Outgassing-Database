import io
from pathlib import Path

import requests
import pandas as pd
import polars as pl
from loguru import logger
from ordered_set import OrderedSet

URL_TEMPLATE = "https://outgassing.nasa.gov/outgassing-data-table?field_material_value_op=contains&field_material_value=&field_application_value_op=contains&field_application_value=&field_data_ref_value_op=contains&field_data_ref_value=&field_cvcm_value_op=%3C%3D&field_cvcm_value%5Bvalue%5D=&field_cvcm_value%5Bmin%5D=&field_cvcm_value%5Bmax%5D=&field_tml_value_op=%3C%3D&field_tml_value%5Bvalue%5D=&field_tml_value%5Bmin%5D=&field_tml_value%5Bmax%5D=&sort_by=field_material_value&sort_order=ASC&items_per_page=500&page={page_num}"


def do_scrape(page_num: int) -> pl.DataFrame | None:
    url = URL_TEMPLATE.format(page_num=page_num)
    response = requests.get(url)
    response.raise_for_status()

    if "<table" not in response.text:
        return None

    # Read the HTML table into a DataFrame
    df_pd = pd.read_html(  # pyright: ignore
        io.StringIO(response.text),
        flavor="lxml",
    )[0]

    df = pl.from_pandas(df_pd)

    df = (
        df.with_columns(
            pl.all().cast(pl.String),
        )
        .with_columns(pl.all().str.strip_chars().replace("", None))
        .rename(
            {
                "TML %": "TML_Pct",
                "WVR": "WVR_Pct",
                "CVCM": "CVCM_Pct",
                "Mfr.": "Manufacturer",
            }
        )
        .rename(lambda col: col.strip().replace(" ", "_"))
        .cast(
            {
                "TML_Pct": pl.Decimal,
                "WVR_Pct": pl.Decimal,
                "CVCM_Pct": pl.Decimal,
            }
        )
        .with_columns(
            # Remove decimal places from Year:
            pl.col("Year").cast(pl.Float64).cast(pl.UInt16),
            # Calculate SpaceX "Recoverable Mass Loss (RML)" from TML and WVR:
            RML_Pct=pl.col("TML_Pct") - pl.col("WVR_Pct").fill_null(0.0),
        )
    )
    return df


def scrape_nasa_outgassing() -> pl.DataFrame:
    df_list: list[pl.DataFrame] = []
    page_num = 1

    while True:
        df = do_scrape(page_num)

        if df is None:
            logger.info(f"No data table found on page {page_num}. End of scrape.")
            break

        logger.info(f"Scraped page {page_num}: {len(df)} rows")

        df_list.append(df)
        page_num += 1

    df = pl.concat(df_list, how="vertical")

    count_before = len(df)
    df = df.unique(maintain_order=True)
    count_after = len(df)
    logger.info(
        f"Removed {count_before - count_after} duplicate rows. Remaining rows: {count_after:,} rows."
    )

    # Add col: "SpaceX_Classification"
    df = df.with_columns(
        SpaceX_Classification=(
            pl.when(
                # Per RPUG V10 Rev 2024-09, Table 5-5.
                (pl.col("RML_Pct") <= pl.lit(1.0)) & (pl.col("CVCM_Pct") <= pl.lit(0.1))
            )
            .then(pl.lit("Pass"))
            .when(
                # Per RPUG V10 Rev 2024-09, Table 6-7.
                (pl.col("RML_Pct") <= pl.lit(3.0)) & (pl.col("CVCM_Pct") <= pl.lit(0.1))
            )
            .then(pl.lit("Rationale Code A (up to 2 sq-in)"))
            .when(
                # Per RPUG V10 Rev 2024-09, Table 6-7
                (pl.col("RML_Pct") > pl.lit(3.0)) | (pl.col("CVCM_Pct") > pl.lit(0.1))
            )
            .then(pl.lit("Rationale Code B (up to 0.25 sq-in)"))
            .otherwise(pl.lit("Fail"))
        )
    )

    acronyms = (
        "RF",
        "RFI",
        "EMC",
        "EMI",
        "ESD",
        "UV",
        "DVD",
        "IC",
        "ID",
        "PC",
        "PCB",
        "LED",
        "LCD",
        "TC",
        "LRO",
        "PCB",
    )

    # Clean up the "Application" column.
    unique_applications_before = df["Application"].n_unique()
    df = df.with_columns(
        pl.col("Application").str.to_titlecase(),
        Raw_Application=pl.col("Application"),  # Unmodified from source.
    )
    for acronym in acronyms:
        df = df.with_columns(
            pl.col("Application").str.replace_all(rf"(?i)\b{acronym}\b", acronym)
        )
    df = df.with_columns(
        pl.col("Application")
        .str.replace_all("&", " and ", literal=True)
        .str.replace_all(r"(?i)\b(Adh)\b", "Adhesive")
        .str.replace_all(r"(?i)\b(Tpe)\b", "Tape")
        .str.replace_all(r"(?i)\b(Anti.?static)\b", "Antistatic")
        .str.replace_all(r"(?i)\b(Conf)\b", "Conformal")
        .str.replace_all(r"(?i)\b(Cond)\b", "Conductive")
        .str.replace_all(r"(?i)\b(Cpnd)\b", "Compound")
        .str.replace_all(r"(?i)\b(Cmpd)\b", "Compound")
        .str.replace_all(r"(?i)\b(Elec)\b", "Electrical")
        .str.replace_all(r"(?i)\b(electrical.conductive)\b", "Electrically-Conductive")
        .str.replace_all(r"(?i)\b(blk)\b", "Black")
        .str.replace_all(r"(?i)\b(unk)\b", "Unknown")
        .str.replace_all(r"(?i)\b(vib)\b", "Vibration")
        .str.replace_all(r"(?i)\b(opt)\b", "Optical")  # Abbreviation unification.
        .str.replace_all(r"(?i)\b(therm?)\b", "Thermal")  # Abbreviation unification.
        .str.replace_all(
            r"(?i)\b(therm\w*.conductive)\b", "Thermally-Conductive"
        )  # Abbreviation unification.
        .str.replace_all(r"(?i)\b(lube)\b", "Lubricant")  # Abbreviation unification.
        .str.replace_all(r"(?i)\b(matl|mtl)s?\.?\b", "Material")  # Abbreviation fix.
        .str.replace_all(r"(?i)\bmaerials\b", "Materials")  # Typo fix.
        .str.replace_all(r"(?i)\bcoatint\b", "Coating")  # Typo fix.
        .str.replace_all(r"(?i)\bWrap+i[nm]g\b", "Wrapping")  # Typo fix.
        .str.replace_all(r"\s+", " ")  # Remove extra spaces
        .str.replace_all(r"3.?[Dd]", "3D")  # Normalize 3D/3D (e.g., "3D Printing")
        .str.replace_all(r"(?i)\b[0O].?ring\b", "O-Ring")  # Normalize O-ring
        .str.replace_all(r"(?i)\band\b", "and")  # Lowercase "and"
        .str.replace_all(r"(?i)\bfor\b", "for")  # Lowercase "for"
        .str.replace_all(".", "", literal=True)  # Remove periods
        .str.replace_all(r"\s+", " ")  # Remove extra spaces
        .str.strip_chars()
        .replace(
            {
                "2 Side Tape": "Tape, Double-Sided",
                "2 Sided Tape": "Tape, Double-Sided",
                "Tape 2 Side": "Tape, Double-Sided",
                "Tape 2 Sided": "Tape, Double-Sided",
                "Capicator": "Capacitor",
                "Electrical Comp": "Electrical Component",
                "Electrical Components": "Electrical Component",
                "PC Board": "PCB",
            }
        )
    )
    unique_applications_after = df["Application"].n_unique()
    logger.info(
        f"Normalized 'Application' column: {unique_applications_before} unique values before, "
        f"{unique_applications_after} unique values after."
    )

    # Re-order the columns.
    main_cols = OrderedSet(
        [
            "Material",
            "Application",
            "TML_Pct",
            "WVR_Pct",
            "CVCM_Pct",
            "RML_Pct",
            "Year",
            "SpaceX_Classification",
            "Data_Ref",
            "Manufacturer",
        ]
    )
    other_cols = OrderedSet(df.columns) - main_cols
    all_cols = list(main_cols | other_cols)
    df = df.select(all_cols)

    # Note: Seems that no individual column is a proper unique key. Not "Material" nor "Data_Ref".

    logger.info(
        f"Summary by 'SpaceX_Classification': {df['SpaceX_Classification'].value_counts(sort=True)}"
    )

    return df


def main() -> None:
    df = scrape_nasa_outgassing()

    output_folder = Path(__file__).parent / "output"
    df.write_csv(output_folder / "nasa_outgassing.csv")
    df.write_parquet(output_folder / "nasa_outgassing.pq")
    df.write_excel(output_folder / "nasa_outgassing.xlsx")
    logger.info("Scraping completed and data saved to nasa_outgassing.csv")


if __name__ == "__main__":
    logger.info("Starting NASA outgassing data scrape")
    main()
    logger.info("Scraping finished successfully")
