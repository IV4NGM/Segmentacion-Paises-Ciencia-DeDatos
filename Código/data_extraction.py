from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Install world_bank_data if not available
try:
    import world_bank_data as wb

    print("✓ world_bank_data module available")
except ImportError:
    print("Installing world_bank_data...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "world_bank_data"])
    import world_bank_data as wb

    print("✓ world_bank_data installed successfully")

indicators = {
    # Economic indicators
    "gdp_per_capita": "NY.GDP.PCAP.PP.KD",  # GDP per capita, PPP (constant 2017 international $)
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    "inflation": "FP.CPI.TOTL.ZG",  # Inflation, consumer prices (annual %)
    "unemployment": "SL.UEM.TOTL.ZS",  # Unemployment, total (% of total labor force)
    "exports_gdp": "NE.EXP.GNFS.ZS",  # Exports of goods and services (% of GDP)
    # Social & Health indicators
    "life_expectancy": "SP.DYN.LE00.IN",  # Life expectancy at birth, total (years)
    "infant_mortality": "SP.DYN.IMRT.IN",  # Infant mortality rate (per 1,000 live births)
    "fertility_rate": "SP.DYN.TFRT.IN",  # Fertility rate, total (births per woman)
    "health_expenditure": "SH.XPD.CHEX.GD.ZS",  # Current health expenditure (% of GDP)
    "physicians": "SH.MED.PHYS.ZS",  # Physicians (per 1,000 people)
    # Education indicators
    "literacy_rate": "SE.ADT.LITR.ZS",  # Literacy rate, adult total (% of people ages 15 and above)
    "school_enrollment_secondary": "SE.SEC.ENRR",  # School enrollment, secondary (% gross)
    "school_enrollment_tertiary": "SE.TER.ENRR",  # School enrollment, tertiary (% gross)
    # Infrastructure & Environment
    "electricity_access": "EG.ELC.ACCS.ZS",  # Access to electricity (% of population)
    "co2_emissions": "EN.ATM.CO2E.PC",  # CO2 emissions (metric tons per capita)
    "renewable_energy": "EG.FEC.RNEW.ZS",  # Renewable energy consumption (% of total final energy consumption)
    "mobile_subscriptions": "IT.CEL.SETS.P2",  # Mobile cellular subscriptions (per 100 people)
    "internet_users": "IT.NET.USER.ZS",  # Individuals using the Internet (% of population)
    # Governance & Business
    "ease_business": "IC.BUS.EASE.XQ",  # Ease of doing business index (1=most business-friendly regulations)
    "control_corruption": "CC.EST",  # Control of Corruption estimate
    "government_effectiveness": "GE.EST",  # Government Effectiveness estimate
    # Demographic
    "population": "SP.POP.TOTL",  # Population, total
    "urban_population": "SP.URB.TOTL.IN.ZS",  # Urban population (% of total population)
    "poverty_ratio": "SI.POV.NAHC",  # Poverty headcount ratio at national poverty lines (% of population)
}

print(f"Number of indicators to fetch: {len(indicators)}")

# ------------- 2. Download data -------------
print("Fetching country list...")

# Get all countries (non-aggregate)
try:
    # Get all economies and filter out aggregates
    all_economies = wb.get_countries()
    countries = all_economies[~all_economies.region.isin(["Aggregates"])].index.tolist()
    print(f"Found {len(countries)} non-aggregate economies")

except Exception as e:
    print(f"Error getting country list: {e}")
    # Fallback to major countries
    countries = [
        "USA",
        "CHN",
        "IND",
        "BRA",
        "GBR",
        "FRA",
        "DEU",
        "JPN",
        "CAN",
        "MEX",
        "AUS",
        "ZAF",
        "NGA",
        "EGY",
        "KEN",
        "ETH",
        "MAR",
        "TUN",
        "TUR",
        "SAU",
        "RUS",
        "IDN",
        "KOR",
        "ESP",
        "ITA",
        "ARG",
        "COL",
        "PER",
        "VEN",
        "CHL",
        "PAK",
        "BGD",
        "VNM",
        "THA",
        "UKR",
        "POL",
        "NLD",
        "BEL",
        "SWE",
        "CHE",
        "AUT",
        "NOR",
        "DNK",
        "FIN",
        "GRC",
        "PRT",
        "IRL",
        "ISR",
        "ARE",
        "SGP",
        "MYS",
        "PHL",
        "KAZ",
        "QAT",
        "KWT",
        "OMN",
        "NZL",
        "HUN",
        "CZE",
        "ROU",
    ]
    print(f"Using predefined list of {len(countries)} countries")

print(f"Number of countries to analyze: {len(countries)}")

# %%
# Create a DataFrame by fetching data for each indicator
df_list = []
successful_indicators = []

for name, code in indicators.items():
    print(f"Fetching: {name} ({code})")
    try:
        # Fetch most recent value for each country
        data = wb.get_series(code, country=countries, simplify_index=True, mrv=1)

        if data is not None and len(data) > 0:
            # Convert to DataFrame
            df_temp = pd.DataFrame(data).reset_index()
            df_temp.columns = ["country", name]

            df_list.append(df_temp)
            successful_indicators.append(name)
            print(f"  ✓ Successfully fetched {len(df_temp)} records")
        else:
            print(f"  ✗ No data returned for {name}")

    except Exception as e:
        print(f"  ✗ Error fetching {name}: {str(e)[:100]}...")
        continue

# Check if we have any data
if not df_list:
    raise Exception("No data was successfully fetched.")

print(f"\nSuccessfully fetched {len(df_list)} out of {len(indicators)} indicators")
print(f"Successful indicators: {successful_indicators}")

# Merge all dataframes on country
print("Merging data...")
df_final = df_list[0]
for df_temp in df_list[1:]:
    df_final = pd.merge(df_final, df_temp, on="country", how="outer")

print(f"Final dataset shape: {df_final.shape}")
print(f"Countries in final dataset: {len(df_final)}")

# Set country as index for easier handling
df_final.set_index("country", inplace=True)

# Display basic info about the data
print("\nDataset info:")
print(df_final.info())
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
df_final.to_csv(f"fetched_data_{timestamp}.csv")

print(f"\nData saved to fetched_data_{timestamp}.csv")


print(f"\nMissing values per column:")
missing_summary = df_final.isnull().sum().sort_values(ascending=False)
print(missing_summary)

# Visualize missing data
plt.figure(figsize=(12, 8))
missing_data = df_final.isnull().sum().sort_values(ascending=True)
plt.barh(missing_data.index, missing_data.values, color="salmon", alpha=0.7)
plt.title("Missing Values by Indicator", fontsize=14, fontweight="bold")
plt.xlabel("Number of Missing Values")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
