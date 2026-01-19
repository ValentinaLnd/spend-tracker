import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spend Tracker", layout="wide")

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

# ---------------------------
# Categories + rules (same logic: keyword contains)
# ---------------------------

DEFAULT_CATEGORIES = [
    "Groceries",
    "Restaurants & Cafes",
    "Food delivery",
    "Transport",
    "Parking",
    "Shopping",
    "Beauty",
    "Fitness",
    "Rent / Mortgage",
    "Utilities & Bills",
    "Subscriptions",
    "Healthcare",
    "Travel",
    "Income",
    "Savings / Investments",
    "Other",
]

DEFAULT_RULES = [
    # Groceries
    {"keyword": "SPINNEYS", "category": "Groceries"},
    {"keyword": "CARREFOUR", "category": "Groceries"},
    {"keyword": "GMG CONSUMER", "category": "Groceries"},
    {"keyword": "LULU", "category": "Groceries"},
    {"keyword": "WAITROSE", "category": "Groceries"},

    # Restaurants & Cafes
    {"keyword": "RESTAURANT", "category": "Restaurants & Cafes"},
    {"keyword": "EATALY", "category": "Restaurants & Cafes"},
    {"keyword": "CAFE", "category": "Restaurants & Cafes"},
    {"keyword": "CAFÉ", "category": "Restaurants & Cafes"},
    {"keyword": "BISTRO", "category": "Restaurants & Cafes"},
    {"keyword": "TRATTORIA", "category": "Restaurants & Cafes"},

    # Food delivery
    {"keyword": "DELIVEROO", "category": "Food delivery"},
    {"keyword": "TALABAT", "category": "Food delivery"},
    {"keyword": "ZOMATO", "category": "Food delivery"},

    # Transport
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "RTA", "category": "Transport"},
    {"keyword": "SALIK", "category": "Transport"},
    {"keyword": "VALTRANS", "category": "Transport"},
    {"keyword": "TAXI", "category": "Transport"},

    # Parking
    {"keyword": "PARKING", "category": "Parking"},

    # Shopping
    {"keyword": "AMAZON", "category": "Shopping"},
    {"keyword": "NOON", "category": "Shopping"},
    {"keyword": "IKEA", "category": "Shopping"},
    {"keyword": "ACE", "category": "Shopping"},

    # Beauty
    {"keyword": "SEPHORA", "category": "Beauty"},
    {"keyword": "FACES", "category": "Beauty"},
    {"keyword": "TIPS & TOES", "category": "Beauty"},
    {"keyword": "BEAUTY", "category": "Beauty"},
    {"keyword": "THE BODY SHOP", "category": "Beauty"},
    {"keyword": "KIKO", "category": "Beauty"},
    {"keyword": "MAC", "category": "Beauty"},

    # Fitness
    {"keyword": "FITNESS FIRST", "category": "Fitness"},
    {"keyword": "CLASS PASS", "category": "Fitness"},
    {"keyword": "GYM", "category": "Fitness"},
    {"keyword": "DECATHLON", "category": "Fitness"},

    # Utilities & Bills
    {"keyword": "DEWA", "category": "Utilities & Bills"},
    {"keyword": "DU", "category": "Utilities & Bills"},
    {"keyword": "ETISALAT", "category": "Utilities & Bills"},

    # Subscriptions
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "AMAZON PRIME", "category": "Subscriptions"},
    {"keyword": "APPLE", "category": "Subscriptions"},
    {"keyword": "GOOGLE", "category": "Subscriptions"},
]


# ---------------------------
# Parsing helpers for NEW input: single column with raw transaction lines
# Example line:
# 01/03/2025 02/03/2025 VALTRANS TRANSPORTATIO DUBAI ARE 60.00
# ---------------------------

RAW_LINE_REGEX = re.compile(
    r"^(?P<date1>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<date2>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<details>.+?)\s+"
    r"(?P<currency>[A-Z]{3})\s+"
    r"(?P<amount>[-\d\.,]+)\s*$"
)

def parse_amount_to_float(x: str) -> float | None:
    """
    Handles: 60.00, 1,040.00, 53,00
    """
    s = str(x).strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")       # 1,040.00 -> 1040.00
    elif "," in s and "." not in s:
        s = s.replace(",", ".")      # 53,00 -> 53.00
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def read_single_column_export(file) -> pd.DataFrame:
    """
    Reads single-column XLSX/CSV where each row is a raw transaction line.
    IMPORTANT: header=None so first transaction isn't treated as header.
    """
    name = file.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file, engine="openpyxl", header=None)
        raw_series = df.iloc[:, 0]
    elif name.endswith(".csv"):
        df = pd.read_csv(file, header=None)
        raw_series = df.iloc[:, 0]
    else:
        raise ValueError(f"Unsupported file type: {file.name}. Please upload XLSX or CSV.")

    raw_series = raw_series.dropna().astype(str).str.strip()
    raw_series = raw_series[raw_series != ""]

    rows = []
    for line in raw_series.tolist():
        m = RAW_LINE_REGEX.match(line)
        if not m:
            # Keep the raw line as a non-parsed row marker (won't appear in analytics)
            rows.append(
                {"raw": line, "date": pd.NaT, "details": line, "currency": None, "amount": None}
            )
            continue

        amt = parse_amount_to_float(m.group("amount"))
        rows.append(
            {
                "raw": line,
                "date": pd.to_datetime(m.group("date1"), format="%d/%m/%Y", errors="coerce"),
                "details": m.group("details").strip(),
                "currency": m.group("currency").strip(),
                # Expenses-only file => store as negative
                "amount": -abs(amt) if amt is not None else None,
            }
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["date", "amount"]).copy()
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["year"] = out["date"].dt.year
    return out


# ---------------------------
# Categorisation (same keyword logic; fill empty only)
# ---------------------------

def apply_rules_fill_empty_only(df: pd.DataFrame, rules_df: pd.DataFrame, default_category: str = "Other") -> pd.DataFrame:
    df = df.copy()
    if "Categories" not in df.columns:
        df["Categories"] = ""

    desc = df["details"].fillna("").astype(str).str.upper()
    cats = df["Categories"].fillna("").astype(str)

    empty_mask = cats.str.strip() == ""
    result = cats.copy()

    # default for empties
    result.loc[empty_mask] = default_category

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue
        hit = desc.str.contains(re.escape(kw), na=False)
        result.loc[empty_mask & hit] = cat

    df["Categories"] = result
    return df


# ---------------------------
# Export (Google Sheets tabs per month)
# ---------------------------

def safe_sheet_name(name: str) -> str:
    name = re.sub(r'[:\\/?*\[\]]', "-", name)
    return name[:31]

def build_monthly_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    df_sorted = df.sort_values(["month", "date"]).copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False, sheet_name="All")
        for m in sorted(df_sorted["month"].unique()):
            sheet = safe_sheet_name(str(m))
            df_m = df_sorted[df_sorted["month"] == m]
            df_m.to_excel(writer, index=False, sheet_name=sheet)

    return output.getvalue()


# ---------------------------
# App UI
# ---------------------------

def main():
    st.title("💳 Spend Tracker (Merged accounts, single-column bank export)")

    st.sidebar.header("1) Upload your bank export")
    st.sidebar.write("Upload **one** XLSX/CSV where each row is a single transaction line (expenses only).")

    upload = st.sidebar.file_uploader(
        "Merged bank export (XLSX/CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
        key="merged_export",
    )

    st.sidebar.header("2) Categories")
    categories_text = st.sidebar.text_area(
        "Categories (one per line)",
        value="\n".join(DEFAULT_CATEGORIES),
        height=220,
    )
    category_list = [c.strip() for c in categories_text.splitlines() if c.strip()]

    if "rules_df" not in st.session_state:
        st.session_state.rules_df = pd.DataFrame(DEFAULT_RULES)

    if "df_all" not in st.session_state:
        st.session_state.df_all = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load data", type="primary"):
            if upload is None:
                st.warning("Please upload your XLSX/CSV file first.")
            else:
                try:
                    df_all = read_single_column_export(upload)
                    # Start empty categories; user applies rules
                    df_all["Categories"] = ""
                    st.session_state.df_all = df_all

                    st.success(f"Loaded {len(df_all)} transactions ✅")
                    st.dataframe(df_all.head(25), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    return

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload your file and click **Load data**.")
        st.markdown(
            "Expected row format example:\n\n"
            "`01/03/2025 02/03/2025 CAREEM HALA DUBAI ARE 53,00`"
        )
        return

    st.subheader("Step 1 – Rules (auto-categorisation)")
    rules_editor = st.data_editor(
        st.session_state.rules_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "keyword": st.column_config.TextColumn("Keyword (contains)"),
            "category": st.column_config.SelectboxColumn("Category", options=category_list),
        },
        key="rules_editor",
    )

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("⚙️ Apply rules and categorize"):
            st.session_state.rules_df = rules_editor
            st.session_state.df_all = apply_rules_fill_empty_only(
                st.session_state.df_all, rules_editor, default_category="Other"
            )
            st.success("Rules applied ✅")

    df_all = st.session_state.df_all

    st.subheader("Step 2 – Trends")

    # --- chart-safe categories to prevent Altair schema ValueError ---
    df_all_for_charts = df_all.copy()
    df_all_for_charts["Categories"] = (
        df_all_for_charts.get("Categories", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Other")
    )
    df_all_for_charts["Categories"] = df_all_for_charts["Categories"].str.replace(r"\s+", " ", regex=True)

    spend = df_all_for_charts[df_all_for_charts["amount"] < 0].copy()
    spend["spend"] = spend["amount"].abs()

    if spend.empty:
        st.warning("No spend detected. Your file is expected to contain expenses only.")
    else:
        monthly = (
            spend.groupby(["month", "Categories"], as_index=False)["spend"]
            .sum()
            .pivot(index="month", columns="Categories", values="spend")
            .fillna(0)
        )

        # Make column names safe + unique for Altair
        monthly.columns = [str(c).strip() if str(c).strip() else "Other" for c in monthly.columns]
        seen = {}
        new_cols = []
        for c in monthly.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c} ({seen[c]})")
        monthly.columns = new_cols

        st.line_chart(monthly)

        top_cat = spend.groupby("Categories", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
        st.bar_chart(top_cat.set_index("Categories"))

    st.subheader("Export to Google Sheets (tabs per month)")
    xlsx_bytes = build_monthly_excel(df_all_for_charts)
    st.download_button(
        label="⬇️ Download Excel (one tab per month)",
        data=xlsx_bytes,
        file_name="spend_tracker_monthly_tabs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Upload to Google Drive → Open with Google Sheets to get monthly tabs.")

    st.subheader("Raw data")
    st.dataframe(df_all.sort_values("date"), use_container_width=True)


if __name__ == "__main__":
    main()
