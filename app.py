import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spending Tracker", layout="wide")

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

DEFAULT_CATEGORIES = [
    "Groceries",
    "Restaurants & Cafes",
    "Transport",
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
    {"keyword": "DELIVEROO", "category": "Restaurants & Cafes"},
    {"keyword": "TALABAT", "category": "Restaurants & Cafes"},
    {"keyword": "ZOMATO", "category": "Restaurants & Cafes"},
    {"keyword": "STARBUCKS", "category": "Restaurants & Cafes"},
    {"keyword": "PAUL", "category": "Restaurants & Cafes"},

    # Transport
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "RTA", "category": "Transport"},
    {"keyword": "SALIK", "category": "Transport"},
    {"keyword": "VALTRANS", "category": "Transport"},
    {"keyword": "TAXI", "category": "Transport"},

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
    {"keyword": "EMARAT", "category": "Utilities & Bills"},

    # Subscriptions
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "AMAZON PRIME", "category": "Subscriptions"},
    {"keyword": "APPLE", "category": "Subscriptions"},
    {"keyword": "GOOGLE", "category": "Subscriptions"},

    # Healthcare
    {"keyword": "PHARMACY", "category": "Healthcare"},
    {"keyword": "CLINIC", "category": "Healthcare"},
    {"keyword": "MEDICAL", "category": "Healthcare"},

    # Travel
    {"keyword": "EMIRATES", "category": "Travel"},
    {"keyword": "ETIHAD", "category": "Travel"},
    {"keyword": "BOOKING.COM", "category": "Travel"},
    {"keyword": "AIRBNB", "category": "Travel"},
]


# ---------------------------
# Parsing helpers (your bank format)
# ---------------------------

def parse_money(series: pd.Series) -> pd.Series:
    """
    Parses amounts like '53,00' -> 53.00, removes spaces, keeps digits/dot/minus.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def read_file_to_df(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(f)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(f, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {f.name}. Please upload CSV or Excel (XLSX/XLS).")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_statement_df(
    df_raw: pd.DataFrame,
    account_name: str,
    date_col: str | None,
    desc_col: str | None,
    amount_col: str | None,
    dc_col: str | None,
) -> pd.DataFrame:
    """
    Expects columns like:
      Date | Details | Amount | Currency | Debit/Credit | Status
    Converts Debit rows to negative amounts.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def to_none(v):
        return None if (v is None or v == "(auto)") else v

    date_col = to_none(date_col)
    desc_col = to_none(desc_col)
    amount_col = to_none(amount_col)
    dc_col = to_none(dc_col)

    # Auto-detect if user left "(auto)"
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if desc_col is None:
        desc_col = next((c for c in df.columns if c.lower() in ["details", "description", "narration"]), None)
    if amount_col is None:
        amount_col = next((c for c in df.columns if "amount" in c.lower() or "amt" in c.lower()), None)
    if dc_col is None:
        dc_col = next((c for c in df.columns if "debit/credit" in c.lower() or "debit credit" in c.lower()), None)

    if not all([date_col, desc_col, amount_col]):
        raise ValueError(
            "Could not detect required columns.\n"
            f"Columns found: {list(df.columns)}\n"
            "Select Date + Description + Amount in the sidebar."
        )

    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "description": df[desc_col].astype(str),
            "amount": parse_money(df[amount_col]),
        }
    )

    # Debit => negative spend
    if dc_col is not None:
        dc = df[dc_col].astype(str).str.upper().str.strip()
        tmp.loc[dc == "DEBIT", "amount"] *= -1

    tmp["account"] = account_name
    tmp = tmp.dropna(subset=["date", "amount"])
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    tmp["year"] = tmp["date"].dt.year

    if tmp.empty:
        raise ValueError(
            "0 rows parsed after cleaning.\n"
            "Make sure Date/Details/Amount are selected and Amount is numeric (e.g. 53,00)."
        )

    return tmp


def load_and_clean(files, account_name: str, mapping: dict) -> pd.DataFrame:
    dfs = []
    for f in files:
        df_raw = read_file_to_df(f)
        tmp = normalize_statement_df(
            df_raw=df_raw,
            account_name=account_name,
            date_col=mapping["date_col"],
            desc_col=mapping["desc_col"],
            amount_col=mapping["amount_col"],
            dc_col=mapping["dc_col"],
        )
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------
# Categorisation
# ---------------------------

def apply_rules(df: pd.DataFrame, rules_df: pd.DataFrame, default_category: str = "Other") -> pd.DataFrame:
    df = df.copy()
    desc = df["description"].fillna("").str.upper()
    categories = pd.Series(default_category, index=df.index)

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue
        categories[desc.str.contains(kw, na=False)] = cat

    df["category"] = categories
    return df


# ---------------------------
# Export (Google Sheets-friendly)
# ---------------------------

def safe_sheet_name(name: str) -> str:
    # Excel sheet name rules: max 31 chars, no : \ / ? * [ ]
    name = re.sub(r'[:\\/?*\[\]]', "-", name)
    return name[:31]


def build_monthly_excel(df: pd.DataFrame) -> bytes:
    """
    Excel output:
      - All
      - One sheet per month (YYYY-MM)
    Upload to Google Drive and open with Google Sheets -> tabs appear.
    """
    output = io.BytesIO()
    df_sorted = df.sort_values(["month", "date"]).copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False, sheet_name="All")

        for month in sorted(df_sorted["month"].unique()):
            sheet = safe_sheet_name(str(month))
            df_m = df_sorted[df_sorted["month"] == month]
            df_m.to_excel(writer, index=False, sheet_name=sheet)

    return output.getvalue()


# ---------------------------
# App
# ---------------------------

def main():
    st.title("💳 12-Month Spending Tracker (Excel/CSV, Auto-categorized)")

    st.sidebar.header("1. Upload files")
    acc1_files = st.sidebar.file_uploader(
        "Account 1 (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="acc1",
    )
    acc2_files = st.sidebar.file_uploader(
        "Account 2 (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="acc2",
    )

    account1_name = st.sidebar.text_input("Account 1 name", value="Account 1")
    account2_name = st.sidebar.text_input("Account 2 name", value="Account 2")

    st.sidebar.header("0. Column mapping (for your export format)")

    sample_cols = []
    try:
        sample_file = (acc1_files or acc2_files or [None])[0]
        if sample_file is not None:
            sample_df = read_file_to_df(sample_file)
            sample_cols = [str(c).strip() for c in sample_df.columns]
    except Exception:
        sample_cols = []

    date_pick = st.sidebar.selectbox("Date column", ["(auto)"] + sample_cols, index=0)
    desc_pick = st.sidebar.selectbox("Description column", ["(auto)"] + sample_cols, index=0)
    amount_pick = st.sidebar.selectbox("Amount column", ["(auto)"] + sample_cols, index=0)
    dc_pick = st.sidebar.selectbox("Debit/Credit column (recommended)", ["(auto)"] + sample_cols, index=0)

    mapping = {
        "date_col": date_pick,
        "desc_col": desc_pick,
        "amount_col": amount_pick,
        "dc_col": dc_pick,
    }

    st.sidebar.header("2. Categories")
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

    if st.button("Load & combine data", type="primary"):
        try:
            all_dfs = []
            if acc1_files:
                all_dfs.append(load_and_clean(acc1_files, account1_name, mapping))
            if acc2_files:
                all_dfs.append(load_and_clean(acc2_files, account2_name, mapping))

            df_all = pd.concat(all_dfs, ignore_index=True).sort_values("date")
            df_all["category"] = "Other"
            st.session_state.df_all = df_all

            st.success(f"Loaded {len(df_all)} transactions ✅")
            st.dataframe(df_all.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading statements: {e}")
            return

    df_all = st.session_state.df_all
    if df_all is None:
        st.info(
            "Upload files, choose mapping (Date/Details/Amount/Debit-Credit), then click **Load & combine data**.\n\n"
            "For your file, pick: **Date → Date**, **Description → Details**, **Amount → Amount**, "
            "**Debit/Credit → Debit/Credit**."
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

    if st.button("⚙️ Apply rules and categorize"):
        st.session_state.rules_df = rules_editor
        st.session_state.df_all = apply_rules(st.session_state.df_all, rules_editor, default_category="Other")
        st.success("Rules applied ✅")

    df_all = st.session_state.df_all

    st.subheader("Step 2 – Trends")

    spend = df_all[df_all["amount"] < 0].copy()
    spend["spend"] = spend["amount"].abs()

    if spend.empty:
        st.warning(
            "No spend (negative amounts) detected. Ensure you selected the **Debit/Credit** column in mapping."
        )
    else:
        monthly = (
            spend.groupby(["month", "category"], as_index=False)["spend"]
            .sum()
            .pivot(index="month", columns="category", values="spend")
            .fillna(0)
        )
        st.line_chart(monthly)

        top_cat = spend.groupby("category", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
        st.bar_chart(top_cat.set_index("category"))

    st.subheader("Export to Google Sheets (tabs per month)")

    xlsx_bytes = build_monthly_excel(df_all.sort_values(["month", "date"]))
    st.download_button(
        label="⬇️ Download Excel (one tab per month) for Google Sheets",
        data=xlsx_bytes,
        file_name="spend_tracker_monthly_tabs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption("To use in Google Sheets: upload this file to Google Drive → Open with Google Sheets.")

    st.subheader("Raw data")
    st.dataframe(df_all.sort_values("date"), use_container_width=True)


if __name__ == "__main__":
    main()
