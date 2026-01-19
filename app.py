import io
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spend Tracker — Categorise", layout="wide")


# ---------------------------
# Approved categories + rules
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
# Helpers
# ---------------------------

def safe_sheet_name(name: str) -> str:
    name = re.sub(r'[:\\/?*\[\]]', "-", name)
    return name[:31]


def build_monthly_excel(df: pd.DataFrame) -> bytes:
    """
    Excel output:
      - All
      - one sheet per month (YYYY-MM)
    """
    output = io.BytesIO()
    df_sorted = df.sort_values(["month", "date"]).copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False, sheet_name="All")
        for m in sorted(df_sorted["month"].dropna().unique()):
            sheet = safe_sheet_name(str(m))
            df_m = df_sorted[df_sorted["month"] == m]
            df_m.to_excel(writer, index=False, sheet_name=sheet)

    return output.getvalue()


def apply_rules_fill_empty_only(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    default_category: str = "Other",
) -> pd.DataFrame:
    """
    Keyword-based categorisation.
    Only fills category where empty.
    """
    df = df.copy()

    if "category" not in df.columns:
        df["category"] = ""

    desc = df["details"].fillna("").astype(str).str.upper()
    cats = df["category"].fillna("").astype(str)

    empty_mask = cats.str.strip() == ""
    result = cats.copy()

    # Default only for empty rows
    result.loc[empty_mask] = default_category

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue
        hit = desc.str.contains(re.escape(kw), na=False)
        result.loc[empty_mask & hit] = cat

    df["category"] = result
    return df


# ---------------------------
# Input loader: 2 columns (Date + Details)
# ---------------------------

def read_two_col_export(file) -> pd.DataFrame:
    """
    Supports XLSX/CSV with:
      - Column A: Date
      - Column B: Details
    Headers are optional.
    """
    name = file.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(file, engine="openpyxl", header=None)
    elif name.endswith(".csv"):
        raw = pd.read_csv(file, header=None)
    else:
        raise ValueError(f"Unsupported file type: {file.name}. Please upload XLSX or CSV.")

    if raw.shape[1] < 2:
        raise ValueError("Expected at least 2 columns: Date + Details.")

    col0 = raw.iloc[:, 0]
    col1 = raw.iloc[:, 1]

    # Detect header row
    first0 = str(col0.iloc[0]).strip().lower()
    first1 = str(col1.iloc[0]).strip().lower()
    has_header = ("date" in first0) and ("detail" in first1 or "description" in first1)

    if has_header:
        col0 = col0.iloc[1:]
        col1 = col1.iloc[1:]

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(col0, errors="coerce", dayfirst=True),
            "details": col1.astype(str).str.strip(),
        }
    )

    df = df.dropna(subset=["date"]).copy()
    df = df[df["details"].str.strip() != ""].copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df


# ---------------------------
# App
# ---------------------------

def main():
    st.title("💳 Spend Tracker — Categorise Expenses")

    st.sidebar.header("1) Upload your file(s)")
    st.sidebar.write("Upload XLSX or CSV with **2 columns**: **Date** + **Details**.")

    files = st.sidebar.file_uploader(
        "Upload files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
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

    if st.button("Load & combine data", type="primary"):
        if not files:
            st.warning("Upload at least one file.")
        else:
            try:
                dfs = [read_two_col_export(f) for f in files]
                df_all = pd.concat(dfs, ignore_index=True).sort_values("date")
                df_all["category"] = ""
                st.session_state.df_all = df_all

                st.success(f"Loaded {len(df_all)} rows ✅")
                st.dataframe(df_all.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading files: {e}")
                return

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload your file(s) and click **Load & combine data**.")
        st.markdown(
            "Expected format:\n\n"
            "- Column A: Date (e.g. `01/03/2025`)\n"
            "- Column B: Details (e.g. `CAREEM HALA DUBAI`)\n"
        )
        return

    st.subheader("Rules (auto-categorisation)")
    rules_editor = st.data_editor(
        st.session_state.rules_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "keyword": st.column_config.TextColumn("Keyword (contains)"),
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=category_list,
            ),
        },
        key="rules_editor",
    )

    if st.button("⚙️ Apply rules"):
        st.session_state.rules_df = rules_editor
        st.session_state.df_all = apply_rules_fill_empty_only(
            st.session_state.df_all,
            rules_editor,
            default_category="Other",
        )
        st.success("Rules applied ✅")

    df_all = st.session_state.df_all

    st.subheader("Categorised data")
    st.dataframe(df_all.sort_values("date"), use_container_width=True)

    st.subheader("Export to Google Sheets")
    xlsx_bytes = build_monthly_excel(df_all)
    st.download_button(
        label="⬇️ Download Excel (tabs per month)",
        data=xlsx_bytes,
        file_name="spend_tracker_monthly_tabs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Upload to Google Drive → Open with Google Sheets.")


if __name__ == "__main__":
    main()
