import io
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spend Tracker — Categorise", layout="wide")

# ---------------------------
# Approved taxonomy (includes Pharmacy)
# ---------------------------

CATEGORIES = [
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
    "Pharmacy",
    "Travel",
    "Income",
    "Savings / Investments",
    "Other",
]

# ---------------------------
# Default rules (refined with your lists)
# ---------------------------

DEFAULT_RULES = [
    # -------------------
    # Groceries
    # -------------------
    {"keyword": "SPINNEYS", "category": "Groceries"},
    {"keyword": "CARREFOUR", "category": "Groceries"},
    {"keyword": "GMG CONSUMER", "category": "Groceries"},
    {"keyword": "LULU", "category": "Groceries"},
    {"keyword": "WAITROSE", "category": "Groceries"},
    {"keyword": "ORGANIC FOODS", "category": "Groceries"},
    {"keyword": "CCA AL ACCAD", "category": "Groceries"},
    {"keyword": "DEPARTMEN", "category": "Groceries"},  # catches DEPARTMENT/DEPARTMEN variants

    # -------------------
    # Restaurants & Cafes
    # -------------------
    {"keyword": "RESTAURANT", "category": "Restaurants & Cafes"},
    {"keyword": "CAFE", "category": "Restaurants & Cafes"},
    {"keyword": "CAFÉ", "category": "Restaurants & Cafes"},
    {"keyword": "COFFEE", "category": "Restaurants & Cafes"},
    {"keyword": "COFFEE SHOP", "category": "Restaurants & Cafes"},
    {"keyword": "ROASTERS", "category": "Restaurants & Cafes"},
    {"keyword": "ESPRESSO", "category": "Restaurants & Cafes"},
    {"keyword": "BISTRO", "category": "Restaurants & Cafes"},
    {"keyword": "TRATTORIA", "category": "Restaurants & Cafes"},
    {"keyword": "EATALY", "category": "Restaurants & Cafes"},

    # Merchant boosters you asked for
    {"keyword": "BARISTA", "category": "Restaurants & Cafes"},
    {"keyword": "ANGEL PARK REST", "category": "Restaurants & Cafes"},
    {"keyword": "DIN TAI FUNG", "category": "Restaurants & Cafes"},
    {"keyword": "LDCKITCHEN", "category": "Restaurants & Cafes"},
    {"keyword": "IBRIC COFFE", "category": "Restaurants & Cafes"},

    # New list (restaurants)
    {"keyword": "SUFRET MARYAM", "category": "Restaurants & Cafes"},
    {"keyword": "ONE AND ONLY", "category": "Restaurants & Cafes"},
    {"keyword": "KAI BEACH", "category": "Restaurants & Cafes"},
    {"keyword": "REIF KUSHIYAKI", "category": "Restaurants & Cafes"},
    {"keyword": "COMPTOIR 102", "category": "Restaurants & Cafes"},
    {"keyword": "LA FABBRICA ITALIANA", "category": "Restaurants & Cafes"},
    {"keyword": "BDC HOSPITALITY", "category": "Restaurants & Cafes"},
    {"keyword": "MYTHOS KOUZINA", "category": "Restaurants & Cafes"},
    {"keyword": "KYOCHON", "category": "Restaurants & Cafes"},
    {"keyword": "THE MAINE", "category": "Restaurants & Cafes"},  # fixes THE MAINE STREET EATER

    # -------------------
    # Food delivery
    # -------------------
    {"keyword": "DELIVEROO", "category": "Food delivery"},
    {"keyword": "TALABAT", "category": "Food delivery"},
    {"keyword": "ZOMATO", "category": "Food delivery"},
    {"keyword": "CASINETTO", "category": "Food delivery"},

    # -------------------
    # Transport
    # -------------------
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "RTA", "category": "Transport"},
    {"keyword": "SALIK", "category": "Transport"},
    {"keyword": "VALTRANS", "category": "Transport"},
    {"keyword": "TAXI", "category": "Transport"},
    # EPPCO ambiguity in your examples -> handle by site-specific rules
    {"keyword": "EPPCO SITE 1060", "category": "Transport"},

    # -------------------
    # Parking
    # -------------------
    {"keyword": "PARKING", "category": "Parking"},
    {"keyword": "PARKONIC", "category": "Parking"},
    {"keyword": "ANE PARK", "category": "Parking"},  # covers ANE PARK POINT (if it's parking)

    # -------------------
    # Shopping
    # -------------------
    {"keyword": "AMAZON", "category": "Shopping"},
    {"keyword": "NOON", "category": "Shopping"},
    {"keyword": "IKEA", "category": "Shopping"},
    {"keyword": "ACE", "category": "Shopping"},
    {"keyword": "FLYING TIGER", "category": "Shopping"},
    {"keyword": "LIFESTYLE CHOICE", "category": "Shopping"},
    {"keyword": "SUN AND SAND SPORTS", "category": "Shopping"},
    {"keyword": "OKA CERAMICS", "category": "Shopping"},
    {"keyword": "OUNASS", "category": "Shopping"},
    {"keyword": "OYSHO", "category": "Shopping"},
    {"keyword": "POTTERY BARN", "category": "Shopping"},
    {"keyword": "THEOUTNET", "category": "Shopping"},
    {"keyword": "AL FUTTAIM TRD CO", "category": "Shopping"},
    {"keyword": "VISIQUE OPTICAL", "category": "Shopping"},
    {"keyword": "STITCH IN TIME", "category": "Shopping"},
    {"keyword": "ONTHELIST", "category": "Shopping"},
    {"keyword": "SWATCH", "category": "Shopping"},
    {"keyword": "JACADI", "category": "Shopping"},  # you had it as Other before; setting as Shopping is more consistent

    # -------------------
    # Beauty
    # -------------------
    {"keyword": "SEPHORA", "category": "Beauty"},
    {"keyword": "FACES", "category": "Beauty"},
    {"keyword": "TIPS & TOES", "category": "Beauty"},
    {"keyword": "BEAUTY", "category": "Beauty"},
    {"keyword": "THE BODY SHOP", "category": "Beauty"},
    {"keyword": "KIKO", "category": "Beauty"},
    {"keyword": "MAC", "category": "Beauty"},
    {"keyword": "BENEFIT BOUTIQUE", "category": "Beauty"},

    # -------------------
    # Fitness
    # -------------------
    {"keyword": "FITNESS FIRST", "category": "Fitness"},
    {"keyword": "CLASS PASS", "category": "Fitness"},
    {"keyword": "CLASSPASS", "category": "Fitness"},
    {"keyword": "GYM", "category": "Fitness"},
    {"keyword": "DECATHLON", "category": "Fitness"},

    # -------------------
    # Utilities & Bills
    # -------------------
    {"keyword": "DEWA", "category": "Utilities & Bills"},
    {"keyword": "DUBAI ELECTRICITY", "category": "Utilities & Bills"},
    {"keyword": "URBAN COMPANY", "category": "Utilities & Bills"},
    {"keyword": "NESTLE WATERS", "category": "Utilities & Bills"},
    {"keyword": "ENOC", "category": "Utilities & Bills"},
    {"keyword": "ADNOC", "category": "Utilities & Bills"},
    {"keyword": "LOOTAH BC GAS", "category": "Utilities & Bills"},
    {"keyword": "ETISALAT", "category": "Utilities & Bills"},
    {"keyword": "DU", "category": "Utilities & Bills"},  # safe-matched as a whole token, not inside DUBAI
    {"keyword": "SMART DUBAI", "category": "Utilities & Bills"},
    {"keyword": "ABU DHABI POLICE", "category": "Utilities & Bills"},
    {"keyword": "EPPCO SITE 1076", "category": "Utilities & Bills"},  # your example
    {"keyword": "DSC COMMERCIAL", "category": "Utilities & Bills"},  # building fees often fall here

    # -------------------
    # Travel
    # -------------------
    {"keyword": "HOTEL", "category": "Travel"},  # anything with HOTEL => Travel
    {"keyword": "MARRIOTT", "category": "Travel"},
    {"keyword": "WESTIN", "category": "Travel"},
    {"keyword": "POINTS.COM", "category": "Travel"},
    {"keyword": "EMIRATES", "category": "Travel"},
    {"keyword": "AIRALO", "category": "Travel"},

    # -------------------
    # Pharmacy
    # -------------------
    {"keyword": "ACACIA COMMUNITY", "category": "Pharmacy"},
    {"keyword": "PHARM", "category": "Pharmacy"},  # catches PHAR / PHARMACY

    # -------------------
    # Subscriptions
    # -------------------
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "AMAZON PRIME", "category": "Subscriptions"},
    {"keyword": "APPLE", "category": "Subscriptions"},
    {"keyword": "GOOGLE", "category": "Subscriptions"},
]

# ---------------------------
# Input loader: 2 columns (Date + Details) — keep ALL rows + preserve order
# ---------------------------

def read_two_col_export(file) -> pd.DataFrame:
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

    # Optional header detection
    first0 = str(col0.iloc[0]).strip().lower() if len(col0) else ""
    first1 = str(col1.iloc[0]).strip().lower() if len(col1) else ""
    has_header = ("date" in first0) and ("detail" in first1 or "description" in first1)

    if has_header:
        col0 = col0.iloc[1:].reset_index(drop=True)
        col1 = col1.iloc[1:].reset_index(drop=True)
    else:
        col0 = col0.reset_index(drop=True)
        col1 = col1.reset_index(drop=True)

    df = pd.DataFrame(
        {
            "row_id": range(len(col0)),  # preserves original order within file
            "date_raw": col0,
            "details": col1.astype(str).fillna("").str.strip(),
        }
    )

    # Parse date (do not drop rows)
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)

    # Row status flags
    df["row_status"] = "OK"
    df.loc[df["date"].isna(), "row_status"] = "INVALID_DATE"
    df.loc[df["details"].str.strip().eq(""), "row_status"] = "EMPTY_DETAILS"

    # Optional month (blank if invalid date)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df

# ---------------------------
# Matching fix: short keywords must match whole tokens (DU != DUBAI)
# ---------------------------

def keyword_to_regex(kw: str) -> str:
    """
    If keyword is short (1-3 chars alphanumeric), match as a whole token.
    Example: DU matches ' DU ' but NOT 'DUBAI'.
    Otherwise, safe substring match.
    """
    kw = kw.strip().upper()
    if re.fullmatch(r"[A-Z0-9]{1,3}", kw):
        return rf"(?<![A-Z0-9]){re.escape(kw)}(?![A-Z0-9])"
    return re.escape(kw)

# ---------------------------
# Categorisation: keyword match, fill empty only, OK rows only
# ---------------------------

def apply_rules_fill_empty_only(df: pd.DataFrame, rules_df: pd.DataFrame, default_category: str = "Other") -> pd.DataFrame:
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = ""

    ok_mask = df["row_status"].eq("OK")

    desc = df["details"].fillna("").astype(str).str.upper()
    cats = df["category"].fillna("").astype(str)

    empty_mask = cats.str.strip().eq("")
    target = ok_mask & empty_mask

    result = cats.copy()
    result.loc[target] = default_category

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue

        pattern = keyword_to_regex(kw)
        hit = desc.str.contains(pattern, na=False, regex=True)
        result.loc[target & hit] = cat

    df["category"] = result
    return df

# ---------------------------
# Export: download exactly what is shown
# ---------------------------

def build_exact_export_xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Categorised data")
    return buf.getvalue()

# ---------------------------
# App
# ---------------------------

def main():
    st.title("💳 Spend Tracker — Categorise Expenses")

    # Session state
    if "rules_df" not in st.session_state:
        st.session_state.rules_df = pd.DataFrame(DEFAULT_RULES)
    if "df_all" not in st.session_state:
        st.session_state.df_all = None

    st.sidebar.header("1) Upload")
    files = st.sidebar.file_uploader(
        "XLSX/CSV with 2 columns: Date + Details",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )

    st.sidebar.header("2) Rules")
    if st.sidebar.button("Reset rules to default"):
        st.session_state.rules_df = pd.DataFrame(DEFAULT_RULES)
        st.sidebar.success("Rules reset ✅")

    if st.sidebar.button("Load & combine data", type="primary"):
        if not files:
            st.sidebar.warning("Upload at least one file.")
        else:
            dfs = []
            for f in files:
                df_part = read_two_col_export(f)
                df_part["source_file"] = f.name
                dfs.append(df_part)

            df_all = pd.concat(dfs, ignore_index=True)

            # Preserve file upload order + original row order (stable)
            file_order_map = {f.name: i for i, f in enumerate(files)}
            df_all["file_order"] = df_all["source_file"].map(file_order_map)
            df_all = df_all.sort_values(["file_order", "row_id"], kind="stable").reset_index(drop=True)

            df_all["category"] = ""
            st.session_state.df_all = df_all

            counts = df_all["row_status"].value_counts(dropna=False).to_dict()
            st.sidebar.success(f"Loaded {len(df_all)} rows ✅")
            st.caption(f"Row status counts: {counts}")

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload file(s) and click **Load & combine data**.")
        return

    st.subheader("Rules (auto-categorisation)")
    rules_editor = st.data_editor(
        st.session_state.rules_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "keyword": st.column_config.TextColumn("Keyword (contains)"),
            "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
        },
        key="rules_editor",
    )

    # Validate category values before apply
    invalid = rules_editor[~rules_editor["category"].isin(CATEGORIES)]
    if not invalid.empty:
        st.error("Some rules have invalid categories. Fix them or click **Reset rules to default**.")
        st.dataframe(invalid, use_container_width=True)

    if st.button("⚙️ Apply rules"):
        if not invalid.empty:
            st.stop()
        st.session_state.rules_df = rules_editor
        st.session_state.df_all = apply_rules_fill_empty_only(
            st.session_state.df_all,
            rules_editor,
            default_category="Other",
        )
        st.success("Rules applied ✅")

    df_all = st.session_state.df_all

    st.subheader("Categorised data (exact input order)")
    st.dataframe(df_all, use_container_width=True)

    st.subheader("Download")
    export_bytes = build_exact_export_xlsx(df_all)
    st.download_button(
        "⬇️ Download XLSX (exactly as shown)",
        data=export_bytes,
        file_name="categorised_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption(
        "No rows are dropped. Rows flagged INVALID_DATE / EMPTY_DETAILS are kept. "
        "Categorisation applies only to rows with row_status = OK. "
        "Short keywords like 'DU' match whole tokens, so they won't match 'DUBAI'."
    )

if __name__ == "__main__":
    main()
