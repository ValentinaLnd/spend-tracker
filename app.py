import io
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spend Tracker — Categorise", layout="wide")


# ---------------------------
# Fixed approved categories + default rules (stable)
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
    {"keyword": "PARKONIC", "category": "Parking"},

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
    {"keyword": "CLASSPASS", "category": "Fitness"},
    {"keyword": "GYM", "category": "Fitness"},
    {"keyword": "DECATHLON", "category": "Fitness"},

    # Utilities & Bills
    {"keyword": "DEWA", "category": "Utilities & Bills"},
    {"keyword": "DUBAI ELECTRICITY", "category": "Utilities & Bills"},
    {"keyword": "ETISALAT", "category": "Utilities & Bills"},
    {"keyword": "DU", "category": "Utilities & Bills"},

    # Subscriptions
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
            "row_id": range(len(col0)),
            "date_raw": col0,
            "details": col1.astype(str).fillna("").str.strip(),
        }
    )

    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)

    df["row_status"] = "OK"
    df.loc[df["date"].isna(), "row_status"] = "INVALID_DATE"
    df.loc[df["details"].str.strip().eq(""), "row_status"] = "EMPTY_DETAILS"

    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# ---------------------------
# Categorisation: keyword contains, fill empty only, OK rows only
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
        hit = desc.str.contains(re.escape(kw), na=False)
        result.loc[target & hit] = cat

    df["category"] = result
    return df


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

            # Preserve upload order + original row order (stable)
            file_order_map = {f.name: i for i, f in enumerate(files)}
            df_all["file_order"] = df_all["source_file"].map(file_order_map)
            df_all = df_all.sort_values(["file_order", "row_id"], kind="stable").reset_index(drop=True)

            df_all["category"] = ""
            st.session_state.df_all = df_all
            st.sidebar.success(f"Loaded {len(df_all)} rows ✅")

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload file(s) and click **Load & combine data**.")
        return

    # Rules editor (fixed category options; prevents mismatch)
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

    # Validate rules before applying (prevents silent “everything became Utilities”)
    invalid = rules_editor[~rules_editor["category"].isin(CATEGORIES)]
    if not invalid.empty:
        st.error("Some rules have invalid categories. Fix them or click **Reset rules to default**.")
        st.dataframe(invalid, use_container_width=True)

    if st.button("⚙️ Apply rules"):
        # If user somehow introduced invalid category values, block apply
        if not invalid.empty:
            st.stop()

        st.session_state.rules_df = rules_editor
        st.session_state.df_all = apply_rules_fill_empty_only(st.session_state.df_all, rules_editor, default_category="Other")
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


if __name__ == "__main__":
    main()
