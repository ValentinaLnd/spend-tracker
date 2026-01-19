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
# Input loader: 2 columns (Date + Details) — KEEP ALL ROWS
# ---------------------------

def read_two_col_export(file) -> pd.DataFrame:
    """
    Supports XLSX/CSV with:
      - Column A: Date
      - Column B: Details
    Headers optional.

    IMPORTANT:
      - Does NOT drop rows.
      - Adds row_id to preserve original order.
      - Adds row_status to explain non-transaction rows.
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

    # Detect a header row like: "Date" | "Details"
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
            "row_id": range(len(col0)),       # preserves original order
            "date_raw": col0,                 # keep original value for debugging
            "details": col1.astype(str).fillna("").str.strip(),
        }
    )

    # Parse date (but do NOT drop rows)
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)

    # Flag row issues
    df["row_status"] = "OK"
    df.loc[df["date"].isna(), "row_status"] = "INVALID_DATE"
    df.loc[df["details"].str.strip().eq(""), "row_status"] = "EMPTY_DETAILS"

    # Optional month (blank if invalid date)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df


# ---------------------------
# Categorisation (keyword contains, fill empty only, ONLY for OK rows)
# ---------------------------

def apply_rules_fill_empty_only(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    default_category: str = "Other",
) -> pd.DataFrame:
    df = df.copy()

    if "category" not in df.columns:
        df["category"] = ""

    ok_mask = df.get("row_status", "OK").eq("OK")

    desc = df["details"].fillna("").astype(str).str.upper()
    cats = df["category"].fillna("").astype(str)

    empty_mask = cats.str.strip() == ""
    target_mask = ok_mask & empty_mask

    result = cats.copy()
    result.loc[target_mask] = default_category

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue
        hit = desc.str.contains(re.escape(kw), na=False)
        result.loc[target_mask & hit] = cat

    df["category"] = result
    return df


# ---------------------------
# Export: download exactly what is shown (same order)
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

    st.sidebar.header("1) Upload your file(s)")
    st.sidebar.write("Upload XLSX or CSV with **2 columns**: **Date** + **Details** (headers optional).")

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
                dfs = []
                for f in files:
                    df_part = read_two_col_export(f)
                    df_part["source_file"] = f.name  # helps trace issues
                    dfs.append(df_part)

                df_all = pd.concat(dfs, ignore_index=True)

                # Preserve exact input order across multiple files:
                # first by file upload order, then by original row_id
                df_all["file_order"] = df_all["source_file"].map({f.name: i for i, f in enumerate(files)})
                df_all = df_all.sort_values(["file_order", "row_id"], kind="stable").reset_index(drop=True)

                df_all["category"] = ""
                st.session_state.df_all = df_all

                st.success(f"Loaded {len(df_all)} rows ✅")

                # Quick transparency so you see why lines may be non-OK
                counts = df_all["row_status"].value_counts(dropna=False).to_dict()
                st.caption(f"Row status counts: {counts}")

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
            "category": st.column_config.SelectboxColumn("Category", options=category_list),
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

    st.subheader("Categorised data (same order as input)")
    # Keep exact preserved order
    df_view = df_all.copy()
    st.dataframe(df_view, use_container_width=True)

    st.subheader("Download")
    export_bytes = build_exact_export_xlsx(df_view)
    st.download_button(
        label="⬇️ Download XLSX (exactly as shown)",
        data=export_bytes,
        file_name="categorised_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption(
        "Notes: No rows are dropped. Rows with INVALID_DATE / EMPTY_DETAILS are kept and shown as-is; "
        "categorisation applies only to rows with row_status = OK."
    )


if __name__ == "__main__":
    main()
