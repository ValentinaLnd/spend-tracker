import streamlit as st
import pandas as pd
from pathlib import Path
import pdfplumber

st.set_page_config(page_title="Spending Tracker", layout="wide")

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

# -------- CATEGORIES & RULES --------

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
    {"keyword": "JBS RESTAURANT", "category": "Restaurants & Cafes"},

    # Transport
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "RTA", "category": "Transport"},
    {"keyword": "TAXI", "category": "Transport"},
    {"keyword": "VALTRANS", "category": "Transport"},
    {"keyword": "SALIK", "category": "Transport"},

    # Shopping
    {"keyword": "AMAZON", "category": "Shopping"},
    {"keyword": "NOON", "category": "Shopping"},
    {"keyword": "DUBAI MALL", "category": "Shopping"},
    {"keyword": "MALL OF THE EMIRATES", "category": "Shopping"},
    {"keyword": "IKEA", "category": "Shopping"},
    {"keyword": "ACE", "category": "Shopping"},

    # Beauty
    {"keyword": "SEPHORA", "category": "Beauty"},
    {"keyword": "FACES", "category": "Beauty"},
    {"keyword": "COSMETICS", "category": "Beauty"},
    {"keyword": "THE BODY SHOP", "category": "Beauty"},
    {"keyword": "KIKO", "category": "Beauty"},
    {"keyword": "MAC COSMETICS", "category": "Beauty"},

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
    {"keyword": "URBAN COMPANY", "category": "Utilities & Bills"},

    # Subscriptions
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "APPLE", "category": "Subscriptions"},
    {"keyword": "GOOGLE", "category": "Subscriptions"},
    {"keyword": "AMAZON PRIME", "category": "Subscriptions"},

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


# -------- HELPERS TO READ STATEMENTS --------

def pdf_to_df(file) -> pd.DataFrame:
    """
    Extract tables from a PDF into a DataFrame.
    This assumes the PDF has real text tables (not scanned images).
    """
    dfs = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                rows = table[1:]
                df_page = pd.DataFrame(rows, columns=header)
                dfs.append(df_page)

    if not dfs:
        raise ValueError("No tables found in PDF. The statement format may not be supported.")
    return pd.concat(dfs, ignore_index=True)


def normalize_statement_df(df_raw: pd.DataFrame, account_name: str) -> pd.DataFrame:
    """
    Try to detect date / description / amount in a raw statement DataFrame.
    Supports:
    - single 'amount' column, or
    - 'debit'/'credit' columns (builds signed amount).
    """
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c), None)
    desc_col = next(
        (c for c in df.columns
         if c.startswith("descr") or "details" in c or "narration" in c or "description" in c),
        None,
    )
    amount_col = next((c for c in df.columns if "amount" in c or "amt" in c), None)
    debit_col = next((c for c in df.columns if "debit" in c), None)
    credit_col = next((c for c in df.columns if "credit" in c), None)

    if amount_col is None and (debit_col is not None or credit_col is not None):
        # Build signed amount from debit / credit
        df["__amount_tmp_debit"] = pd.to_numeric(df.get(debit_col), errors="coerce").fillna(0)
        df["__amount_tmp_credit"] = pd.to_numeric(df.get(credit_col), errors="coerce").fillna(0)
        df["amount"] = df["__amount_tmp_credit"] - df["__amount_tmp_debit"]
        amount_col = "amount"

    if not all([date_col, desc_col, amount_col]):
        raise ValueError(
            f"Could not detect date/description/amount columns. "
            f"Detected date: {date_col}, description: {desc_col}, amount: {amount_col}"
        )

    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "description": df[desc_col].astype(str),
            "amount": pd.to_numeric(df[amount_col], errors="coerce"),
        }
    )
    tmp["account"] = account_name
    tmp = tmp.dropna(subset=["date", "amount"])
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    tmp["year"] = tmp["date"].dt.year
    return tmp


def load_and_clean_any(files, account_name: str) -> pd.DataFrame:
    """
    Load statements that can be CSV or PDF.
    """
    dfs = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".csv"):
            df_raw = pd.read_csv(f)
        elif name.endswith(".pdf"):
            df_raw = pdf_to_df(f)
        else:
            raise ValueError(f"Unsupported file type for {f.name}. Only CSV and PDF are accepted.")
        tmp = normalize_statement_df(df_raw, account_name)
        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)


def apply_rules(df: pd.DataFrame, rules_df: pd.DataFrame, default_category: str = "Other") -> pd.DataFrame:
    """
    Apply keyword rules to descriptions.
    Anything not matching any rule stays as `default_category` (Other).
    """
    df = df.copy()
    desc = df["description"].fillna("").str.upper()
    categories = pd.Series(default_category, index=df.index)

    for _, row in rules_df.iterrows():
        kw = str(row.get("keyword", "")).strip().upper()
        cat = str(row.get("category", "")).strip()
        if not kw or not cat:
            continue
        mask = desc.str.contains(kw, na=False)
        categories[mask] = cat

    df["category"] = categories
    return df


# -------- MAIN APP --------

def main():
    st.title("💳 12-Month Spending Tracker (PDF & CSV, Auto-categorized)")

    # --- Sidebar: upload + config ---
    st.sidebar.header("1. Upload bank statements")
    st.sidebar.write("Upload CSV or PDF statements for each account.")

    acc1_files = st.sidebar.file_uploader(
        "Account 1 files (CSV or PDF)",
        type=["csv", "pdf"],
        accept_multiple_files=True,
        key="acc1",
    )
    acc2_files = st.sidebar.file_uploader(
        "Account 2 files (CSV or PDF)",
        type=["csv", "pdf"],
        accept_multiple_files=True,
        key="acc2",
    )

    account1_name = st.sidebar.text_input("Account 1 name", value="Account 1")
    account2_name = st.sidebar.text_input("Account 2 name", value="Account 2")

    st.sidebar.header("2. Categories")
    categories_text = st.sidebar.text_area(
        "Edit categories (one per line)",
        value="\n".join(DEFAULT_CATEGORIES),
        height=200,
    )
    category_list = [c.strip() for c in categories_text.splitlines() if c.strip()]

    st.sidebar.header("3. Keyword rules")
    st.sidebar.write("Each rule: if description contains keyword, assign category.")

    if "rules_df" not in st.session_state:
        st.session_state.rules_df = pd.DataFrame(DEFAULT_RULES)

    rules_df = st.session_state.rules_df

    if "df_all" not in st.session_state:
        st.session_state.df_all = None

    # --- Load and combine data ---
    if st.button("Load & combine data", type="primary"):
        if not acc1_files and not acc2_files:
            st.warning("Please upload at least one CSV or PDF file.")
        else:
            try:
                all_dfs = []
                if acc1_files:
                    all_dfs.append(load_and_clean_any(acc1_files, account1_name))
                if acc2_files:
                    all_dfs.append(load_and_clean_any(acc2_files, account2_name))
                df_all = pd.concat(all_dfs, ignore_index=True)
                df_all = df_all.sort_values("date")
                df_all["category"] = "Other"
                st.session_state.df_all = df_all
            except Exception as e:
                st.error(f"Error loading statements: {e}")
                return

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload your CSV/PDF files and click **Load & combine data** to get started.")
        return

    # --- Step 1: rules editor ---
    st.subheader("Step 1 – Define / update auto-categorisation rules")
    rules_editor = st.data_editor(
        rules_df,
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

    col_apply1, col_apply2 = st.columns([1, 2])
    with col_apply1:
        if st.button("⚙️ Apply rules and categorize"):
            st.session_state.rules_df = rules_editor
            df_all = apply_rules(st.session_state.df_all, rules_editor, default_category="Other")
            st.session_state.df_all = df_all
            st.success("Rules applied. Transactions have been auto-categorized.")

    with col_apply2:
        if st.button("💾 Save categorized data to CSV"):
            if st.session_state.df_all is None:
                st.warning("No data to save yet.")
            else:
                output_file = DATA_PATH / "transactions_categorized.csv"
                st.session_state.df_all.to_csv(output_file, index=False)
                st.success(f"Saved to {output_file.resolve()}")

    df_all = st.session_state.df_all
    if df_all is None or df_all.empty:
        st.warning("No data to analyze yet.")
        return

    # --- Step 2: filters + KPIs ---
    st.subheader("Step 2 – Overview and filters")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_date = df_all["date"].min()
        max_date = df_all["date"].max()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with col2:
        selected_accounts = st.multiselect(
            "Accounts",
            sorted(df_all["account"].unique()),
            default=list(sorted(df_all["account"].unique())),
        )
    with col3:
        selected_categories = st.multiselect(
            "Categories",
            sorted(df_all["category"].unique()),
            default=list(sorted(df_all["category"].unique())),
        )

    filtered = df_all.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["date"] >= pd.to_datetime(start))
            & (filtered["date"] <= pd.to_datetime(end))
        ]
    if selected_accounts:
        filtered = filtered[filtered["account"].isin(selected_accounts)]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    total_spend = filtered.loc[filtered["amount"] < 0, "amount"].sum()
    total_income = filtered.loc[filtered["amount"] > 0, "amount"].sum()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total spend", f"{total_spend:,.2f}")
    with m2:
        st.metric("Total income", f"{total_income:,.2f}")
    with m3:
        st.metric("Net", f"{(total_income + total_spend):,.2f}")

    # --- Step 3: trends ---
    st.subheader("Step 3 – Trends")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Monthly spend by category**")
        monthly_cat = (
            filtered[filtered["amount"] < 0]
            .assign(amount=lambda x: x["amount"].abs())
            .groupby(["month", "category"], as_index=False)["amount"]
            .sum()
        )
        if not monthly_cat.empty:
            pivot = monthly_cat.pivot(index="month", columns="category", values="amount").fillna(0)
            st.line_chart(pivot)
        else:
            st.write("No spending data in this period.")

    with col_right:
        st.markdown("**Top categories by spend**")
        cat_totals = (
            filtered[filtered["amount"] < 0]
            .assign(amount=lambda x: x["amount"].abs())
            .groupby("category", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
        )
        if not cat_totals.empty:
            st.bar_chart(cat_totals.set_index("category"))
        else:
            st.write("No spending data in this period.")

    # --- Raw data ---
    st.subheader("Raw filtered data (auto-categorized)")
    st.dataframe(filtered.sort_values("date"), use_container_width=True)


if __name__ == "__main__":
    main()
