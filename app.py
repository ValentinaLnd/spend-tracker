import streamlit as st
import pandas as pd
from pathlib import Path

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
    {"keyword": "SPINNEYS", "category": "Groceries"},
    {"keyword": "CARREFOUR", "category": "Groceries"},
    {"keyword": "DELIVEROO", "category": "Restaurants & Cafes"},
    {"keyword": "TALABAT", "category": "Restaurants & Cafes"},
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "RTA", "category": "Transport"},
    {"keyword": "SALIK", "category": "Transport"},
    {"keyword": "AMAZON", "category": "Shopping"},
    {"keyword": "NOON", "category": "Shopping"},
    {"keyword": "SEPHORA", "category": "Beauty"},
    {"keyword": "FACES", "category": "Beauty"},
    {"keyword": "FITNESS FIRST", "category": "Fitness"},
    {"keyword": "CLASS PASS", "category": "Fitness"},
    {"keyword": "DEWA", "category": "Utilities & Bills"},
    {"keyword": "DU", "category": "Utilities & Bills"},
    {"keyword": "ETISALAT", "category": "Utilities & Bills"},
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "EMIRATES", "category": "Travel"},
]

def try_read_first_file(files):
    if not files:
        return None
    f = files[0]
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(f, engine="openpyxl")
    return None

def normalize_statement_df(
    df_raw: pd.DataFrame,
    account_name: str,
    date_col: str | None,
    desc_col: str | None,
    amount_col: str | None,
    debit_col: str | None,
    credit_col: str | None,
) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def val_or_none(x):
        return None if (x is None or x == "(auto)") else x

    date_col = val_or_none(date_col)
    desc_col = val_or_none(desc_col)
    amount_col = val_or_none(amount_col)
    debit_col = val_or_none(debit_col)
    credit_col = val_or_none(credit_col)

    # If user selected debit/credit, build a signed amount
    if amount_col is None and (debit_col or credit_col):
        debit_series = pd.to_numeric(df[debit_col], errors="coerce").fillna(0) if debit_col else 0
        credit_series = pd.to_numeric(df[credit_col], errors="coerce").fillna(0) if credit_col else 0
        df["__amount"] = credit_series - debit_series
        amount_col = "__amount"

    if not all([date_col, desc_col, amount_col]):
        raise ValueError(
            "Column mapping is required for your file.\n"
            f"Columns found: {list(df.columns)}\n"
            "Please select Date + Description + (Amount OR Debit/Credit) from the sidebar."
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

def load_and_clean(files, account_name, mapping):
    dfs = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".csv"):
            df_raw = pd.read_csv(f)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df_raw = pd.read_excel(f, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {f.name}")

        tmp = normalize_statement_df(
            df_raw,
            account_name,
            mapping["date_col"],
            mapping["desc_col"],
            mapping["amount_col"],
            mapping["debit_col"],
            mapping["credit_col"],
        )
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)

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

def main():
    st.title("💳 12-Month Spending Tracker (Excel/CSV, Auto-categorized)")

    st.sidebar.header("1. Upload files")
    acc1_files = st.sidebar.file_uploader("Account 1 (CSV/XLSX)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    acc2_files = st.sidebar.file_uploader("Account 2 (CSV/XLSX)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

    account1_name = st.sidebar.text_input("Account 1 name", value="Account 1")
    account2_name = st.sidebar.text_input("Account 2 name", value="Account 2")

    st.sidebar.header("0. Column mapping (required for your exports)")
    sample_df = try_read_first_file(acc1_files or acc2_files)
    sample_cols = list(sample_df.columns) if sample_df is not None else []

    date_col = st.sidebar.selectbox("Date column", ["(auto)"] + [str(c) for c in sample_cols])
    desc_col = st.sidebar.selectbox("Description column", ["(auto)"] + [str(c) for c in sample_cols])

    mode = st.sidebar.radio("Amount mode", ["Single Amount column", "Debit + Credit columns"])
    amount_col = "(auto)"
    debit_col = "(auto)"
    credit_col = "(auto)"
    if mode == "Single Amount column":
        amount_col = st.sidebar.selectbox("Amount column", ["(auto)"] + [str(c) for c in sample_cols])
    else:
        debit_col = st.sidebar.selectbox("Debit column", ["(auto)"] + [str(c) for c in sample_cols])
        credit_col = st.sidebar.selectbox("Credit column", ["(auto)"] + [str(c) for c in sample_cols])

    mapping = {
        "date_col": date_col,
        "desc_col": desc_col,
        "amount_col": amount_col if mode == "Single Amount column" else "(auto)",
        "debit_col": debit_col if mode != "Single Amount column" else "(auto)",
        "credit_col": credit_col if mode != "Single Amount column" else "(auto)",
    }

    st.sidebar.header("2. Categories")
    categories_text = st.sidebar.text_area("Categories (one per line)", value="\n".join(DEFAULT_CATEGORIES), height=220)
    category_list = [c.strip() for c in categories_text.splitlines() if c.strip()]

    st.sidebar.header("3. Rules")
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
        except Exception as e:
            st.error(f"Error loading statements: {e}")
            return

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload files, choose column mapping in the sidebar, then click **Load & combine data**.")
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
    )

    if st.button("⚙️ Apply rules and categorize"):
        st.session_state.rules_df = rules_editor
        st.session_state.df_all = apply_rules(st.session_state.df_all, rules_editor, default_category="Other")
        st.success("Rules applied.")

    df_all = st.session_state.df_all

    st.subheader("Step 2 – Trends")
    monthly = (
        df_all[df_all["amount"] < 0]
        .assign(spend=lambda x: x["amount"].abs())
        .groupby(["month", "category"], as_index=False)["spend"]
        .sum()
        .pivot(index="month", columns="category", values="spend")
        .fillna(0)
    )
    st.line_chart(monthly)

    st.subheader("Raw data")
    st.dataframe(df_all.sort_values("date"), use_container_width=True)

if __name__ == "__main__":
    main()
