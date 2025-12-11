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
    "Rent / Mortgage",
    "Utilities & Bills",
    "Subscriptions",
    "Healthcare",
    "Travel",
    "Income",
    "Savings / Investments",
    "Other",
]

# A few example rules – adjust to your own merchants later
DEFAULT_RULES = [
    {"keyword": "CARREFOUR", "category": "Groceries"},
    {"keyword": "SPINNEYS", "category": "Groceries"},
    {"keyword": "DELIVEROO", "category": "Restaurants & Cafes"},
    {"keyword": "TALABAT", "category": "Restaurants & Cafes"},
    {"keyword": "UBER", "category": "Transport"},
    {"keyword": "CAREEM", "category": "Transport"},
    {"keyword": "NETFLIX", "category": "Subscriptions"},
    {"keyword": "SPOTIFY", "category": "Subscriptions"},
    {"keyword": "DEWA", "category": "Utilities & Bills"},
    {"keyword": "SALIK", "category": "Transport"},
]


def load_and_clean(files, account_name: str):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in df.columns if "date" in c), None)
        desc_col = next(
            (c for c in df.columns
             if c.startswith("descr") or "details" in c or "narration" in c),
            None,
        )
        amount_col = next((c for c in df.columns if "amount" in c or "amt" in c), None)

        if not all([date_col, desc_col, amount_col]):
            st.error(f"Could not detect date/description/amount columns in file {f.name}.")
            st.stop()

        tmp = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "description": df[desc_col].astype(str),
                "amount": pd.to_numeric(df[amount_col], errors="coerce"),
            }
        )
        tmp["account"] = account_name
        dfs.append(tmp)

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["date", "amount"])
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["year"] = out["date"].dt.year
    return out


def apply_rules(df: pd.DataFrame, rules_df: pd.DataFrame, default_category: str = "Uncategorized") -> pd.DataFrame:
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


def main():
    st.title("💳 12-Month Spending Tracker (Auto-categorized)")

    st.sidebar.header("1. Upload bank statements")

    acc1_files = st.sidebar.file_uploader(
        "Account 1 CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="acc1",
    )
    acc2_files = st.sidebar.file_uploader(
        "Account 2 CSV files",
        type=["csv"],
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
    st.sidebar.write("Each rule: if description contains **keyword**, assign **category**.")

    if "rules_df" not in st.session_state:
        st.session_state.rules_df = pd.DataFrame(DEFAULT_RULES)

    rules_df = st.session_state.rules_df

    if "df_all" not in st.session_state:
        st.session_state.df_all = None

    if st.button("Load & combine data", type="primary"):
        if not acc1_files and not acc2_files:
            st.warning("Please upload at least one CSV file.")
        else:
            all_dfs = []
            if acc1_files:
                all_dfs.append(load_and_clean(acc1_files, account1_name))
            if acc2_files:
                all_dfs.append(load_and_clean(acc2_files, account2_name))
            df_all = pd.concat(all_dfs, ignore_index=True)
            df_all = df_all.sort_values("date")
            df_all["category"] = "Uncategorized"
            st.session_state.df_all = df_all

    df_all = st.session_state.df_all
    if df_all is None:
        st.info("Upload your CSV files and click **Load & combine data** to get started.")
        return

    st.subheader("Step 1 – Define / update auto-categorisation rules")

    st.markdown(
        "Manage the rules used to auto-tag transactions. "
        "Example: keyword `CARREFOUR` + category `Groceries` → any description containing "
        "`CARREFOUR` will be tagged as Groceries."
    )

    rules_editor = st.data_editor(
        rules_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "keyword": st.column_config.TextColumn(
                "Keyword (contains)",
                help="Text to search in the transaction description (case-insensitive).",
            ),
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=category_list,
                help="Category to assign when the keyword matches.",
            ),
        },
        key="rules_editor",
    )

    col_apply1, col_apply2 = st.columns([1, 2])
    with col_apply1:
        if st.button("⚙️ Apply rules and categorize"):
            st.session_state.rules_df = rules_editor
            df_all = apply_rules(st.session_state.df_all, rules_editor, default_category="Uncategorized")
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
            pivot = (
                monthly_cat.pivot(index="month", columns="category", values="amount")
                .fillna(0)
            )
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

    st.subheader("Raw filtered data (auto-categorized)")
    st.dataframe(filtered.sort_values("date"), use_container_width=True)


if __name__ == "__main__":
    main()
