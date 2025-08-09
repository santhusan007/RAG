from __future__ import annotations
import re
import pandas as pd


def df_profile_snippet(df: pd.DataFrame, max_cols: int = 12, max_rows: int = 6) -> str:
    parts = []
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols = list(df.columns)[:max_cols]
    parts.append("Columns: " + ", ".join(str(c) for c in cols) + (" …" if df.shape[1] > max_cols else ""))
    miss = df.isna().mean().sort_values(ascending=False)[:8]
    miss_txt = ", ".join([f"{c}: {v:.0%}" for c, v in miss.items()]) if not miss.empty else "(none)"
    parts.append("Missing (top): " + miss_txt)
    sample = df.head(max_rows).to_csv(index=False)
    parts.append("Sample (CSV):\n" + sample)
    return "\n".join(parts)


def analyze_spreadsheet(df: pd.DataFrame) -> str:
    try:
        desc = df.describe(include="all").transpose().head(8).fillna("")
        lines = ["Quick insights:"]
        lines.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        lines.append(f"- Numeric columns: {len(num_cols)}; Categorical columns: {len(cat_cols)}")
        if cat_cols:
            top = df[cat_cols[0]].astype(str).value_counts().head(5)
            lines.append(f"- Top values in {cat_cols[0]}: " + ", ".join([f"{k} ({v})" for k, v in top.items()]))
        return "\n".join(lines)
    except Exception:
        return "(Could not derive insights)"


def answer_dataset_question(df: pd.DataFrame, question: str) -> str | None:
    """Heuristic NL -> pandas answers using the full DataFrame.

    Handles common tasks:
    - row/column counts
    - mean/avg, sum/total, min, max of a numeric column
    - most frequent/mode of a categorical column
    - count/distribution by a categorical column (top 10)
    - top N <group> by <metric> (e.g., top 5 sub-category by sales)
    """
    try:
        q_raw = question or ""
        q = q_raw.lower()

        # 1) Basic shape
        if ("rows" in q and ("how many" in q or "count" in q)) or re.search(r"\brow count\b", q):
            return f"Rows: {df.shape[0]}"
        if ("columns" in q or "cols" in q) and ("how many" in q or "count" in q):
            return f"Columns: {df.shape[1]}"

        # Helpers for column matching
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        col_map = {c: _norm(str(c)) for c in df.columns}

        def _find_col(name: str, prefer_numeric: bool | None = None) -> str | None:
            tokens = [t for t in _norm(name).split() if t]
            if not tokens:
                return None
            best, best_score = None, -1
            for c, cnorm in col_map.items():
                score = 0
                for t in tokens:
                    if t in cnorm:
                        score += 1
                if cnorm.endswith(tokens[-1]):
                    score += 0.5
                if prefer_numeric is True and pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                if prefer_numeric is False and not pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                if score > best_score:
                    best, best_score = c, score
            return best

        def _parse_n(text: str, default: int = 5) -> int:
            m = re.search(r"\btop\s*(\d{1,3})\b", text)
            if m:
                try:
                    return max(1, int(m.group(1)))
                except Exception:
                    pass
            return default

        # Choose metric/group columns by synonyms
        def _choose_metric(prefer: list[str] | None = None) -> str | None:
            prefer = prefer or ["sales", "revenue", "amount", "net sales", "qty", "quantity", "units"]
            best, best_score = None, -1.0
            for c, cnorm in col_map.items():
                score = 0.0
                for i, key in enumerate(prefer[::-1]):
                    if key in cnorm:
                        score += 1.0 + i * 0.1
                if pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.5
                if score > best_score:
                    best, best_score = c, score
            return best

        def _choose_group(prefer: list[str] | None = None) -> str | None:
            prefer = prefer or ["product name", "product", "item", "sku", "sub-category", "subcategory", "category"]
            best, best_score = None, -1.0
            for c, cnorm in col_map.items():
                score = 0.0
                for i, key in enumerate(prefer[::-1]):
                    if key in cnorm:
                        score += 1.0 + i * 0.1
                if not pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                if "id" in cnorm and not any(k in cnorm for k in ("name", "product")):
                    score -= 0.5
                if score > best_score:
                    best, best_score = c, score
            return best

        # Top N <group> by <metric>
        m = re.search(r"top\s*(\d{1,3})\s+([a-z0-9 _\-/]+?)\s+by\s+([a-z0-9 _\-/]+)", q)
        if m:
            n = int(m.group(1))
            group_name = m.group(2).strip()
            metric_name = m.group(3).strip()
            group_col = _find_col(group_name, prefer_numeric=False)
            metric_col = _find_col(metric_name, prefer_numeric=True)
            if group_col and metric_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        # "Top selling products" / "best selling products" (no explicit metric/by clause)
        if ("top" in q or "best" in q) and ("sell" in q) and ("product" in q or "item" in q):
            n = _parse_n(q, 5)
            group_col = _choose_group(["product name", "product", "item", "sku"])
            metric_col = _choose_metric(["sales", "revenue", "amount", "qty", "quantity", "units"])
            if group_col and metric_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        # Most frequent / mode
        m = re.search(r"(most\s+(?:common|frequent)\s+|mode\s+of\s+)([a-z0-9 _\-/]+)", q)
        if m:
            col_name = m.group(2).strip()
            col = _find_col(col_name, prefer_numeric=False)
            if col:
                vc = df[col].astype(str).value_counts().head(_parse_n(q, 5))
                items = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                return f"Most frequent in {col}: {items}"

        # Count by <column> / distribution of <column>
        m = re.search(r"(count\s+by|distribution\s+of)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col_name = m.group(2).strip()
            col = _find_col(col_name, prefer_numeric=False)
            if col:
                vc = df[col].astype(str).value_counts().head(_parse_n(q, 10))
                items = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                return f"Counts by {col}: {items}"

        # Aggregates: mean/avg, sum/total, min, max
        m = re.search(r"(average|avg|mean)\s+of\s+([a-z0-9 _\-/]+)", q)
        if not m:
            m = re.search(r"what\s+is\s+the\s+(average|avg|mean)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Average {col}: {df[col].dropna().mean():,.2f}"

        m = re.search(r"(sum|total|overall)\s+of\s+([a-z0-9 _\-/]+)", q)
        if not m:
            m = re.search(r"what\s+is\s+the\s+(sum|total)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Total {col}: {df[col].dropna().sum():,.2f}"

        m = re.search(r"(min|minimum|lowest)\s+of\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Minimum {col}: {df[col].dropna().min():,.2f}"

        m = re.search(r"(max|maximum|highest)\s+of\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Maximum {col}: {df[col].dropna().max():,.2f}"

        # Special: most profitable/most sales by <group>
        m = re.search(r"most\s+(profitable|revenue|sales)\s+(?:by|for)\s+([a-z0-9 _\-/]+)", q)
        if m:
            metric_hint = m.group(1)
            group_name = m.group(2).strip()
            metric_col = _find_col("profit" if "profit" in metric_hint else "sales", prefer_numeric=True)
            group_col = _find_col(group_name, prefer_numeric=False)
            if metric_col and group_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(_parse_n(q, 5))
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        return None
    except Exception:
        return None
