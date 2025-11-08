# asymptotic_visualizer.py
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objs as go
import math

# --------------------------
# Safe factorial for lambdify
# --------------------------
def np_factorial_safe(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    # handle integers elementwise
    for i, val in np.ndenumerate(x):
        try:
            iv = int(val)
        except Exception:
            out[i] = np.nan
            continue
        if iv < 0:
            out[i] = np.nan
        elif iv > 170:
            out[i] = np.inf
        else:
            out[i] = math.factorial(iv)
    return out

# --------------------------
# Page config and title
# --------------------------
st.set_page_config(layout="wide", page_title="Asymptotic Function Visualizer")
st.title("📈 Asymptotic Function Visualizer")
st.markdown(
    "Enter one function per line . Examples: `1`, `log(n)`, `n`, `n*log(n)`, `n**2`, `2**n`, `factorial(n)`."
)

# --------------------------
# Symbol and parsing setup
# --------------------------
n = sp.Symbol("n", positive=True)
TRANSFORMATIONS = ()
# Basis of standard complexities (ordered from small to large roughly)
BASIS = [
    ("1", sp.Integer(1)),
    ("log(n)", sp.log(n)),
    ("sqrt(n)", sp.sqrt(n)),
    ("n", n),
    ("n*log(n)", n * sp.log(n)),
    ("n**2", n ** 2),
    ("n**3", n ** 3),
    ("2**n", 2 ** n),
    ("factorial(n)", sp.factorial(n)),
]

ALGO_EXAMPLES = {
    "1": ["Array access by index", "Return constant", "Stack push/pop (amortized)"],
    "log(n)": ["Binary Search", "Search in balanced BST (log n)", "Some divide-and-conquer steps"],
    "sqrt(n)": ["Square-root decomposition queries", "Trial division primality (basic)"],
    "n": ["Linear search", "Traversing linked list", "Single pass algorithms"],
    "n*log(n)": ["Merge Sort", "Heap Sort", "Average-case Quick Sort"],
    "n**2": ["Bubble Sort", "Insertion Sort", "Naive matrix multiply (O(n²) for built-in dims)"],
    "n**3": ["Cubic DP", "Naive matrix multiplication (O(n³))", "Floyd–Warshall"],
    "2**n": ["Subset generation", "Brute-force backtracking (TSP brute force)"],
    "factorial(n)": ["Full permutation generation", "Bruteforce Hamiltonian cycle variants"],
}

# --------------------------
# Inputs: functions + sliders
# --------------------------
with st.sidebar:
    st.header("Plot range")
    max_n = st.slider("Max n (x-axis)", min_value=10, max_value=2000, value=60, step=10)
    sample_points = st.slider("Number of sample points", min_value=50, max_value=2000, value=400, step=10)
    plot_basis = st.checkbox("Also plot matched basis g(n) (dashed)", value=True)
    st.markdown("**Notes:** using larger `max_n` and `sample_points` makes asymptotic trends clearer but may be slower.")
    st.markdown("Use SymPy syntax. `log` is natural log. Use `factorial(n)` for n!.")

user_text = st.text_area("Enter one function per line:", value="n\nn*log(n)\nn**2\n2**n\nfactorial(n)")

functions_input = [line.strip() for line in user_text.splitlines() if line.strip()]

# --------------------------
# Helpers
# --------------------------
def safe_parse(expr_str: str):
    try:
        # allow common names
        local_dict = {
            "n": n,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "factorial": sp.factorial,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
        }
        return sp.simplify(sp.sympify(expr_str, locals=local_dict))
    except Exception as e:
        raise ValueError(f"Could not parse expression `{expr_str}`: {e}")

def numeric_func_from_sympy(expr_sympy):
    # prefer mapping factorial to safe numpy function
    try:
        f = sp.lambdify(n, expr_sympy, modules=[{"factorial": np_factorial_safe}, "numpy"])
        return f
    except Exception:
        return sp.lambdify(n, expr_sympy, "numpy")

def compute_limit_symbolic(f_expr, g_expr):
    try:
        lim = sp.limit(sp.simplify(f_expr / g_expr), n, sp.oo)
        return lim
    except Exception:
        return None

def classify_limit(lim):
    # interpret sympy limit
    if lim is None:
        return None
    try:
        if lim == sp.oo:
            return {"O": False, "o": False, "Omega": True, "omega": True, "Theta": False}
        if lim == 0:
            return {"O": True, "o": True, "Omega": False, "omega": False, "Theta": False}
        if lim.is_real:
            return {"O": True, "o": False, "Omega": True, "omega": False, "Theta": True}
    except Exception:
        pass
    return None

# --------------------------
# Visualization action
# --------------------------
if st.button("📈 Visualize"):
    if not functions_input:
        st.warning("Please enter at least one function.")
    else:
        xs = np.linspace(1, max_n, sample_points)
        fig = go.Figure()
        results = []

        # color palette (plotly default) will assign distinct colors automatically
        for f_str in functions_input:
            try:
                f_sym = safe_parse(f_str)
            except Exception as e:
                st.error(str(e))
                continue

            # numeric function
            f_num = numeric_func_from_sympy(f_sym)
            try:
                f_vals = f_num(xs)
                # sanitize: convert inf/nan to np.nan for plotting
                f_vals = np.array(f_vals, dtype=float)
                f_vals[~np.isfinite(f_vals)] = np.nan
            except Exception as e:
                st.error(f"Cannot sample numerically `{f_str}`: {e}")
                continue

            # plot f(n)
            fig.add_trace(go.Scatter(x=xs, y=f_vals, name=f"f(n) = {f_str}", mode="lines"))

            # compute g_ratio = f(n)/n symbolically
            try:
                g_ratio_sym = sp.simplify(sp.simplify(f_sym) / n)
                g_ratio_str = str(sp.simplify(g_ratio_sym))
            except Exception:
                g_ratio_sym = None
                g_ratio_str = "(could not simplify)"

            # numeric samples for g_ratio (optional)
            if g_ratio_sym is not None:
                try:
                    g_ratio_num = numeric_func_from_sympy(g_ratio_sym)(xs)
                    g_ratio_num = np.array(g_ratio_num, dtype=float)
                    g_ratio_num[~np.isfinite(g_ratio_num)] = np.nan
                except Exception:
                    g_ratio_num = None
            else:
                g_ratio_num = None

            # Find matched basis g_basis using symbolic limit where possible, else numeric heuristic
            chosen_basis = None
            for label, basis_sym in BASIS:
                # compute symbolic limit if possible
                lim = compute_limit_symbolic(f_sym, basis_sym)
                accepts_O = False
                if lim is not None:
                    try:
                        if lim == sp.oo:
                            accepts_O = False
                        else:
                            accepts_O = True
                    except Exception:
                        accepts_O = True
                else:
                    # numeric heuristic
                    try:
                        basis_func = numeric_func_from_sympy(basis_sym)
                        ratio = np.divide(f_vals, basis_func(xs), out=np.full_like(f_vals, np.nan), where=(basis_func(xs) != 0))
                        finite_mask = np.isfinite(ratio)
                        numeric_max = np.nanmax(np.abs(ratio[finite_mask])) if np.any(finite_mask) else np.inf
                        if numeric_max < 1e8:
                            accepts_O = True
                    except Exception:
                        accepts_O = False

                if accepts_O:
                    chosen_basis = {"label": label, "sym": basis_sym, "sym_lim": lim}
                    break

            # If user asked to also plot matched basis, add dashed line
            if plot_basis and chosen_basis is not None:
                try:
                    g_basis_num = numeric_func_from_sympy(chosen_basis["sym"])(xs)
                    g_basis_num = np.array(g_basis_num, dtype=float)
                    g_basis_num[~np.isfinite(g_basis_num)] = np.nan
                    fig.add_trace(go.Scatter(x=xs, y=g_basis_num, name=f"g_basis(n)={chosen_basis['label']} (for {f_str})", mode="lines", line=dict(dash="dash")))
                except Exception:
                    pass

            # pick example algos for chosen basis
            examples_list = ["(No common examples found)"]
            if chosen_basis is not None:
                key = chosen_basis["label"]
                examples_list = ALGO_EXAMPLES.get(key, ["(No common examples found)"])

            # classification using symbolic limit (if available)
            classification = None
            if chosen_basis is not None:
                classification = classify_limit(chosen_basis["sym_lim"])

            results.append({
                "f_str": f_str,
                "f_sym": f_sym,
                "g_ratio_str": g_ratio_str,
                "matched_basis": chosen_basis["label"] if chosen_basis else None,
                "matched_basis_limit": chosen_basis["sym_lim"] if chosen_basis else None,
                "examples": examples_list,
                "classification": classification,
            })

        # finalize plot layout
        fig.update_layout(
            title="f(n) (solid) and matched basis g(n) (dashed if shown)",
            xaxis_title="n",
            yaxis_title="Value (clipped/NaN for display)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=560
        )
        st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # Display textual results
        # --------------------------
        st.subheader("Analysis & Matches")
        for r in results:
            st.markdown(f"### 🔹 `f(n) = {r['f_str']}`")
            st.markdown(f"- **g(n) = f(n)/n** (simplified): `{r['g_ratio_str']}`")
            if r["matched_basis"]:
                st.markdown(f"- **Matched standard complexity (g_basis)**: `{r['matched_basis']}`")
                st.markdown(f"  - symbolic `lim_{'{n->∞}'} f(n)/g_basis(n)` = `{r['matched_basis_limit']}`")
                if r["classification"]:
                    yesno = lambda v: "Yes" if v else "No"
                    st.markdown(f"  - f = O(g_basis)? {yesno(r['classification']['O'])}, Θ? {yesno(r['classification']['Theta'])}, Ω? {yesno(r['classification']['Omega'])}")
            else:
                st.markdown("- **Matched standard complexity**: None found among BASIS (try larger `max_n` or expand BASIS).")

            st.markdown("- **Example algorithms with this matched complexity:**")
            for ex in r["examples"]:
                st.markdown(f"  - {ex}")

            st.divider()
