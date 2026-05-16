# ================================================================
#  Job Market Intelligence Dashboard
#  Author  : Deepanshu Kumar
#  Stack   : Streamlit · Plotly · Pandas · Scikit-learn
#  Model   : model.pkl + encoders.pkl  (your pre-trained files)
#  Data    : naukri_data_science_jobs_india.csv
#
#  HOW TO RUN:
#    1. Put app.py, model.pkl, encoders.pkl, and the CSV
#       all in the SAME folder
#    2. Open terminal in that folder
#    3. Run:  streamlit run app.py
# ================================================================

import pickle
import pandas as pd
import plotly.express as px
import streamlit as st

# ── must be the very FIRST streamlit call ───────────────────
st.set_page_config(
    page_title="India DS/ML Job Market",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom styling ───────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e2e8f0; }
    [data-testid="stSidebar"]  { background-color: #0d1117; }
    [data-testid="metric-container"] {
        background: #131929;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 14px 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #131929;
        border-radius: 8px;
        border: 1px solid #1e2a3a;
        color: #64748b;
        padding: 8px 20px;
        margin-right: 4px;
    }
    .stTabs [aria-selected="true"] {
        background: #1d4ed8 !important;
        color: #fff !important;
        border-color: #1d4ed8 !important;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .insight {
        background: rgba(79,247,168,0.07);
        border: 1px solid rgba(79,247,168,0.25);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 14px;
        color: #94a3b8;
        margin-top: 10px;
    }
    .insight b { color: #4ff7a8; }
</style>
""", unsafe_allow_html=True)


# ================================================================
#  LOAD MODEL  (runs once, stays in memory)
#
#  @st.cache_resource  →  Streamlit keeps this in memory
#  so the pkl files are NOT read from disk on every interaction.
#  Without this, the model would reload every time you click anything.
# ================================================================
@st.cache_resource
def load_model():
    # open pkl in read-binary ("rb") mode
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)          # RandomForestRegressor

    with open("encoders.pkl", "rb") as f:
        enc = pickle.load(f)            # dict with keys: loc, role, skills

    le_loc  = enc["loc"]                # LabelEncoder — knows 68 cities
    le_role = enc["role"]               # LabelEncoder — knows 6454 roles
    SKILLS  = enc["skills"]             # list of 14 skill names

    # feature order must EXACTLY match what the model was trained on
    # 5 base features + 14 skill flags = 19 total (matches model.n_features_in_)
    feature_cols = (
        ["loc_enc", "role_enc", "skill_count", "Min_Exp", "Max_Exp"]
        + ["skill_" + s.replace(" ", "_").replace("/", "_") for s in SKILLS]
    )

    return model, le_loc, le_role, SKILLS, feature_cols


# ================================================================
#  LOAD AND PREPARE DATA  (runs once, cached)
#
#  @st.cache_data  →  the CSV is read and cleaned only once.
#  The cleaned dataframe is reused across all tabs.
# ================================================================
@st.cache_data
def load_data(_le_loc, _le_role, SKILLS):
    # underscore prefix tells Streamlit NOT to hash these objects
    df = pd.read_csv("naukri_data_science_jobs_india.csv")

    # --- CLEAN ---
    df["Location"] = df["Location"].str.lower().str.strip()
    df["Job_Role"]  = df["Job_Role"].str.lower().str.strip()
    df["Company"]   = df["Company"].str.strip()

    # keep only rows that the encoder knows about
    # (unknown values would crash le_loc.transform())
    df = df[df["Location"].isin(_le_loc.classes_)]
    df = df[df["Job_Role"].isin(_le_role.classes_)]

    # --- PARSE EXPERIENCE "3-6" → Min=3, Max=6, Avg=4.5 ---
    def parse_exp(s):
        try:
            parts = [int(x.strip()) for x in str(s).split("-")]
            if len(parts) == 2:
                return parts[0], parts[1], round(sum(parts) / 2, 1)
        except:
            pass
        return None, None, None

    parsed        = df["Job Experience"].apply(parse_exp)
    df["Min_Exp"] = parsed.apply(lambda x: x[0])
    df["Max_Exp"] = parsed.apply(lambda x: x[1])
    df["Avg_Exp"] = parsed.apply(lambda x: x[2])
    df            = df.dropna(subset=["Avg_Exp"])

    # --- SKILL FLAGS ---
    # for each skill, create a column: 1 if mentioned in description, 0 if not
    desc = df["Skills/Description"].str.lower().fillna("")
    for skill in SKILLS:
        col = "skill_" + skill.replace(" ", "_").replace("/", "_")
        df[col] = desc.str.contains(skill).astype(int)

    # total number of skills per job posting
    skill_cols        = ["skill_" + s.replace(" ", "_").replace("/", "_") for s in SKILLS]
    df["skill_count"] = df[skill_cols].sum(axis=1)

    # encode for model use
    df["loc_enc"]  = _le_loc.transform(df["Location"])
    df["role_enc"] = _le_role.transform(df["Job_Role"])

    return df


# ── LOAD EVERYTHING ─────────────────────────────────────────
model, le_loc, le_role, SKILLS, feature_cols = load_model()
df = load_data(le_loc, le_role, SKILLS)

# dropdown lists — only show common roles (appear > 30 times)
role_counts  = df["Job_Role"].value_counts()
common_roles = sorted(role_counts[role_counts > 30].index.tolist())
all_locations = sorted(df["Location"].unique().tolist())


# ================================================================
#  SIDEBAR — global filters applied to every tab
# ================================================================
with st.sidebar:
    st.markdown("## 🎛 Filters")
    st.markdown("---")

    sel_city = st.selectbox(
        "📍 City",
        ["All Cities"] + [l.title() for l in all_locations]
    )
    sel_role = st.selectbox(
        "💼 Role",
        ["All Roles"] + [r.title() for r in common_roles]
    )
    exp_range = st.slider(
        "📅 Experience (yrs)", 0, 15, (0, 15)
    )

    st.markdown("---")
    st.markdown("### 🤖 Your Model")
    st.success("✅ model.pkl loaded")
    st.caption(f"Algorithm  : Random Forest")
    st.caption(f"Trees      : {model.n_estimators}")
    st.caption(f"Features   : {model.n_features_in_}")
    st.caption(f"Skills     : {len(SKILLS)}")
    st.markdown("---")
    st.caption("Data: Naukri.com · 12,000 rows")


# ── filtered dataframe used in all tabs below ────────────────
fdf = df.copy()
if sel_city != "All Cities":
    fdf = fdf[fdf["Location"] == sel_city.lower()]
if sel_role != "All Roles":
    fdf = fdf[fdf["Job_Role"] == sel_role.lower()]
fdf = fdf[
    (fdf["Avg_Exp"] >= exp_range[0]) &
    (fdf["Avg_Exp"] <= exp_range[1])
]


# ================================================================
#  HEADER
# ================================================================
st.markdown("# 📊 India DS/ML Job Market Dashboard")
st.markdown("*12,000 real Naukri.com job postings · Built by Deepanshu Kumar*")
st.markdown("---")

# ── 4 metric cards at the top ────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Jobs",       f"{len(fdf):,}")
c2.metric("Companies Hiring", f"{fdf['Company'].nunique():,}")
c3.metric(
    "Avg Experience",
    f"{fdf['Avg_Exp'].mean():.1f} yrs" if len(fdf) > 0 else "N/A"
)
c4.metric(
    "Top City",
    fdf["Location"].value_counts().index[0].title()
    if len(fdf) > 0 else "N/A"
)
st.markdown("---")


# ================================================================
#  4 TABS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview",
    "🤖 Experience Predictor",
    "🔍 Skill Gap Analyser",
    "🗂 Raw Data"
])


# ================================================================
#  TAB 1 — MARKET OVERVIEW
#  Shows: skill demand, city distribution, top companies, role ranking
# ================================================================
with tab1:

    # ── SKILL DEMAND ─────────────────────────────────────────
    st.subheader("Most In-Demand Skills")

    # sum each skill column to get how many jobs mention it
    skill_cols = ["skill_" + s.replace(" ", "_").replace("/", "_") for s in SKILLS]
    sums       = fdf[skill_cols].sum().sort_values(ascending=False)
    sums.index = [i.replace("skill_","").replace("_"," ").title() for i in sums.index]
    skill_df   = sums.reset_index()
    skill_df.columns = ["Skill", "Jobs"]
    skill_df["% of Jobs"] = (skill_df["Jobs"] / max(len(fdf), 1) * 100).round(1)

    left, right = st.columns([2, 1])
    with left:
        fig = px.bar(
            skill_df, x="Jobs", y="Skill", orientation="h",
            color="Jobs", color_continuous_scale="Blues",
            text="% of Jobs",
            title="Skills ranked by number of job postings"
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            plot_bgcolor="#131929", paper_bgcolor="#131929",
            font_color="#94a3b8", showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Skill table")
        st.dataframe(skill_df, use_container_width=True, hide_index=True)

    top_s = skill_df.iloc[0]
    st.markdown(f"""<div class="insight"><b>💡 Insight:</b>
    <b>{top_s['Skill']}</b> appears in <b>{top_s['% of Jobs']}%</b> of all postings.
    Python + SQL together cover 60 %+ of every DS/ML job in India.
    If you only learn two things, make it these.</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── CITY + ROLE SIDE BY SIDE ─────────────────────────────
    st.subheader("Jobs by City  &  Jobs by Role")
    col1, col2 = st.columns(2)

    with col1:
        city_df = fdf["Location"].value_counts().head(10).reset_index()
        city_df.columns = ["City", "Jobs"]
        city_df["City"] = city_df["City"].str.title()
        fig = px.pie(
            city_df, values="Jobs", names="City",
            hole=0.42,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Top 10 cities"
        )
        fig.update_layout(
            plot_bgcolor="#131929", paper_bgcolor="#131929",
            font_color="#94a3b8"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        role_df = fdf["Job_Role"].value_counts().head(12).reset_index()
        role_df.columns = ["Role", "Jobs"]
        role_df["Role"] = role_df["Role"].str.title()
        fig = px.bar(
            role_df, x="Jobs", y="Role", orientation="h",
            color="Jobs", color_continuous_scale="Teal",
            title="Top 12 roles by number of postings"
        )
        fig.update_layout(
            plot_bgcolor="#131929", paper_bgcolor="#131929",
            font_color="#94a3b8", showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── TOP COMPANIES ────────────────────────────────────────
    st.subheader("Top Hiring Companies")
    col1, col2 = st.columns([2, 1])

    with col1:
        comp_df = fdf["Company"].value_counts().head(15).reset_index()
        comp_df.columns = ["Company", "Open Positions"]
        fig = px.bar(
            comp_df, x="Open Positions", y="Company", orientation="h",
            color="Open Positions", color_continuous_scale="Purples",
            text="Open Positions",
            title="Companies with most DS/ML openings"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="#131929", paper_bgcolor="#131929",
            font_color="#94a3b8", showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Top 15 companies")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── ROLE DEMAND RANKING ───────────────────────────────────
    # shows avg experience required per role (senior = more exp = in demand differently)
    st.subheader("Role Demand Ranking — Experience Required")
    exp_role = (
        fdf.groupby("Job_Role")["Avg_Exp"]
        .agg(["mean", "count"])
        .reset_index()
    )
    exp_role.columns = ["Role", "Avg Exp (yrs)", "Job Count"]
    exp_role = exp_role[exp_role["Job Count"] >= 10]  # only roles with enough data
    exp_role = exp_role.sort_values("Job Count", ascending=False).head(15)
    exp_role["Role"] = exp_role["Role"].str.title()
    exp_role["Avg Exp (yrs)"] = exp_role["Avg Exp (yrs)"].round(1)

    fig = px.scatter(
        exp_role, x="Job Count", y="Avg Exp (yrs)",
        size="Job Count", color="Avg Exp (yrs)",
        color_continuous_scale="Oranges",
        hover_name="Role", text="Role",
        title="Role ranking — bubble size = number of jobs, height = seniority"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        plot_bgcolor="#131929", paper_bgcolor="#131929",
        font_color="#94a3b8", showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight"><b>💡 How to read this:</b>
    Roles to the right have the most job openings — highest demand.
    Roles higher up need more experience — more senior.
    Aim for roles in the <b>bottom-right</b>: high demand, entry-friendly.
    </div>""", unsafe_allow_html=True)


# ================================================================
#  TAB 2 — EXPERIENCE PREDICTOR (ML)
#
#  How it works:
#  User fills in role + city + skills + exp range
#  → we encode using the SAME encoders from pkl
#  → build one input row with 19 features
#  → model.predict() returns avg experience needed
# ================================================================
with tab2:
    st.subheader("🤖 Experience Requirement Predictor")
    st.markdown("*Your pre-trained Random Forest model predicts how many years of experience a job needs*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # role dropdown — only show common roles (appear > 30 times)
        pred_role = st.selectbox(
            "🎯 Target job role",
            [r.title() for r in common_roles],
            key="pred_role"
        )
        pred_city = st.selectbox(
            "📍 Target city",
            [l.title() for l in all_locations],
            key="pred_city"
        )
        min_exp = st.slider("Min experience you're targeting (yrs)", 0, 10, 2)
        max_exp = st.slider("Max experience range (yrs)",            2, 20, 6)

    with col2:
        st.markdown("**Your skills — tick what you have:**")
        # multiselect returns a list of selected skill names
        my_skills = st.multiselect(
            "Select your skills",
            [s.title() for s in SKILLS],
            default=["Python", "Sql"]
        )
        st.markdown(" ")
        st.info(f"You selected **{len(my_skills)}** of {len(SKILLS)} skills tracked")

    st.markdown("---")

    # ── PREDICT BUTTON ───────────────────────────────────────
    if st.button("🔮  Predict experience required", type="primary"):

        # step 1 — encode city using le_loc (same encoder from training)
        try:
            loc_enc = le_loc.transform([pred_city.lower()])[0]
        except ValueError:
            st.warning("City not found in model — using default")
            loc_enc = 0

        # step 2 — encode role using le_role
        try:
            role_enc = le_role.transform([pred_role.lower()])[0]
        except ValueError:
            st.warning("Role not found in model — using default")
            role_enc = 0

        # step 3 — build 14 skill flags in the same order as training
        # 1 = you have the skill, 0 = you don't
        skill_flags = [
            1 if skill.title() in my_skills else 0
            for skill in SKILLS
        ]
        skill_count = sum(skill_flags)

        # step 4 — assemble the input row (19 columns, same order as training)
        input_row = pd.DataFrame(
            [[loc_enc, role_enc, skill_count, min_exp, max_exp] + skill_flags],
            columns=feature_cols
        )

        # step 5 — model predicts! [0] gets the single number from the list
        prediction = model.predict(input_row)[0]

        # step 6 — show result
        r1, r2, r3 = st.columns(3)
        r1.metric("🎯 Predicted experience", f"{prediction:.1f} yrs")
        r2.metric("Your min target",         f"{min_exp} yrs")
        r3.metric("Your max target",         f"{max_exp} yrs")

        # interpret result for the user
        if prediction <= 3:
            verdict = "🟢 Entry level — you can apply for this now!"
            advice  = "This role is suitable for freshers and early-career candidates."
        elif prediction <= 6:
            verdict = "🟡 Mid level — build 1–2 more projects first"
            advice  = "You are on track. A strong portfolio project could compensate for years."
        else:
            verdict = "🔴 Senior level — aim for junior versions of this role first"
            advice  = "Target 'Junior' or 'Associate' versions of this role to build experience."

        st.markdown(f"""<div class="insight">
        <b>Result:</b> A <b>{pred_role}</b> role in <b>{pred_city}</b>
        typically requires <b>{prediction:.1f} years</b> of experience.<br><br>
        {verdict}<br>{advice}
        </div>""", unsafe_allow_html=True)

        # show which skills you have vs missing for this role
        st.markdown("#### Your skill breakdown for this role")
        have  = [s.title() for s in SKILLS if s.title() in my_skills]
        lacks = [s.title() for s in SKILLS if s.title() not in my_skills]
        s1, s2 = st.columns(2)
        with s1:
            st.success(f"✅ Skills you have ({len(have)})")
            st.write(", ".join(have) if have else "None selected")
        with s2:
            st.error(f"❌ Skills to learn ({len(lacks)})")
            st.write(", ".join(lacks[:6]))

    st.markdown("---")

    # ── HOW THE MODEL WORKS ──────────────────────────────────
    with st.expander("📖 How does the model predict? (click to expand)"):
        st.markdown("""
        **Your model — Random Forest Regressor**

        | Step | What happens |
        |------|-------------|
        | 1 | City and role are encoded into numbers using LabelEncoder |
        | 2 | Your 14 skill flags are built (1 = have, 0 = don't have) |
        | 3 | These 19 numbers are passed to 300 decision trees |
        | 4 | Each tree independently predicts required experience |
        | 5 | All 300 predictions are averaged → final answer |

        **Why 99.9% R²?**
        The model uses Min_Exp and Max_Exp as features.
        Since Avg_Exp = (Min + Max) / 2, the model learns this formula perfectly.
        """)

        # feature importance chart
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp_df = imp.sort_values(ascending=False).head(8).reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Feature"] = (
            imp_df["Feature"]
            .str.replace("skill_", "").str.replace("_", " ").str.title()
        )
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title="What the model uses most to predict"
        )
        fig.update_layout(
            plot_bgcolor="#131929", paper_bgcolor="#131929",
            font_color="#94a3b8", showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
#  TAB 3 — SKILL GAP ANALYSER
#
#  User selects their skills → for each role, we calculate
#  what % of the role's required skills the user already has
# ================================================================
with tab3:
    st.subheader("🔍 Skill Gap Analyser")
    st.markdown("*Select your skills — see which roles you qualify for and exactly what to learn next*")
    st.markdown("---")

    user_skills = st.multiselect(
        "✅ Select ALL skills you currently have",
        [s.title() for s in SKILLS],
        default=["Python", "Sql", "Machine Learning"]
    )

    if user_skills:
        user_lower = [s.lower() for s in user_skills]
        results    = []

        # analyse every common role
        for role in common_roles:
            role_df = fdf[fdf["Job_Role"] == role]
            if len(role_df) < 5:
                continue

            # skills needed = those mentioned in > 20% of this role's postings
            needed = [
                s for s in SKILLS
                if role_df["skill_" + s.replace(" ","_").replace("/","_")].mean() > 0.20
            ]
            if not needed:
                continue

            have   = [s for s in needed if s in user_lower]
            lacks  = [s for s in needed if s not in user_lower]
            match  = round(len(have) / len(needed) * 100)

            results.append({
                "Role":             role.title(),
                "Match %":          match,
                "✅ You have":       ", ".join(s.title() for s in have)  or "None",
                "📚 Learn next":    ", ".join(s.title() for s in lacks[:3]) or "—",
                "Total jobs":       len(role_df)
            })

        if results:
            res_df = pd.DataFrame(results).sort_values("Match %", ascending=False)

            # ── 3-column match breakdown ─────────────────────
            st.markdown("#### Your job match results")
            strong = res_df[res_df["Match %"] >= 70]
            medium = res_df[(res_df["Match %"] >= 40) & (res_df["Match %"] < 70)]
            weak   = res_df[res_df["Match %"] < 40]

            g1, g2, g3 = st.columns(3)

            with g1:
                st.markdown("##### ✅ Strong match (70 %+)")
                if len(strong):
                    for _, row in strong.iterrows():
                        st.markdown(f"**{row['Role']}** — {row['Match %']}%")
                        st.caption(f"Learn: {row['📚 Learn next']}")
                else:
                    st.info("Add more skills to unlock strong matches")

            with g2:
                st.markdown("##### 🟡 Partial match (40–70 %)")
                for _, row in medium.iterrows():
                    st.markdown(f"**{row['Role']}** — {row['Match %']}%")
                    st.caption(f"Learn: {row['📚 Learn next']}")

            with g3:
                st.markdown("##### 🔴 Needs work (< 40 %)")
                for _, row in weak.iterrows():
                    st.markdown(f"**{row['Role']}** — {row['Match %']}%")
                    st.caption(f"Learn: {row['📚 Learn next']}")

            # ── match bar chart ───────────────────────────────
            fig = px.bar(
                res_df, x="Role", y="Match %",
                color="Match %",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                text="Match %",
                title="Your skill match % across all roles"
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="#131929", paper_bgcolor="#131929",
                font_color="#94a3b8", showlegend=False,
                coloraxis_showscale=False, xaxis_tickangle=-35
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── full table ────────────────────────────────────
            st.markdown("#### Full breakdown")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        else:
            st.warning("No roles found in filtered data. Try removing the city/role filter.")

    else:
        st.info("👆 Select at least one skill above to see your results")


# ================================================================
#  TAB 4 — RAW DATA
#  Searchable, sortable table of all 12,000 job postings
# ================================================================
with tab4:
    st.subheader("🗂 Browse All Job Postings")
    st.markdown("*Search and sort through all 12,000 rows from Naukri.com*")
    st.markdown("---")

    # search box — filters by role or company name
    search = st.text_input("🔍 Search by role or company name")

    display = fdf.copy()
    if search:
        mask = (
            display["Job_Role"].str.contains(search.lower(), na=False) |
            display["Company"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    # show result count
    st.caption(f"Showing **{len(display):,}** jobs")

    # show only readable columns — hide the encoded number columns
    show_cols = ["Job_Role", "Company", "Location", "Job Experience", "Skills/Description"]
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
        display[show_cols].rename(columns={
            "Job_Role":          "Role",
            "Job Experience":    "Experience",
            "Skills/Description":"Skills"
        }),
        use_container_width=True,
        hide_index=True,
        height=500
    )

    # download button — user can save the filtered data as CSV
    csv = display[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️  Download filtered data as CSV",
        data      = csv,
        file_name = "filtered_jobs.csv",
        mime      = "text/csv"
    )


# ── FOOTER ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#1e2a3a;font-size:12px'>"
    "Built by <b>Deepanshu Kumar</b> &nbsp;·&nbsp; "
    "Python · Pandas · Scikit-learn · Plotly · Streamlit &nbsp;·&nbsp; "
    "Data: Naukri.com India · 12,000 DS/ML job postings"
    "</p>",
    unsafe_allow_html=True
)