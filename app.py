import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
import joblib
import os
import gdown

st.set_page_config(layout="wide", page_title="LIX - Longevity Intervention Exchange")

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_body_age_model():
    """Download and load the body age prediction model"""
    model_path = "body_age_model.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading body age model..."):
            try:
                # Google Drive file ID from your link
                file_id = "1Cz3ayWtDov-8SqEqCt2EMj8eUnWnIZGX"
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the model
body_age_model = load_body_age_model()

# -------------------------
# Prediction Functions
# -------------------------
def predict_body_age(profile_data):
    """Predict body age using the trained model"""
    if body_age_model is None:
        st.error("Model not available. Using fallback calculation.")
        return profile_data.get("age", 45)
    
    # Prepare data in the format expected by the model
    data = pd.DataFrame([{
        "age":profile_data.get("age",16),
        "bmi": profile_data.get("bmi", 25.0),
        "avg_sleep_hours": profile_data.get("avg_sleep_hours", 7.0),
        "exercise_frequency": profile_data.get("exercise_frequency", 3),
        "smoking_status": "yes" if profile_data.get("smoking", 0) == 1 else "no",
        "alcohol_intake": profile_data.get("alcohol_intake", "no"),
        "cholesterol_level": profile_data.get("cholesterol_level", 200),
        "blood_sugar_fasting": profile_data.get("blood_sugar_fasting", 95),
        "stress_level": profile_data.get("stress_level", 5)
    }])

    cat_map = {"no": 0, "yes": 1}
    for col in ["smoking_status", "alcohol_intake"]:
        if col in data.columns:
            data[col] = data[col].map(cat_map)
        
    try:
        pred = body_age_model.predict(data)[0]
        return float(pred)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return profile_data.get("age", 45)

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight and height"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def body_age_to_health_score(body_age, chronological_age):
    """Convert body age difference to health score (0-100)"""
    age_diff = chronological_age - body_age
    # If body age < chronological age → healthier → higher score
    # Scale: -20 years diff = 100, 0 diff = 70, +20 years diff = 40
    base_score = 70
    score = base_score + (age_diff * 1.5)
    return float(np.clip(score, 0, 100))

def years_from_score_delta(delta_score):
    """Convert health score change to estimated years gained/lost"""
    return float(delta_score) * 0.15

def apply_delta_to_profile(profile, delta):
    """Apply intervention changes to profile"""
    profile2 = profile.copy()
    for k, v in delta.items():
        if callable(v):
            profile2[k] = v(profile2.get(k, None))
        else:
            profile2[k] = v
    # Recalculate BMI if weight or height changed
    if "weight_kg" in profile2 and "height_cm" in profile2:
        profile2["bmi"] = calculate_bmi(profile2["weight_kg"], profile2["height_cm"])
    return profile2

def mission_for_intervention(it):
    """Generate mission cards for interventions"""
    templates = {
        "quit_smoke": "7-day starter: ลดการสูบลง 20% ต่อวัน + ติดต่อ hotline/กลุ่มสนับสนุน",
        "add_sleep": "7-day starter: นอนเพิ่ม 15 นาทีทุกคืนจนถึง +1 ชั่วโมง, ปิดหน้าจอก่อนนอน 30 นาที",
        "add_exercise": "7-day starter: เพิ่มออกกำลังกาย 2 ครั้ง/สัปดาห์ เริ่มจาก 20 นาที/ครั้ง",
        "reduce_stress": "7-day starter: ฝึกสมาธิ 10 นาที/วัน, จดบันทึกความเครียดและหาวิธีจัดการ",
        "lower_cholesterol": "ติดต่อแพทย์เพื่อตรวจและวางแผนการรักษา ปรับอาหาร เพิ่มไฟเบอร์",
        "control_blood_sugar": "ติดต่อแพทย์เพื่อตรวจและวางแผนการรักษา ลดน้ำตาล ออกกำลังกายสม่ำเสมอ",
        "reduce_alcohol": "7-day starter: ลดการดื่มลง 50%, เปลี่ยนเป็นน้ำหรือเครื่องดื่มไม่มีแอลกอฮอล์",
    }
    return templates.get(it["id"], "7-day starter: small, achievable steps to begin this intervention.")

# -------------------------
# Intervention catalog
# -------------------------
DEFAULT_INTERVENTIONS = [
    {
        "id": "quit_smoke",
        "name": "Quit Smoking",
        "delta": {"smoking": 0, "smoking_status": "no"},
        "cost": 150,
        "effort": 9
    },
    {
        "id": "add_sleep",
        "name": "Sleep +1 hour",
        "delta": {"avg_sleep_hours": lambda x: min(10, (x or 6) + 1)},
        "cost": 0,
        "effort": 3
    },
    {
        "id": "add_exercise",
        "name": "Exercise +2 times/week",
        "delta": {"exercise_frequency": lambda x: min(7, (x or 2) + 2)},
        "cost": 30,
        "effort": 5
    },
    {
        "id": "reduce_stress",
        "name": "Stress reduction program",
        "delta": {"stress_level": lambda x: max(1, (x or 5) - 2)},
        "cost": 80,
        "effort": 6
    },
    {
        "id": "lower_cholesterol",
        "name": "Cholesterol management",
        "delta": {"cholesterol_level": lambda x: max(150, (x or 220) - 30)},
        "cost": 100,
        "effort": 7
    },
    {
        "id": "control_blood_sugar",
        "name": "Blood sugar control",
        "delta": {"blood_sugar_fasting": lambda x: max(80, (x or 110) - 15)},
        "cost": 120,
        "effort": 7
    },
    {
        "id": "reduce_alcohol",
        "name": "Reduce alcohol intake",
        "delta": {"alcohol_intake": "moderate"},
        "cost": 50,
        "effort": 6
    },
]

# -------------------------
# UI: Header and mode toggle
# -------------------------
st.title("🧬 LIX — Longevity Intervention Exchange")
st.markdown("**AI-powered body age prediction and personalized longevity interventions**")

mode = st.radio("Mode", ["Personal", "Community"], horizontal=True)

# Layout columns
if mode == "Personal":
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    # ---------- LEFT: profile form ----------
    with col_left:
        st.header("👤 Your Profile")
        
        # Initialize session state
        if "profile" not in st.session_state:
            st.session_state["profile"] = {
                "age": 45,
                "gender": "Male",
                "height_cm": 175,
                "weight_kg": 80,
                "avg_sleep_hours": 7.0,
                "exercise_frequency": 3,
                "smoking": 0,
                "alcohol_intake": "no",
                "cholesterol_level": 200,
                "blood_sugar_fasting": 95,
                "stress_level": 5
            }
        
        if st.button("📋 Use sample profile"):
            st.session_state["profile"] = {
                "age": 45,
                "gender": "Male",
                "height_cm": 175,
                "weight_kg": 80,
                "avg_sleep_hours": 6.0,
                "exercise_frequency": 2,
                "smoking": 1,
                "alcohol_intake": "moderate",
                "cholesterol_level": 220,
                "blood_sugar_fasting": 105,
                "stress_level": 7
            }
            st.rerun()
        
        p = st.session_state["profile"]
        
        # Basic info
        st.subheader("Basic Information")
        p["age"] = st.number_input("Age (years)", min_value=18, max_value=100, value=int(p.get("age", 45)))
        p["gender"] = st.selectbox("Gender", ["Male", "Female"], index=0 if p.get("gender", "Male") == "Male" else 1)
        p["height_cm"] = st.number_input("Height (cm)", min_value=100, max_value=250, value=int(p.get("height_cm", 175)))
        p["weight_kg"] = st.number_input("Weight (kg)", min_value=30, max_value=200, value=int(p.get("weight_kg", 80)))
        
        # Calculate and display BMI
        bmi = calculate_bmi(p["weight_kg"], p["height_cm"])
        p["bmi"] = bmi
        st.metric("BMI", f"{bmi:.1f}")
        
        # Lifestyle factors
        st.subheader("Lifestyle")
        p["avg_sleep_hours"] = st.slider("Average sleep (hours/night)", 0.0, 12.0, float(p.get("avg_sleep_hours", 7.0)), 0.5)
        p["exercise_frequency"] = st.slider("Exercise (times/week)", 0, 7, int(p.get("exercise_frequency", 3)))
        p["smoking"] = 1 if st.checkbox("Current smoker", value=bool(p.get("smoking", 0))) else 0
        p["alcohol_intake"] = st.selectbox("Alcohol intake", ["no", "moderate", "high"], 
                                           index=["no", "moderate", "high"].index(p.get("alcohol_intake", "no")))
        p["stress_level"] = st.slider("Stress level (1-10)", 1, 10, int(p.get("stress_level", 5)))
        
        # Health metrics
        st.subheader("Health Metrics")
        p["cholesterol_level"] = st.number_input("Cholesterol level (mg/dL)", min_value=100, max_value=400, 
                                                  value=int(p.get("cholesterol_level", 200)))
        p["blood_sugar_fasting"] = st.number_input("Fasting blood sugar (mg/dL)", min_value=50, max_value=300, 
                                                     value=int(p.get("blood_sugar_fasting", 95)))

    # ---------- CENTER: results ----------
    with col_center:
        st.header("📊 Your Results")
        
        # Predict body age
        predicted_body_age = predict_body_age(p)
        chronological_age = p["age"]
        age_difference = chronological_age - predicted_body_age
        health_score = body_age_to_health_score(predicted_body_age, chronological_age)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chronological Age", f"{chronological_age} years")
        with col2:
            st.metric("Body Age", f"{predicted_body_age:.1f} years", 
                     f"{age_difference:+.1f} years",
                     delta_color="normal" if age_difference >= 0 else "inverse")
        with col3:
            st.metric("Health Score", f"{health_score:.1f}/100")
        
        # Interpretation
        if age_difference > 5:
            st.success("🎉 Great! Your body is younger than your chronological age!")
        elif age_difference > 0:
            st.info("👍 Your body age is close to your chronological age.")
        elif age_difference > -5:
            st.warning("⚠️ Your body age is slightly higher than your chronological age.")
        else:
            st.error("🚨 Your body age is significantly higher than your chronological age. Consider lifestyle changes.")
        
        # Timeline visualization
        st.markdown("#### 📈 Projected Timeline")
        years = list(range(0, 21))
        baseline_ages = [predicted_body_age + 0.8*yi for yi in years]  # Aging trajectory
        optimized_ages = [predicted_body_age + 0.5*yi for yi in years]  # With interventions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=baseline_ages, mode="lines", name="Current trajectory", 
                                line=dict(color="orange", width=2)))
        fig.add_trace(go.Scatter(x=years, y=optimized_ages, mode="lines", name="With interventions", 
                                line=dict(color="green", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=years, y=[chronological_age + yi for yi in years], 
                                mode="lines", name="Chronological age", 
                                line=dict(color="gray", width=1, dash="dot")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), 
                         xaxis_title="Years from now", yaxis_title="Age (years)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interventions marketplace
        st.subheader("🛒 Intervention Marketplace")
        
        # Calculate intervention impacts
        results = []
        for intervention in DEFAULT_INTERVENTIONS:
            new_profile = apply_delta_to_profile(p, intervention["delta"])
            new_body_age = predict_body_age(new_profile)
            new_health_score = body_age_to_health_score(new_body_age, chronological_age)
            
            score_gain = new_health_score - health_score
            age_reduction = predicted_body_age - new_body_age
            years_gained = years_from_score_delta(score_gain)
            
            results.append({
                **intervention,
                "new_body_age": new_body_age,
                "new_health_score": new_health_score,
                "score_gain": score_gain,
                "age_reduction": age_reduction,
                "years_gained": years_gained,
                "mission": mission_for_intervention(intervention)
            })
        
        # Sort by impact
        results = sorted(results, key=lambda x: x["years_gained"], reverse=True)
        
        # Display interventions
        for r in results:
            with st.expander(f"**{r['name']}** — Reduces body age by {r['age_reduction']:.1f} years | "
                           f"Gain {r['years_gained']:.1f} healthy years"):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.write(f"**Impact:** +{r['score_gain']:.1f} health score points")
                    st.write(f"**New body age:** {r['new_body_age']:.1f} years")
                    st.info(r["mission"])
                with col_b:
                    st.metric("Cost", f"${r['cost']}")
                    st.metric("Effort", f"{r['effort']}/10")
        
        # Optimization
        st.markdown("---")
        st.subheader("🎯 Optimize Your Plan")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            budget = st.number_input("Budget ($)", min_value=0, value=300, step=50)
        with col_opt2:
            effort_cap = st.slider("Max total effort", 0, 50, 20)
        
        if st.button("🔍 Find Optimal Bundle", type="primary"):
            # Greedy optimization by impact per cost
            candidates = sorted(results, key=lambda x: x["years_gained"] / max(1, x["cost"]), reverse=True)
            selected = []
            total_cost = 0
            total_effort = 0
            total_years = 0.0
            total_age_reduction = 0.0
            
            for c in candidates:
                if total_cost + c["cost"] <= budget and total_effort + c["effort"] <= effort_cap:
                    selected.append(c)
                    total_cost += c["cost"]
                    total_effort += c["effort"]
                    total_years += c["years_gained"]
                    total_age_reduction += c["age_reduction"]
            
            if selected:
                st.success(f"✨ **Optimal Bundle Found!**")
                st.write(f"**Total impact:** Reduce body age by {total_age_reduction:.1f} years, "
                        f"gain {total_years:.1f} healthy years")
                st.write(f"**Total cost:** ${total_cost} | **Total effort:** {total_effort}/50")
                
                st.markdown("#### Selected Interventions:")
                for s in selected:
                    st.write(f"**{s['name']}** — {s['age_reduction']:.1f} years younger "
                           f"(effort {s['effort']}, cost ${s['cost']})")
                    st.caption(s["mission"])
            else:
                st.warning("No interventions fit within your budget and effort constraints. Try increasing them.")

    # ---------- RIGHT: summary ----------
    with col_right:
        st.header("💾 Summary")
        
        st.markdown("### Quick Stats")
        st.write(f"**Age:** {chronological_age} → {predicted_body_age:.1f}")
        st.write(f"**Health Score:** {health_score:.1f}/100")
        st.write(f"**BMI:** {bmi:.1f}")
        
        st.markdown("---")
        
        # Save/Load profiles
        st.markdown("### 📁 Save Profile")
        profile_name = st.text_input("Profile name", value=f"Profile_{chronological_age}")
        if st.button("💾 Save"):
            if "saved_profiles" not in st.session_state:
                st.session_state["saved_profiles"] = {}
            st.session_state["saved_profiles"][profile_name] = dict(p)
            st.success(f"Saved: {profile_name}")
        
        if st.session_state.get("saved_profiles"):
            st.markdown("### 📂 Load Profile")
            for name in st.session_state["saved_profiles"].keys():
                if st.button(f"📄 {name}"):
                    st.session_state["profile"] = dict(st.session_state["saved_profiles"][name])
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📄 Export")
        if st.button("📥 Download Report (PDF)"):
            st.info("PDF export feature coming soon!")

else:
    # -------------------------
    # COMMUNITY MODE
    # -------------------------
    st.header("🌍 Community Mode — Cohort Optimization")
    st.markdown("Upload a cohort CSV with columns: age, bmi, avg_sleep_hours, exercise_frequency, "
                "smoking_status, alcohol_intake, cholesterol_level, blood_sugar_fasting, stress_level")
    
    uploaded = st.file_uploader("Upload cohort CSV", type=["csv"])
    
    if uploaded:
        cohort = pd.read_csv(uploaded)
        st.write("📊 Cohort Preview")
        st.dataframe(cohort.head(10))
        
        # Predict body ages for cohort
        if body_age_model is not None:
            with st.spinner("Calculating body ages for cohort..."):
                cohort["predicted_body_age"] = cohort.apply(lambda row: predict_body_age(row.to_dict()), axis=1)
                cohort["health_score"] = cohort.apply(
                    lambda row: body_age_to_health_score(row["predicted_body_age"], row.get("age", 45)), axis=1
                )
            
            st.success("✅ Predictions complete!")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cohort Size", len(cohort))
            with col2:
                avg_age_diff = (cohort["age"] - cohort["predicted_body_age"]).mean()
                st.metric("Avg Age Difference", f"{avg_age_diff:.1f} years")
            with col3:
                st.metric("Avg Health Score", f"{cohort['health_score'].mean():.1f}/100")
            
            # Policy and budget
            st.markdown("---")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                policy = st.selectbox("Policy Objective", 
                                     ["Maximize total healthy-years", 
                                      "Minimize disparity (equity)",
                                      "Target high-risk individuals"])
            with col_p2:
                total_budget = st.number_input("Total Budget ($)", value=10000, step=1000)
            
            if st.button("🎯 Optimize Cohort Interventions", type="primary"):
                st.markdown("### 📋 Allocation Results")
                
                # Simple allocation: calculate intervention impact per person
                intervention_impacts = []
                for intervention in DEFAULT_INTERVENTIONS:
                    cohort_new = cohort.apply(lambda row: apply_delta_to_profile(row.to_dict(), intervention["delta"]), 
                                             axis=1, result_type='expand')
                    new_body_ages = cohort_new.apply(lambda row: predict_body_age(row.to_dict()), axis=1)
                    
                    avg_reduction = (cohort["predicted_body_age"] - new_body_ages).mean()
                    total_cost = intervention["cost"] * len(cohort)
                    total_years_gained = avg_reduction * len(cohort) * 0.15
                    
                    intervention_impacts.append({
                        "name": intervention["name"],
                        "total_cost": total_cost,
                        "total_years_gained": total_years_gained,
                        "avg_reduction": avg_reduction,
                        "roi": total_years_gained / max(1, total_cost)
                    })
                
                # Sort by ROI
                intervention_impacts = sorted(intervention_impacts, key=lambda x: x["roi"], reverse=True)
                
                # Allocate budget
                selected_interventions = []
                remaining_budget = total_budget
                total_impact = 0.0
                
                for interv in intervention_impacts:
                    if interv["total_cost"] <= remaining_budget:
                        selected_interventions.append(interv)
                        remaining_budget -= interv["total_cost"]
                        total_impact += interv["total_years_gained"]
                
                st.success(f"🎉 **Total healthy-years gained:** {total_impact:.1f} years")
                st.write(f"**Budget used:** ${total_budget - remaining_budget:,.0f} / ${total_budget:,.0f}")
                
                for interv in selected_interventions:
                    st.write(f"**{interv['name']}** — {interv['total_years_gained']:.1f} years gained "
                           f"(${interv['total_cost']:,.0f})")
        else:
            st.error("Model not loaded. Cannot process cohort.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is a demo application. Results are model-based estimates and not medical advice. "
          "Consult healthcare professionals for medical decisions.")
