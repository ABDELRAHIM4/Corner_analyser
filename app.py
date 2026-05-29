import streamlit as st
import joblib
import numpy as np
import pandas as pd
st.set_page_config(
    page_title='Corner Kick Predictor',
)

def load_model():
    model = joblib.load('models.pkl')
    return model
def load_data():
    df = pd.read_csv('corners_data.csv')
    return df
def all_data():
    data = joblib.load('data.pkl')
    return data

data = all_data()
model = load_model()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("Enter The Corner data")
    team_listed = sorted(data['teams'])
    team = st.selectbox(
        "Choose the team",
        team_listed,

    
    )
    
    player = st.selectbox(
            "Choose the player",
            data['player']
        )
    minute = st.slider(
        "minute",
        min_value=1,
        max_value=120,
        help= 'Chosse the corner time'
    )
    #location for corner
    col_x, col_y = st.columns(2)
    loc_x = 0.0
    with col_x:
        side = st.radio(
            "location of corner",
            options=["Right Side", "Left Side"],

        )
        
        if side == "Right Side":
            loc_X = 90.0
        else:
            loc_x = 10.0
    with col_y:
        loc_y = st.number_input(
            "Vertical distance from the goal line to where the corner is taken along the sideline.",
            min_value = 0.0,
            max_value = 120.0,
            step = 1.0,
            format="%.1f"

        )
    under_press = st.checkbox(
        "Player under pressure ??",
        help="Is the corner taker being pressured by defenders?"
    )
    st.markdown("---")
    predict_button = st.button("Predict Goal", type='primary', use_container_width=True)
    with col2:
        st.markdown("prediction results")
        if predict_button:
            team_code = data['coded_team'][team]
            player_code = data[data['coded_team'] == team_code]['coded_player'][player]
            input_data = np.array([[
                minute, team_code, player_code, float(loc_x), loc_y, int(under_press)

            ]])
            
            st.markdown("predictions")
            res = []
            for name, mod in model.items():
                prob = mod.predict_proba(input_data)[0][1]
                pred = "will score" if prob > 0.5 else "will not score"
                res.append({
                    "model": name,
                    'Probability' : prob,
                    'prediction': pred
                })

            best_model = data['best_model']
            best_prob = best_model.predict_proba(input_data)[0][1]
            if best_prob > 0.8:
                st.success(f"🟢 HIGH PROBABILITY ({best_prob:.1%}) - Long corner recommended")
            elif best_prob > 0.4:
                st.warning(f"🟡 MEDIUM PROBABILITY ({best_prob:.1%}) - Inswinger recommended")
            else:
                st.info(f"🔴 LOW PROBABILITY ({best_prob:.1%}) - Short corner recommended")

