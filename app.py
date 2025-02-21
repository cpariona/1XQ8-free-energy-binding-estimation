import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from padelpy import from_smiles
# from PaDEL_pywrapper import PaDEL
# from PaDEL_pywrapper import descriptors
import numpy as np
import pickle
import joblib

st.title("Test de ML para ligando-receptor")

compound_smiles=st.text_input('Ingresa tu código SMILES','c1cccc(NC2=O)c1[C@]23[C@@]4(C)c5n([C@@H](C3)C(=O)N4)c(=O)c6c(n5)cccc6')
mm = Chem.MolFromSmiles(compound_smiles)

Draw.MolToFile(mm,'mol.png')
st.image('mol.png')

#######
RDKit_select_descriptors = joblib.load('./archivos/RDKit_select_descriptors.pickle')
PaDEL_select_descriptors = joblib.load('./archivos/PaDEL_select_descriptors.pickle')
robust_scaler = joblib.load('./archivos/robust_scaler.pickle')
minmax_scaler = joblib.load('./archivos/minmax_scaler.pickle')
#selector_lgbm = joblib.load('./archivos/selector_LGBM.pickle')
#lgbm_model = joblib.load('./archivos/lgbm_best_model.pickle')
selector_lgbm = joblib.load('./archivos/selector_LGBM.joblib')
lgbm_model = joblib.load('./archivos/lgbm_best_model.joblib')


# RDKit selected descriptors function
def get_selected_RDKitdescriptors(smile, selected_descriptors, missingVal=None):
    ''' Calculates only the selected descriptors for a molecule '''
    res = {}
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return {desc: missingVal for desc in selected_descriptors}

    for nm, fn in Descriptors._descList:
        if nm in selected_descriptors:
            try:
                res[nm] = fn(mol)
            except:
                import traceback
                traceback.print_exc()
                res[nm] = missingVal
    return res

df = pd.DataFrame({'smiles': [compound_smiles]})
#st.dataframe(df)

# Calculate selected RDKit descriptors
RDKit_descriptors = [get_selected_RDKitdescriptors(m, RDKit_select_descriptors) for m in df['smiles']]
RDKit_df = pd.DataFrame(RDKit_descriptors)
st.write("Descriptores RDKit")
st.dataframe(RDKit_df)

# Calculate PaDEL descriptors
PaDEL_descriptors = from_smiles(df['smiles'].tolist())
PaDEL_df_ = pd.DataFrame(PaDEL_descriptors)
PaDEL_df = PaDEL_df_.loc[:,PaDEL_select_descriptors]
st.write("Descriptores PaDEL")
st.dataframe(PaDEL_df)

# Concatenate RDKit and PaDEL dataframes
RDKit_PaDEL_df = pd.concat([RDKit_df, PaDEL_df], axis=1)
RDKit_PaDEL_df_columns = RDKit_PaDEL_df.columns

# Scale data
RDKit_PaDEL_scaled_ = robust_scaler.transform(RDKit_PaDEL_df)
RDKit_PaDEL_scaled = minmax_scaler.transform(RDKit_PaDEL_scaled_)
RDKit_PaDEL_scaled_df = pd.DataFrame(RDKit_PaDEL_scaled)
RDKit_PaDEL_scaled_df.columns = RDKit_PaDEL_df_columns

# Selected features
selected_features_mask = selector_lgbm.support_
Selected_features = RDKit_PaDEL_df_columns[selected_features_mask]
RDKit_PaDEL = RDKit_PaDEL_scaled_df[Selected_features]

# Make predictions
predictions = lgbm_model.predict(RDKit_PaDEL)
st.write("Free Energy Binding Prediction")
st.dataframe(predictions)
