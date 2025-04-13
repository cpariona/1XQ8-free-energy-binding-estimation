import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from padelpy import from_smiles
import joblib
import matplotlib.pyplot as plt

# Cargar archivos del modelo
RDKit_select_descriptors = joblib.load('./archivos/RDKit_select_descriptors.pickle')
PaDEL_select_descriptors = joblib.load('./archivos/PaDEL_select_descriptors.pickle')
robust_scaler = joblib.load('./archivos/robust_scaler.pickle')
minmax_scaler = joblib.load('./archivos/minmax_scaler.pickle')
selector_lgbm = joblib.load('./archivos/selector_LGBM.joblib')
lgbm_model = joblib.load('./archivos/lgbm_best_model.joblib')

# ------------------------- INTERFAZ DE LA APP -------------------------

st.title("🧪 Predicción de Afinidad Energética de Ligandos hacia α-Sinucleína (1XQ8)")
st.markdown("""
Esta herramienta permite estimar la **afinidad energética (ΔG)** de moléculas candidatas hacia la proteína **alfa-sinucleína (αSyn)**, implicada en el desarrollo de la **enfermedad de Parkinson**.

🧪 El modelo fue entrenado a partir de los resultados de **docking molecular**, realizados con compuestos filtrados de las bases de datos **PubChem**, **DrugBank** y **UNPD**, acoplados a la proteína **alfa-sinucleína humana (PDB ID: 1XQ8)**.

📍 El sitio de unión seleccionado abarca el rango de residuos **61 a 100**, reconocido como una región clave en el proceso de agregación de la αSyn.

🔬 Para representar numéricamente las propiedades moleculares, se calcularon **129 descriptores** de **RDKit** y **928 descriptores** de **PaDEL**. Tras un proceso de selección de características, se conservaron **104 descriptores de RDKit** y **426 de PaDEL**, considerados los más relevantes para la predicción.

🤖 Con estos datos, se entrenó un modelo de **aprendizaje automático** utilizando el algoritmo **LGBM Regressor**, aplicando validación cruzada y optimización de hiperparámetros mediante **Grid Search**. El modelo final alcanzó un coeficiente de determinación **R² = 0.7796**, indicando un excelente desempeño predictivo.
""")


# ------------------------- ENTRADA DEL USUARIO -------------------------

compound_smiles = st.text_input("🔬 Ingresa el código SMILES de tu molécula:", 
                                'c1cccc(NC2=O)c1[C@]23[C@@]4(C)c5n([C@@H](C3)C(=O)N4)c(=O)c6c(n5)cccc6')

mol = Chem.MolFromSmiles(compound_smiles)
if mol:
    Draw.MolToFile(mol, 'mol.png')
    st.image('mol.png', caption="Representación molecular")

    # ------------------------- FUNCIONES -------------------------

    def get_selected_RDKitdescriptors(smile, selected_descriptors, missingVal=None):
        res = {}
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return {desc: missingVal for desc in selected_descriptors}
        for nm, fn in Descriptors._descList:
            if nm in selected_descriptors:
                try:
                    res[nm] = fn(mol)
                except:
                    res[nm] = missingVal
        return res

    # ------------------------- PROCESAMIENTO -------------------------

    df = pd.DataFrame({'smiles': [compound_smiles]})

    # RDKit
    with st.spinner("Calculando descriptores RDKit..."):
        RDKit_descriptors = [get_selected_RDKitdescriptors(m, RDKit_select_descriptors) for m in df['smiles']]
        RDKit_df = pd.DataFrame(RDKit_descriptors)
    st.subheader("📊 Descriptores RDKit seleccionados")
    st.dataframe(RDKit_df)

    # PaDEL
    with st.spinner("Calculando descriptores PaDEL..."):
        PaDEL_descriptors = from_smiles(df['smiles'].tolist())
        PaDEL_df_ = pd.DataFrame(PaDEL_descriptors)
        PaDEL_df = PaDEL_df_.loc[:, PaDEL_select_descriptors]
    st.subheader("📊 Descriptores PaDEL seleccionados")
    st.dataframe(PaDEL_df)

    # Concatenar y escalar
    combined_df = pd.concat([RDKit_df, PaDEL_df], axis=1)
    scaled_robust = robust_scaler.transform(combined_df)
    scaled_final = minmax_scaler.transform(scaled_robust)
    scaled_df = pd.DataFrame(scaled_final, columns=combined_df.columns)

    # Selección de características
    selected_features_mask = selector_lgbm.support_
    selected_columns = scaled_df.columns[selected_features_mask]
    input_features = scaled_df[selected_columns]

    # Predicción
    prediction = lgbm_model.predict(input_features)[0]

    # ------------------------- RESULTADO FINAL -------------------------
    st.markdown("## 🧠 Resultado del modelo de ML")
    st.metric(label="🔋 Estimación de Afinidad Energética (ΔG)", value=f"{prediction:.2f} kcal/mol")
    st.markdown("Modelo entrenado con **LGBM Regressor**, con un rendimiento R² = **0.7796** en validación cruzada.")

else:
    st.error("Por favor, ingresa un SMILES válido.")

