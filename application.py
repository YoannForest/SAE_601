"""
📝 **Instructions** :
- Installez toutes les bibliothèques nécessaires en fonction des imports présents dans le code, utilisez la commande suivante :conda create -n projet python pandas numpy ..........
- Complétez les sections en écrivant votre code où c’est indiqué.
- Ajoutez des commentaires clairs pour expliquer vos choix.
- Utilisez des emoji avec windows + ;
- Interprétez les résultats de vos visualisations (quelques phrases).
"""

### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shutil
import plotly.express as px


# Chargement des données
st.title("📊 Visualisation des Salaires en Data Science")
liste = ["Question 2","Question 3","Question 4","Question 5","Question 6","Question 7","Question 8","Question 9","Question 10"]
tabs = st.tabs(liste)


# Download latest version

df = pd.read_csv("https://raw.githubusercontent.com/YoannForest/SAE_601/refs/heads/main/ds_salaries.csv")
# Import des données depuis le fichier du repository

#pour télécharger le dataset sur la racine du lecteur C
#path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")
#project_dir = os.getcwd()
# Créer le répertoire (si nécessaire)
#os.makedirs("data", exist_ok=True)

# Copier le fichier depuis le cache au répertoire du projet
#cache_file = path+"\\ds_salaries.csv"
#shutil.copy(cache_file,f"{project_dir}data\\ds_salaries.csv")

#df = pd.read_csv(f"{project_dir}data\\ds_salaries.csv")



with tabs[0]:
    ### 2. Exploration visuelle des données
    #votre code 

    st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")


    if st.checkbox("Afficher un aperçu des données"):
        st.write(df.head())


    #Statistique générales avec describe pandas 
    #votre code 
    st.subheader("📌 Statistiques générales")
    st.write(df.describe())
    st.write("Affichage du dataset avec la fonction describe de pandas")


with tabs[1]:
    ### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
    #votre code 
    st.subheader("📈 Distribution des salaires en France")
    st.plotly_chart(px.box(df.query("company_location == 'FR'"),y='salary_in_usd',x = 'job_title', color = 'experience_level'))
    st.write("Boxplot avec la fonction box de plotly express représentant le salaire en fonction du titre")



with tabs[2]:
    ### 4. Analyse des tendances de salaires :
    var = st.selectbox('var',options=['experience_level', 'employment_type', 'job_title', 'company_location'],label_visibility = 'hidden')  
    st.plotly_chart(px.bar(df.groupby(var,as_index=False)[["salary_in_usd"]].mean().sort_values("salary_in_usd",ascending = False), y = 'salary_in_usd',x = var,title = f"Analyse des tendances de salaires par {var}"))
    st.write("Histogramme avec le salaire en fonction de la variable sélectionnée")

with tabs[3]:
    ### 5. Corrélation entre variables
    numeric_df = df.select_dtypes(include=[np.number])  # Sélectionner uniquement les colonnes numériques

    # Calcul de la matrice de corrélation
    #votre code
    correlation_matrix = numeric_df.corr()

    # Affichage du heatmap avec sns.heatmap
    #votre code 
    fig3, ax = plt.subplots()
    st.subheader("🔗 Corrélations entre variables numériques")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax = ax)
    st.pyplot(fig3)
    st.write("Affichage d'une carte de chaleur de seaborn avec la matrice des corrélations")


with tabs[4]:
    ### 6. Analyse interactive des variations de salaire
    # Une évolution des salaires pour les 10 postes les plus courants
    # count of job titles pour selectionner les postes
    # calcule du salaire moyen par an
    #utilisez px.line
    #votre code 

    top = df['job_title'].value_counts().nlargest(10).index
    moyenne_salaire = df.groupby(["job_title","work_year"], as_index=False)[["salary_in_usd"]].mean()
    moyenne_salaire_top = moyenne_salaire["job_title"].isin(top)
    st.plotly_chart(px.line(moyenne_salaire[moyenne_salaire_top],x="work_year",y="salary_in_usd",color='job_title'))
    st.write("Courbes des moyennes de salaires par années des 10 titres les plus présents")

with tabs[5]:
    ### 7. Salaire médian par expérience et taille d'entreprise
    # utilisez median(), px.bar
    #votre code 

    st.plotly_chart(px.bar(df.groupby(["experience_level","company_size"])[["salary_in_usd"]].median().reset_index(),x = 'company_size', y = 'salary_in_usd', barmode = 'group', color = 'experience_level'))
    st.write("Histogramme du salaire moyen par taille d'entreprise et niveau d'experience")

with tabs[6]:
    ### 8. Ajout de filtres dynamiques
    #Filtrer les données par salaire utilisant st.slider pour selectionner les plages 
    #votre code 
    low, high = st.slider('Slider des salaires',min_value = df["salary_in_usd"].min(),max_value=df["salary_in_usd"].max(),value=(0,100000))
    st.write("Nombre de lignes : "+str(df.query(f"salary_in_usd >= {low} and salary_in_usd <= {high}").shape[0]))
    st.write(df.query(f"salary_in_usd >= {low} and salary_in_usd <= {high}"))
    st.write(f"Dataframe contenant uniquement les données de salaires comprises entre {str(low)} et {str(high)}")


with tabs[7]:
    ### 9.  Impact du télétravail sur le salaire selon le pays
    #salaire moyen pour 0-50-100 par pays
    df_ratio_moyen = df.groupby(["remote_ratio","company_location"])[["salary_in_usd"]].mean().reset_index()
    df_ratio_moyen["remote_ratio"] = pd.Categorical(df_ratio_moyen["remote_ratio"])
    nb_pays = st.slider('Nombre de pays',min_value = 3,max_value=pd.DataFrame(df_ratio_moyen.groupby("company_location")).shape[0],value=10)
    st.plotly_chart(px.bar(df_ratio_moyen.sort_values("salary_in_usd",ascending = False).head(nb_pays), x = "company_location", y = "salary_in_usd", barmode = "group",color = "remote_ratio"))
    st.write("Salaire moyen par pays en fonction du ratio de télétravail")


with tabs[8]:
    ### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
    #votre code 

    exp = st.multiselect("Sélectionnez le niveau d'expérience",list( df['experience_level'].unique()))
    size = st.multiselect("Sélectionnez la taille d'entreprise",list( df['company_size'].unique()))
    if exp ==[] or size == []:
        st.write("Veullez choisir des modalités")
    if exp ==[] or size == []:
        df_filtre= df
    else:
        df_filtre=df.query(f"experience_level in {exp} and company_size in {size}")
    st.write("Nombre de lignes : "+str(df_filtre.shape[0]))
    st.write(df_filtre)
    st.write("Dataframe contenant les données avec le niveau d'experience et la taille d'entreprise choisis")

