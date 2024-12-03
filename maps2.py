#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#df must include the following variables:
#prediction of piezo_groundwater_level_category
#piezo_station_update_date
#piezo_station_commune_code_insee
df = pd.read_csv('Book1.csv')




print(df.columns)


# <!-- Modifying format of date and geographical location -->



from datetime import datetime

# Define a function to manually parse the date and extract the required components
def extract_date_components(date_str):
    # Remove the timezone abbreviation ('CEST') from the string
    date_str = ' '.join(date_str.split()[:-2]) + ' ' + ' '.join(date_str.split()[-1:])

    # Convert the date string into a datetime object
    date_obj = datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y')

    # Return formatted date as a datetime object
    formatted_date = date_obj  # This will be a datetime object

    # Extract day, month, and year as integers
    day = int(date_obj.strftime('%d'))  # Day as integer
    month = int(date_obj.strftime('%m'))  # Month as integer
    year = int(date_obj.strftime('%Y'))  # Year as integer

    return formatted_date, day, month, year

# Apply the function to the 'piezo_station_update_date' column
df[['formatted_date', 'day', 'month', 'year']] = df['piezo_station_update_date'].apply(
    lambda x: pd.Series(extract_date_components(x))
)
df['formatted_date'] = pd.to_datetime(df['formatted_date'], format='%d/%m/%Y')
df['formatted_date'] = pd.to_datetime(df['formatted_date'])

# Extract only the date (year-month-day) portion
df['formatted_date'] = df['formatted_date'].dt.date
df['formatted_date'] = pd.to_datetime(df['formatted_date'], errors='coerce')




df['department_code'] = df['department_code'] = df['piezo_station_commune_code_insee'].str[:2]

level_map = {
    'Low': 1,
    'Very Low': 2,
    'Average': 3,
    'High': 4,
    'Very High': 5
}
df['groundwater_level_numeric'] = df['piezo_groundwater_level_category'].map(level_map)





df_new = df.sample(5000)




from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import folium

def generate_department_map(department_code, year, df_new):
    # Filter the communes for the selected department
    gdf_communes = gpd.read_file("/content/drive/My Drive/Python/Hackathon/Final Sprint/Communes.json")
    communes_in_department = gdf_communes[gdf_communes['dep'] == department_code]

    # Filter df_new by the selected year
    df_year = df_new[df_new['year'] == year].copy()

# Calculate the average groundwater level per commune
    commune_avg = df_year.groupby('piezo_station_commune_code_insee')['groundwater_level_numeric'].mean().reset_index()

    #Y'a t'il des données pour cette date?
    if df_year.empty:
       print(f"Aucune donnée disponible pour l'année {year}.")
    else:
          # Étape 2 : Etablir une moyenne de la catégorie
          level_map = {
              'Low': 1,
              'Very Low': 2,
              'Average': 3,
              'High': 4,
              'Very High': 5
          }
          df_year.loc['groundwater_level_numeric'] = 0
          df_year.loc[:,'groundwater_level_numeric'] = df_year['piezo_groundwater_level_category'].map(level_map)

          # Calculate the average groundwater level per commune
          commune_avg = df_year.groupby('station_commune_code_insee')['groundwater_level_numeric'].mean().reset_index()

          # Create a function to assign categories based on the average groundwater level
          def assign_category(avg_value):
              rounded_value = round(avg_value)
              if rounded_value == 1:
                  return 'Very Low'
              elif rounded_value == 2:
                  return 'Low'
              elif rounded_value == 3:
                  return 'Average'
              elif rounded_value == 4:
                  return 'High'
              else:
                  return 'Very High'

          # Apply the function to assign categories
          commune_avg['groundwater_level_category'] = commune_avg['groundwater_level_numeric'].apply(assign_category)

          # Merge the averages and categories with the communes
          communes_in_department = communes_in_department.merge(commune_avg, left_on='codgeo', right_on='piezo_station_commune_code_insee')

          # Étape 3 : Séparer les communes colorées et grises
          gdf_colored = communes_in_department[communes_in_department['groundwater_level_numeric'].notnull()]
          gdf_gray = communes_in_department[communes_in_department['groundwater_level_numeric'].isnull()]

          # Étape 4 : Extraire les coordonnées des centroids
          gdf_colored['centroid'] = gdf_colored.geometry.centroid
          gdf_gray['centroid'] = gdf_gray.geometry.centroid

          # Préparer les données pour le KNN
          X_train = np.array([[point.x, point.y] for point in gdf_colored['centroid']])
          y_train = gdf_colored['groundwater_level_category'].values
          X_predict = np.array([[point.x, point.y] for point in gdf_gray['centroid']])

          # Étape 5 : Appliquer le KNN
          knn = KNeighborsClassifier(n_neighbors=5)  # Vous pouvez ajuster le nombre de voisins
          knn.fit(X_train, y_train)
          y_pred = knn.predict(X_predict)

          # Ajouter les catégories prédites aux communes grises
          gdf_gray['groundwater_level_category'] = y_pred
          gdf_gray['color'] = gdf_gray['groundwater_level_category'].map(assign_color)

          # Fusionner les données colorées et mises à jour
          gdf_updated = pd.concat([gdf_colored, gdf_gray])

          # Create a map centered on the department
          department_center = communes_in_department.geometry.centroid.unary_union.centroid
          department_map = folium.Map(location=[department_center.y, department_center.x], zoom_start=10)

          # # Add communes to the map
          # folium.GeoJson(
          #     communes_in_department,
          #     name="Communes",
          #     style_function=lambda feature: {
          #         'fillColor': category_colors[feature['properties']['groundwater_level_category']],
          #         'color': 'black',
          #         'weight': 0.3,
          #         'fillOpacity': 0.6,
          #     },
          #     tooltip=GeoJsonTooltip(fields=['libgeo', 'groundwater_level_category'],
          #                            aliases=['Commune', 'Niveau moyen']),
          #     highlight_function=lambda x: {'weight': 3, 'color': 'blue'}
          # ).add_to(department_map)

          # return department_map

          for _, row in gdf_updated.iterrows():
              folium.GeoJson(
                  data=row['geometry'].__geo_interface__,
                  style_function=lambda feature, color=row['color']: {
                      'fillColor': color,
                      'color': 'black',
                      'weight': 0.5,
                      'fillOpacity': 0.7
                  },
                  tooltip=f"""
                      <b>Commune :</b> {row['libgeo']}<br>
                      <b>Catégorie :</b> {row['groundwater_level_category']}<br>
                      <b>Année :</b> {year}
                  """
              ).add_to(department_map)

          # Étape 7 : Sauvegarder la nouvelle carte
          department_map.save(f"communes_knn_map_{year}-{department_code}.html")
          print(f"Carte pour l'année {year} générée et sauvegardée sous 'communes_knn_map_{year_to_analyze}.html'")


# <!-- #Final Definitions -->




def assign_color(category):
    color_map = {
        "Very Low": "#ff001e",  # Very Rouge
        "Low": "#ff4f64",  #  Red
        "Average": "#ff858d",  # Mediumm Red
        "High": "#ffc2cb",  # Light red
        "Very High": "#ffe3f0"  # White
    }
    return color_map.get(category, "#A9A9A9")  # Gris par défaut


# <!-- Map de la France entière par communes (not weekly rolling average) -->


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import folium
import geopandas as gpd
import pandas as pd

def generate_groundwater_map_communes(date, department, df):

    # Load geographical data
    geojson_path = "Communes.json"
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf['dep'] == department]  # Filter for the department

    # Map groundwater level categories to numeric values
    level_map = {
        'Low': 1,
        'Very Low': 2,
        'Average': 3,
        'High': 4,
        'Very High': 5
    }
    df['groundwater_level_numeric'] = df['piezo_groundwater_level_category'].map(level_map)

    # Filter data for the specified year
    df_year = df[df['year'] == date]

    if df_year.empty:
        print(f"Aucune donnée disponible pour l'année {date}.")
        return None

    # Calculate average groundwater level per commune
    commune_avg = df_year.groupby('piezo_station_commune_code_insee')['groundwater_level_numeric'].mean().reset_index()

    def assign_category(avg_value):
        rounded_value = round(avg_value)
        return {1: 'Very Low', 2: 'Low', 3: 'Average', 4: 'High', 5: 'Very High'}.get(rounded_value, 'Unknown')

    commune_avg['groundwater_level_category'] = commune_avg['groundwater_level_numeric'].apply(assign_category)

    # Merge with geographical data
    gdf['piezo_station_commune_code_insee'] = gdf['codgeo']  # Ensure matching column names
    gdf = gdf.merge(commune_avg, on='piezo_station_commune_code_insee', how='left')

    # Separate communes with and without data
    gdf_colored = gdf[gdf['groundwater_level_category'].notnull()]
    gdf_gray = gdf[gdf['groundwater_level_category'].isnull()]

    # Extract centroids for KNN
    gdf_colored['centroid'] = gdf_colored.geometry.centroid
    gdf_gray['centroid'] = gdf_gray.geometry.centroid

    # Prepare data for KNN
    X_train = np.array([[point.x, point.y] for point in gdf_colored['centroid']])
    y_train = gdf_colored['groundwater_level_category'].values
    X_predict = np.array([[point.x, point.y] for point in gdf_gray['centroid']])

    # Apply KNN to predict missing categories
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_predict)

    gdf_gray['groundwater_level_category'] = y_pred

    gdf_gray['color'] = gdf_gray['groundwater_level_category'].map(assign_color)
    gdf_colored['color'] = gdf_colored['groundwater_level_category'].map(assign_color)

    # Combine updated data
    gdf_updated = pd.concat([gdf_colored, gdf_gray])

    # Generate the map
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=10)

    for _, row in gdf_updated.iterrows():
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda feature, color=row['color']: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=f"""
                <b>Commune :</b> {row['libgeo']}<br>
                <b>Catégorie :</b> {row['groundwater_level_category']}<br>
                <b>Année :</b> {date}
            """
        ).add_to(m)

    # Save the map
    map_file = f"groundwater_map_{department}_{date}.html"
    m.save(map_file)
    print(f"Map saved as {map_file}")
    return m




# generate_groundwater_map_communes(2024, "10", df)


# <!-- Map de la France entière par departement (weekly rolling average) -->




def calculate_rolling_avg(df):
    # Step 1: Create empty columns to store results
    df['weekly_rolling_avg'] = None
    df['weekly_rolling_avg_category'] = None

    # Step 2: Iterate over each unique piezo_station_pe_label
    for label in df['piezo_station_commune_code_insee'].unique():
        # Filter the DataFrame for the current label
        station_df = df[df['piezo_station_commune_code_insee'] == label].copy()

        # Ensure data is sorted by date for the current label
        station_df = station_df.sort_values(by='formatted_date')

        # Initialize an empty list to store rolling averages for the current label
        rolling_avgs = []

        # Step 3: Iterate over the rows within the filtered DataFrame
        for i in range(len(station_df)):
            # Get the current date
            current_date = station_df.iloc[i]['formatted_date']

            # Define the 7-day window: 3 days before and 3 days after (inclusive)
            start_date = current_date - pd.Timedelta(days=3)
            end_date = current_date + pd.Timedelta(days=3)

            # Filter the rows within the 7-day window
            window_df = station_df[
                (station_df['formatted_date'] >= start_date) &
                (station_df['formatted_date'] <= end_date)
            ]

            # Calculate the mean groundwater level for the window
            rolling_avg = window_df['groundwater_level_numeric'].mean()

            # Append the rolling average to the list
            rolling_avgs.append(rolling_avg)

        # Step 4: Assign the rolling averages back to the station DataFrame
        station_df['weekly_rolling_avg'] = rolling_avgs

        # Map the rolling averages back to categorical values
        station_df['weekly_rolling_avg_category'] = (
            station_df['weekly_rolling_avg']
            .round()
            .clip(1, 5)
            .map({v: k for k, v in level_map.items()})
        )

        # Step 5: Update the original DataFrame with the results
        df.loc[df['piezo_station_commune_code_insee'] == label, 'weekly_rolling_avg'] = station_df['weekly_rolling_avg']
        df.loc[df['piezo_station_commune_code_insee'] == label, 'weekly_rolling_avg_category'] = station_df['weekly_rolling_avg_category']

    return df

# df = calculate_rolling_avg(df)




from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import folium
import geopandas as gpd
import pandas as pd

def generate_groundwater_map_rolling_average_communes(date, department, df): #à ne lancer qu'une fois qu'on a lancé calculate_rolling_avg sur df

    # Load geographical data
    geojson_path = "Communes.json"
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf['dep'] == department]  # Filter for the department

    # Map groundwater level categories to numeric values
    level_map = {
        'Low': 1,
        'Very Low': 2,
        'Average': 3,
        'High': 4,
        'Very High': 5
    }
    #df['weekly_rolling_avg'] = df['weekly_rolling_avg_category'].map(level_map)

    # Filter data for the specified year
    df_year = df[df['formatted_date'] == date]

    if df_year.empty:
        print(f"Aucune donnée disponible pour la date {date}.")
        return None

    # Calculate average groundwater level per commune
    commune_avg = df_year.groupby('piezo_station_commune_code_insee')['weekly_rolling_avg'].mean().reset_index()

    def assign_category(avg_value):
        rounded_value = round(avg_value)
        return {1: 'Very Low', 2: 'Low', 3: 'Average', 4: 'High', 5: 'Very High'}.get(rounded_value, 'Unknown')

    commune_avg['groundwater_level_rolling_category'] = commune_avg['weekly_rolling_avg'].apply(assign_category)

    # Merge with geographical data
    gdf['piezo_station_commune_code_insee'] = gdf['codgeo']  # Ensure matching column names
    gdf = gdf.merge(commune_avg, on='piezo_station_commune_code_insee', how='left')

    # Separate communes with and without data
    gdf_colored = gdf[gdf['groundwater_level_rolling_category'].notnull()]
    gdf_gray = gdf[gdf['groundwater_level_rolling_category'].isnull()]

    # Extract centroids for KNN
    gdf_colored['centroid'] = gdf_colored.geometry.centroid
    gdf_gray['centroid'] = gdf_gray.geometry.centroid

    # Prepare data for KNN
    X_train = np.array([[point.x, point.y] for point in gdf_colored['centroid']])
    y_train = gdf_colored['groundwater_level_rolling_category'].values
    X_predict = np.array([[point.x, point.y] for point in gdf_gray['centroid']])

    # Apply KNN to predict missing categories
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_predict)

    gdf_gray['groundwater_level_rolling_category'] = y_pred

    gdf_gray['color'] = gdf_gray['groundwater_level_rolling_category'].map(assign_color)
    gdf_colored['color'] = gdf_colored['groundwater_level_rolling_category'].map(assign_color)

    # Combine updated data
    gdf_updated = pd.concat([gdf_colored, gdf_gray])

    # Generate the map
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=9)

    for _, row in gdf_updated.iterrows():
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda feature, color=row['color']: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=f"""
                <b>Commune :</b> {row['libgeo']}<br>
                <b>Catégorie :</b> {row['groundwater_level_rolling_category']}<br>
                <b>Année :</b> {date}
            """
        ).add_to(m)

    # Save the map
    map_file = f"groundwater_map_{department}_{date}.html"
    m.save(map_file)
    print(f"Map saved as {map_file}")
    return m




# generate_groundwater_map_rolling_average_communes('2024-06-28', "33", df)


# <!-- Map de la France entière par departement (not weekly rolling average) -->




from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import folium
import geopandas as gpd
import pandas as pd

def groundwater_levels_map_France(date, df):

    # Load geographical data
    geojson_path = "Departements.geojson"
    gdf = gpd.read_file(geojson_path)

    # Map groundwater level categories to numeric values
    level_map = {
        'Low': 1,
        'Very Low': 2,
        'Average': 3,
        'High': 4,
        'Very High': 5
    }
    df['groundwater_level_numeric'] = df['piezo_groundwater_level_category'].map(level_map)

    # Filter data for the specified year
    df_year = df[df['year'] == date]

    if df_year.empty:
        print(f"Aucune donnée disponible pour la date {date}.")
        return None

    # Calculate average groundwater level per commune
    commune_avg = df_year.groupby('department_code')['groundwater_level_numeric'].mean().reset_index()

    def assign_category(avg_value):
        rounded_value = round(avg_value)
        return {1: 'Very Low', 2: 'Low', 3: 'Average', 4: 'High', 5: 'Very High'}.get(rounded_value, 'Unknown')

    commune_avg['groundwater_level_category'] = commune_avg['groundwater_level_numeric'].apply(assign_category)

    # Merge with geographical data
    gdf['department_code'] = gdf['code']  # Ensure matching column names
    gdf = gdf.merge(commune_avg, on='department_code', how='left')

    # Separate communes with and without data
    gdf_colored = gdf[gdf['groundwater_level_category'].notnull()]
    gdf_gray = gdf[gdf['groundwater_level_category'].isnull()]

    # Extract centroids for KNN
    gdf_colored['centroid'] = gdf_colored.geometry.centroid
    gdf_gray['centroid'] = gdf_gray.geometry.centroid

    # Prepare data for KNN
    X_train = np.array([[point.x, point.y] for point in gdf_colored['centroid']])
    y_train = gdf_colored['groundwater_level_category'].values
    X_predict = np.array([[point.x, point.y] for point in gdf_gray['centroid']])

    # Apply KNN to predict missing categories
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_predict)

    gdf_gray['groundwater_level_category'] = y_pred

    gdf_gray['color'] = gdf_gray['groundwater_level_category'].map(assign_color)
    gdf_colored['color'] = gdf_colored['groundwater_level_category'].map(assign_color)

    # Combine updated data
    gdf_updated = pd.concat([gdf_colored, gdf_gray])

    # Generate the map
    center = [46.603354, 1.888334]
    m = folium.Map(location=center, zoom_start=6)

    for _, row in gdf_updated.iterrows():
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda feature, color=row['color']: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=f"""
                <b>Commune :</b> {row['nom']}<br>
                <b>Catégorie :</b> {row['groundwater_level_category']}<br>
                <b>Année :</b> {date}
            """
        ).add_to(m)

    # Save the map
    map_file = f"groundwater_map_{date}.html"
    m.save(map_file)
    print(f"Map saved as {map_file}")
    return m



# groundwater_levels_map_France(2024, df)


# <!-- Map de la France entière par departement (weekly rolling average) -->


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import folium
import geopandas as gpd
import pandas as pd

def groundwater_levels_map_rolling_France(date, df): #à ne lancer qu'après la fonction calculate_rolling_average

    # Load geographical data
    geojson_path = "Departements.geojson"
    gdf = gpd.read_file(geojson_path)

    # Filter data for the specified year
    df_year = df[df['formatted_date'] == date]

    if df_year.empty:
        print(f"Aucune donnée disponible pour la date {date}.")
        return None

    # Calculate average groundwater level per commune
    commune_avg = df_year.groupby('department_code')['weekly_rolling_avg'].mean().reset_index()

    def assign_category(avg_value):
        rounded_value = round(avg_value)
        return {1: 'Very Low', 2: 'Low', 3: 'Average', 4: 'High', 5: 'Very High'}.get(rounded_value, 'Unknown')

    commune_avg['groundwater_level_category'] = commune_avg['weekly_rolling_avg'].apply(assign_category)

    # Merge with geographical data
    gdf['department_code'] = gdf['code']  # Ensure matching column names
    gdf = gdf.merge(commune_avg, on='department_code', how='left')

    # Separate communes with and without data
    gdf_colored = gdf[gdf['groundwater_level_category'].notnull()]
    gdf_gray = gdf[gdf['groundwater_level_category'].isnull()]

    # Extract centroids for KNN
    gdf_colored['centroid'] = gdf_colored.geometry.centroid
    gdf_gray['centroid'] = gdf_gray.geometry.centroid

    # Prepare data for KNN
    X_train = np.array([[point.x, point.y] for point in gdf_colored['centroid']])
    y_train = gdf_colored['groundwater_level_category'].values
    X_predict = np.array([[point.x, point.y] for point in gdf_gray['centroid']])

    # Apply KNN to predict missing categories
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_predict)

    gdf_gray['groundwater_level_category'] = y_pred

    gdf_gray['color'] = gdf_gray['groundwater_level_category'].map(assign_color)
    gdf_colored['color'] = gdf_colored['groundwater_level_category'].map(assign_color)

    # Combine updated data
    gdf_updated = pd.concat([gdf_colored, gdf_gray])

    # Generate the map
    center = [46.603354, 1.888334]
    m = folium.Map(location=center, zoom_start=6)

    for _, row in gdf_updated.iterrows():
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda feature, color=row['color']: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=f"""
                <b>Commune :</b> {row['nom']}<br>
                <b>Catégorie :</b> {row['groundwater_level_category']}<br>
                <b>Année :</b> {date}
            """
        ).add_to(m)

    # Save the map
    map_file = f"groundwater_map_{date}.html"
    m.save(map_file)
    print(f"Map saved as {map_file}")
    return m




# groundwater_levels_map_rolling_France("2024-06-28", df)


# <!-- Streamlit interface per year -->




# !pip install streamlit folium streamlit-folium
import streamlit as st
from streamlit_folium import st_folium
import folium
import os 


output_dir = "pre_generated_maps"
os.makedirs(output_dir, exist_ok=True)



# Interface Streamlit
st.title("Cartes de la France par Départements")
year = st.selectbox("Sélectionnez l'année", [2020, 2021, 2022, 2023, 2024])
st.write(f"Vous avez sélectionné l'année: {year}")

for year in [2020, 2021, 2022, 2023, 2024]:
    print(f"Génération de la carte pour l'année {year}...")
    map_object = groundwater_levels_map_France(year, df)  # Appelez votre fonction pour générer la carte
    output_path = os.path.join(output_dir, f"groundwater_map_{year}.html")
    map_object.save(output_path)  # Sauvegarde de la carte en HTML
    print(f"Carte sauvegardée : {output_path}")

def load_saved_map(year):
    map_path = f"pre_generated_maps/groundwater_map_{year}.html"
    if os.path.exists(map_path):
        with open(map_path, "r") as file:
            return file.read()
    else:
        return "<p>Carte introuvable pour l'année sélectionnée.</p>"


# Carte principale
st.write("Cliquez sur un département pour voir les communes.")
main_map = load_saved_map(year)  # Define the main map
main_map_data = st_folium(main_map, height=500, width=800)  # Render the map and capture interaction

# Handle interaction or lack thereof
if main_map_data and 'last_active_drawing' in main_map_data:
    # If there is interaction, extract the clicked department
    if main_map_data['last_active_drawing']:
        clicked_dept = main_map_data['last_active_drawing']['properties'].get('code', None)
        if clicked_dept:
            st.write(f"Vous avez cliqué sur : {clicked_dept}")
            st.write("Carte des communes dans le département :")
            dept_map = generate_groundwater_map_communes(year, clicked_dept, df)  # Generate dept map
            st_folium(dept_map, height=500, width=800)  # Render dept map
        else:
            st.error("Le champ 'code' est manquant dans 'properties'.")
else:
    # If no interaction, display an initial message
    st.write("Aucun département n'a encore été cliqué. Interagissez avec la carte pour commencer.")





# !streamlit run C:\Users\tilia\AppData\Roaming\Python\Python312\site-packages\ipykernel_launcher.py


# <!-- Streamlit interface with rolling average (unfinished) -->



# # Interface Streamlit
# st.title("Cartes de la France par Départements")
# formatted_date = st.selectbox("Sélectionnez la date", [2020, 2021, 2022, 2023, 2024]) #modifier pour avoir une liste de toutes les dates (ou alors pouvoir selectionner le jour mois année séparémment)
# st.write(f"Vous avez sélectionné l'année: {formatted_date}") #year sera "formatted_date"

# # Carte principale
# st.write("Cliquez sur un département pour voir les communes.")
# main_map = groundwater_levels_map_France('formatted_date', df)  # Define the main map
# main_map_data = st_folium(main_map, height=500, width=800)  # Render the map and capture interaction

# # Handle interaction or lack thereof
# if main_map_data and 'last_active_drawing' in main_map_data:
#     # If there is interaction, extract the clicked department
#     if main_map_data['last_active_drawing']:
#         clicked_dept = main_map_data['last_active_drawing']['properties'].get('code', None)
#         if clicked_dept:
#             st.write(f"Vous avez cliqué sur : {clicked_dept}")
#             st.write("Carte des communes dans le département :")
#             dept_map = generate_groundwater_map_communes('formatted_date', clicked_dept, df)  # Generate dept map
#             st_folium(dept_map, height=500, width=800)  # Render dept map
#         else:
#             st.error("Le champ 'code' est manquant dans 'properties'.")
# else:
#     # If no interaction, display an initial message
#     st.write("Aucun département n'a encore été cliqué. Interagissez avec la carte pour commencer.")




# # Define the notebook path (Colab stores it as a temporary file)
# notebook_path = 'maps.py'

# # Convert the notebook to a .py file
# !jupyter nbconvert --to script Clean_Definitions_of_maps.ipynb --output "maps.py"

# print(f"Notebook saved as Python file: {notebook_path}")



# !mv maps.py.txt maps.py



# !streamlit run 'maps.py'




# !pip install pyngrok
# from pyngrok import ngrok
# ngrok config add-authtoken 2paNmEvatHeukoCXn5OftjbVpgw_3bQ66Gqh2xfnKaK2BQGs3

# # Start Streamlit app in background
# !streamlit run '/content/drive/My Drive/Python/Hackathon/Final Sprint/maps.py' &

# # Open a tunnel to the Streamlit app
# public_url = ngrok.connect(8501)
# print('Streamlit app is live at:', public_url)

