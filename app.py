import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.set_page_config(page_title="Developer Survey Analizi", layout="wide")

st.title("Stack Overflow Survey Analizi")
st.write("Bu uygulama Stack Overflow 2025 yılı developer survey anketinin verilerini analiz eder.")

# 1. Veri Yükleme
@st.cache_data
def load_data():
    return pd.read_csv("survey_results_public.zip")

df = load_data()

# 2. Veri Seti Hakkında Bilgi
st.header("1. Veri Seti Genel Bakış")
st.write(f"**Satır Sayısı:** {df.shape[0]}")
st.write(f"**Sütun Sayısı:** {df.shape[1]}")
st.write("**Sütunlar:**", list(df.columns))

# 3. Ham Veri Gösterimi
st.header("2. Ham Veri")
satir_sayisi = st.slider("Gösterilecek satır sayısı:", 5, 50, 10)
st.dataframe(df.head(satir_sayisi))

# 4. İstatistiksel Özet
st.header("3. İstatistiksel Özet")
if st.checkbox("İstatistikleri Göster"):
    st.write(df.describe())

# 5. Eksik Değerler
st.header("4. Eksik Değerler")
st.write(df.isnull().sum())

# 6. Sayısal Değişken Analizi
st.header("Sayısal Sütunların Listesi:")
all_numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# ResponseId ve Hedef Değişkeni (JobSat) analizden ve PCA'dan çıkaralım
cols_to_exclude = ['ResponseId', 'JobSat', 'JobSatisfaction']
numeric_cols = [col for col in all_numeric_cols if col not in cols_to_exclude]

st.write("Analize Dahil Edilen Sayısal Sütunlar:", numeric_cols)

if len(numeric_cols) > 0:
    # Korelasyon Matrisi koyacaktım ama cok karmasık oluyor column sayısı fazla oldugundan koymadım hocam.

    # Standardizasyon ve PCA 
    st.header("6. Standardizasyon ve PCA")
    
    # Eksik verileri çıkarma
    df_numeric = df[numeric_cols].dropna()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    
    st.write("PCA Varyans Oranı:", pca.explained_variance_ratio_)
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA Dağılımı')
    st.pyplot(fig2)

# 7. Kategorik Değişken Analizi 
st.sidebar.header("Kategorik Analiz")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

selected_col = st.sidebar.selectbox("Bir sütun seçin:", categorical_cols)

if selected_col:
    st.header(f"7. Kategorik Dağılım: {selected_col}")
    value_counts = df[selected_col].value_counts().head(10) 
    
    fig3, ax3 = plt.subplots()
    ax3.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    ax3.set_title(f"{selected_col} Dağılımı")
    st.pyplot(fig3)

# 8. Random Forest Regression (JobSat Tahmini)
st.header("8. Job Satisfaction Tahmini (Random Forest)")

target_col = 'JobSat' if 'JobSat' in df.columns else 'JobSatisfaction'

if target_col in df.columns:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    st.write(f"Hedef Değişken: **{target_col}**")
    
    # Veri Hazırlığı
    # Sayısal sütunlar + Hedef değişkeni al, NaN'ları at
    rf_data = df[numeric_cols + [target_col]].dropna()
    
    if len(rf_data) > 0:
        X = rf_data[numeric_cols]
        y = rf_data[target_col]
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"Model Eğitim Verisi Boyutu: {X_train.shape}")
        
        if st.button("Random Forest Modelini Eğit"):
            with st.spinner('Model eğitiliyor...'):
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                
                y_pred = rf_model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**R2 Score:** {r2:.2f}")
                
                # Gerçek vs Tahmin Grafiği
                fig4, ax4 = plt.subplots()
                ax4.scatter(y_test, y_pred, alpha=0.5)
                ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax4.set_xlabel('Gerçek Değerler')
                ax4.set_ylabel('Tahmin Edilen Değerler')
                ax4.set_title('Random Forest: Gerçek vs Tahmin')
                st.pyplot(fig4)
                
                # Feature Importance
                st.subheader("Özellik Önem Düzeyleri (Feature Importance)")
                feature_importances = pd.Series(rf_model.feature_importances_, index=numeric_cols).sort_values(ascending=False).head(10)
                st.bar_chart(feature_importances)
    else:
        st.warning("Model eğitimi için yeterli temiz veri yok.")
else:
    st.error("Job Satisfaction sütunu bulunamadı.")
