import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.set_page_config(page_title="Stack Overflow Survey", layout="wide")

st.title("Stack Overflow Survey Analizi")
st.write("Bu uygulama Stack Overflow 2025 yılı developer survey anketinin verilerini analiz eder.")

# 1. Veri Yükleme
@st.cache_data
def load_data():
    return pd.read_csv("survey_results_public.csv")

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
st.header("Sayısal Sütünların Listesi:")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
st.write("Sayısal Sütunlar:", numeric_cols)

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
