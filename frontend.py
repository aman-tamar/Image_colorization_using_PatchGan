import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000/colorize"

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Colorizer",
    page_icon="🎨",
    layout="wide"
)

# ----------------------------------
# Custom CSS Styles
# ----------------------------------
st.markdown("""
<style>

body {
    background: #0f0f0f;
}

h1, h2, h3, h4 {
    font-family: 'Poppins';
}

.section {
    padding: 30px;
    margin-top: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(10px);
}

.header {
    padding: 40px;
    text-align: center;
    border-radius: 20px;
    background: linear-gradient(to right, #7928CA, #FF0080);
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 0 25px #7928CA55;
}

.result-card {
    padding: 12px;
    border-radius: 15px;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.14);
}

.footer {
    margin-top: 40px;
    padding: 10px;
    text-align: center;
    color: #bbbbbb;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------
st.markdown("""
<div class="header">
    <h1>🎨 Edge-Aware GAN Image Colorizer</h1>
    <h4>Sharper, cleaner & vivid colorization powered by AI</h4>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# INTRODUCTION SECTION
# ----------------------------------
with st.container():
    st.markdown("""
    <div class="section">
        <h2>📘 About This Project</h2>
        <p style='font-size:17px; color:#dddddd;'>
        This project uses an advanced <b>Edge-Aware GAN</b> model to colorize grayscale images.
        Unlike normal colorizers, this model uses:
        <br><br>
        ✔ Edge Maps to reduce blurriness  
        ✔ LAB color space for accurate colors  
        ✔ U-Net Generator with skip connections  
        ✔ GAN loss for realistic outputs  
        ✔ Gradient + Edge consistency losses  
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# UPLOAD + COLORIZE SECTION
# ----------------------------------
st.markdown("""
<div class="section">
    <h2>📤 Upload Your Image</h2>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    with st.spinner("🎨 Processing... Please wait"):
        files = {"file": uploaded.getvalue()}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()

        def decode(b64):
            return Image.open(BytesIO(base64.b64decode(b64)))

        orig   = decode(data["original"])
        gray   = decode(data["grayscale"])
        edges  = decode(data["edges"])
        color  = decode(data["colorized"])

        # RESULTS SECTION
        st.markdown("""
        <div class="section">
            <h2>📸 Results</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col1.image(orig, caption="Original")
        col1.markdown("</div>", unsafe_allow_html=True)

        col2.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col2.image(gray, caption="Grayscale (L)")
        col2.markdown("</div>", unsafe_allow_html=True)

        col3.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col3.image(edges, caption="Edge Map")
        col3.markdown("</div>", unsafe_allow_html=True)

        col4.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col4.image(color, caption="Colorized Output")
        col4.markdown("</div>", unsafe_allow_html=True)


        # Download button
        buffer = BytesIO()
        color.save(buffer, format="PNG")
        st.download_button("⬇️ Download Colorized Image", buffer.getvalue(), "colorized.png", "image/png")


# ----------------------------------
# MODEL DETAILS SECTION
# ----------------------------------
with st.container():
    st.markdown("""
    <div class="section">
        <h2>🧠 Model Details</h2>
        <p style='font-size:17px; color:#dddddd;'>
        The generator is based on a <b>U-Net architecture</b> with:
        <br><br>
        • 8 encoder layers  
        • 7 decoder layers  
        • Skip connections  
        • Tanh output for AB channels  
        <br><br>
        <b>Loss functions used:</b><br>
        - L1 Loss  
        - Adversarial Loss  
        - Edge Consistency Loss  
        - Gradient Loss  
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# PERFORMANCE METRICS SECTION
# ----------------------------------
with st.container():
    st.markdown("""
    <div class="section">
        <h2>📊 Model Performance</h2>
        <p style='font-size:17px; color:#dddddd;'>
        Evaluated on validation set:<br><br>
        ✔ SSIM: <b>0.8911</b><br>
        ✔ PSNR: <b>23.79 dB</b><br>
        ✔ LPIPS: <b>0.1285</b> (lower is better)<br><br>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("""
<div class="footer">
Created by <b>Aman Singh Tamar</b> • Powered by PyTorch + FastAPI + Streamlit  
</div>
""", unsafe_allow_html=True)
