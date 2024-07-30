import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

# Load models
model_nonpotato = load_tflite_model("nokentang.tflite")
model_potato_type = load_tflite_model("KENTANG_KUALITAS.tflite")

class_names_nonpotato = ['Non-Potato', 'Potato']
class_names_potato_type = [
    'Keropeng Hitam', 
    'Kaki Hitam', 
    'Kudis Umum', 
    'Busuk Kering', 
    'Kentang Sehat', 
    'Lain-lain', 
    'Busuk Merah Muda'
]

# Expanded information about potato conditions
potato_info = {
    'Keropeng Hitam': {
        "description": "Keropeng hitam dapat mempengaruhi kualitas kentang, tetapi masih bisa dimakan jika bagian yang terinfeksi dibuang.",
        "price_range": "Rp 10.000 - Rp 15.000 per kg",
        "usage": "Disarankan untuk diolah menjadi sup atau masakan yang dimasak."
    },
    'Kaki Hitam': {
        "description": "Kaki hitam menunjukkan infeksi, dan sebaiknya tidak dimakan.",
        "price_range": "Rp 5.000 - Rp 10.000 per kg (biasanya dibuang)",
        "usage": "Sebaiknya dibuang untuk mencegah penyebaran infeksi."
    },
    'Kudis Umum': {
        "description": "Kudis umum dapat mengurangi kualitas kentang, tetapi tidak berbahaya jika dimasak dengan baik.",
        "price_range": "Rp 8.000 - Rp 12.000 per kg",
        "usage": "Dapat digunakan dalam masakan, tetapi disarankan untuk memasaknya dengan baik."
    },
    'Busuk Kering': {
        "description": "Busuk kering menunjukkan kerusakan, dan kentang ini sebaiknya dibuang.",
        "price_range": "Rp 0 - Rp 5.000 per kg (biasanya dibuang)",
        "usage": "Dibuang untuk menjaga kualitas produk."
    },
    'Kentang Sehat': {
        "description": "Kentang sehat aman untuk dikonsumsi.",
        "price_range": "Rp 15.000 - Rp 25.000 per kg",
        "usage": "Sangat baik untuk dimakan langsung atau diolah menjadi berbagai hidangan."
    },
    'Lain-lain': {
        "description": "Periksa kondisi spesifik, tetapi sebaiknya tidak dimakan tanpa pemeriksaan lebih lanjut.",
        "price_range": "Bervariasi tergantung pada kondisi.",
        "usage": "Konsultasikan dengan ahli jika kondisi tidak jelas."
    },
    'Busuk Merah Muda': {
        "description": "Busuk merah muda sebaiknya dibuang, karena menunjukkan pembusukan yang serius.",
        "price_range": "Rp 0 - Rp 5.000 per kg (biasanya dibuang)",
        "usage": "Hindari mengonsumsi kentang ini."
    }
}

threshold_nonpotato = 0.99

def predict_image(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred_nonpotato = predict_with_tflite(model_nonpotato, img_array.astype(np.float32))[0][0]
    pred_potato_type = predict_with_tflite(model_potato_type, img_array.astype(np.float32))[0]

    label_nonpotato = class_names_nonpotato[int(pred_nonpotato > threshold_nonpotato)]
    conf_nonpotato = pred_nonpotato if pred_nonpotato > threshold_nonpotato else 1 - pred_nonpotato
    label_potato_type = class_names_potato_type[np.argmax(pred_potato_type)]
    conf_potato_type = np.max(pred_potato_type)

    return label_nonpotato, conf_nonpotato, label_potato_type, conf_potato_type

# Streamlit UI
st.set_page_config(page_title="KentangQI", page_icon="🥔", layout="wide")

st.title("KentangQI 🥔")
st.write("Unggah gambar kentang atau gunakan kamera untuk memprediksi apakah gambar tersebut adalah kentang atau bukan, dan jika itu adalah kentang, jenis kentangnya akan diprediksi.")

# Add a camera input option
camera_option = st.radio("Pilih sumber gambar:", ("Unggah Gambar", "Gunakan Kamera"))

uploaded_file = None
camera_image = None

if camera_option == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = keras_image.load_img(uploaded_file)
        st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
        st.write("Mengklasifikasikan gambar...")
elif camera_option == "Gunakan Kamera":
    camera_image = st.camera_input("Ambil gambar")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Gambar yang diambil.', use_column_width=True)
        st.write("Mengklasifikasikan gambar...")

if uploaded_file or camera_image:
    label_nonpotato, conf_nonpotato, label_potato_type, conf_potato_type = predict_image(image)

    if label_nonpotato == 'Non-Potato':
        st.error("Peringatan: Gambar yang Anda masukkan bukan kentang.", icon="🚫")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Non-Potato/Potato", value=label_nonpotato, delta=f"{conf_nonpotato:.2f}", delta_color="normal")
            st.metric(label="Jenis Kentang", value=label_potato_type, delta=f"{conf_potato_type:.2f}", delta_color="normal")

        with col2:
            st.success(f"Jenis Kentang: {label_potato_type} ({conf_potato_type:.2f})")

        # Display additional information about the potato condition
        with st.expander("Informasi tentang kondisi kentang"):
            info = potato_info[label_potato_type]
            st.write(f"**Deskripsi:** {info['description']}")
            st.write(f"**Harga:** {info['price_range']}")
            st.write(f"**Penggunaan:** {info['usage']}")

# Additional sections for better engagement
st.sidebar.header("Informasi Umum")
st.sidebar.write("Kentang adalah sumber karbohidrat yang penting dan dapat diolah menjadi berbagai makanan.")
st.sidebar.write("Penting untuk memilih kentang yang sehat dan bebas dari penyakit.")

st.sidebar.header("Tips Memilih Kentang")
st.sidebar.write("- Pilih kentang yang keras dan tidak lembek.")
st.sidebar.write("- Hindari kentang yang memiliki bercak-bercak gelap.")
st.sidebar.write("- Periksa apakah ada bagian yang busuk atau bercambah.")

st.sidebar.header("Kalkulator Harga")
price_per_kg = st.sidebar.number_input("Harga per kg (Rp)", value=15000)
quantity = st.sidebar.number_input("Jumlah (kg)", value=1)
total_price = price_per_kg * quantity
st.sidebar.write(f"**Total Harga:** Rp {total_price}")

# Add CSS for styling
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #e6ffe6;
        border: 1px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .block-container {
        padding: 4rem 1rem 1rem 1rem; /* Add padding to the top */
    }
    .css-1v0mbdj {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
