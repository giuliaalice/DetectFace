import streamlit as st
import os
import tempfile
import subprocess
import shutil

st.set_page_config(page_title="Emotion Feature Extractor", layout="centered")

st.title("🎥 Estrattore di feature per emozioni da video")
st.markdown("Carica un video. L'app rileverà volto, landmark facciali e articolari e salverà un CSV con i descrittori.")

uploaded_video = st.file_uploader("Carica un video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Salva il video temporaneamente
        video_path = os.path.join(tmp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)

        # Crea le cartelle richieste
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "faces"), exist_ok=True)

        # Lancia lo script di estrazione (assume che extract_features.py sia nella stessa repo)
        st.info("Estrazione in corso...")
        result = subprocess.run([
            "python3", "extract_features.py",
            f"--video_path={video_path}",
            f"--output_dir={data_dir}"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            csv_path = os.path.join(data_dir, "features.csv")
            st.success("Estrazione completata. Scarica il CSV qui sotto:")
            with open(csv_path, "rb") as f:
                st.download_button("📥 Scarica CSV", f, file_name="features.csv")
        else:
            st.error("Errore durante l'estrazione delle feature.")
            st.text(result.stderr)
