import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Device configuration & model loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

print(f"Loading model '{MODEL_ID}' on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# 2. Wavelet high-frequency artifact extraction
def extract_wavelet_artifacts(image_pil):
    """Computes 2D Haar sub-bands to reveal high-frequency compression/edge artifacts."""
    img_gray = np.array(image_pil.convert('L'), dtype=np.float32)

    # Decompose into LL, LH, HL, HH sub-bands
    lh = (img_gray[0::2, 0::2] - img_gray[1::2, 0::2] + img_gray[0::2, 1::2] - img_gray[1::2, 1::2]) / 4.0
    hl = (img_gray[0::2, 0::2] + img_gray[1::2, 0::2] - img_gray[0::2, 1::2] - img_gray[1::2, 1::2]) / 4.0
    hh = (img_gray[0::2, 0::2] - img_gray[1::2, 0::2] - img_gray[0::2, 1::2] + img_gray[1::2, 1::2]) / 4.0

    # High-frequency magnitude
    high_freq = np.sqrt(lh**2 + hl**2 + hh**2)
    norm_hf = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
    
    # Map to colormap (Magma) for visual clarity in UI
    heatmap_colored = cv2.applyColorMap(np.uint8(norm_hf * 255), cv2.COLORMAP_MAGMA)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_colored)

# 3. Main inference & localization function
def analyze_image(input_image):
    if input_image is None:
        return None, {}, None, None

    raw_img = Image.fromarray(input_image).convert("RGB")
    inputs = processor(images=raw_img, return_tensors="pt").to(DEVICE)
    inputs["pixel_values"].requires_grad = True

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]

    # Map label probabilities for Gradio Label component
    labels_dict = {}
    for idx, prob in enumerate(probs):
        label_name = model.config.id2label[idx]
        labels_dict[label_name] = float(prob.item())

    # Saliency / Spatial Localization Map
    pred_idx = torch.argmax(probs).item()
    score = logits[0, pred_idx]
    score.backward()

    gradients = inputs["pixel_values"].grad.abs().squeeze().detach().cpu().numpy()
    sal_map = np.mean(gradients, axis=0)
    sal_map = cv2.resize(sal_map, (raw_img.width, raw_img.height))
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

    # Color overlay for tampering localization
    sal_heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
    sal_heatmap = cv2.cvtColor(sal_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(raw_img), 0.65, sal_heatmap, 0.35, 0)
    overlay_img = Image.fromarray(overlay)

    # Wavelet high-frequency view
    wavelet_img = extract_wavelet_artifacts(raw_img)

    return labels_dict, overlay_img, wavelet_img

# 4. Gradio UI Interface
with gr.Blocks(title="AI Deepfake Detector with WGN & Localization") as demo:
    gr.Markdown(
        """
        # 🛡️ AI Deepfake Image Detector
        ### Comparative Analysis Toolkit: Base Classifier + Wavelet Analysis + Spatial Localization
        Upload a suspect face or image to analyze authenticity and inspect frequency/tampering traces.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Suspect Face/Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Classification Probabilities")
            
    with gr.Row():
        with gr.Column():
            loc_output = gr.Image(type="pil", label="Spatial Localization Heatmap (Tampering Clues)")
        with gr.Column():
            wavelet_output = gr.Image(type="pil", label="Wavelet High-Frequency Analysis (Edge/Artifact Spectrum)")

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_img],
        outputs=[label_output, loc_output, wavelet_output]
    )

if __name__ == "__main__":
    # Launch locally on your laptop: open http://127.0.0.1:7860 in your browser
    demo.launch(inbrowser=True)