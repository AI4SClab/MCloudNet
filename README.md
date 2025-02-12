# MCloudNet
Official repository for MCloudNet, a multi-modal AI framework that enhances ultra-short-term photovoltaic (PV) power forecasting by modeling multi-layer cloud structures from satellite imagery.

## 🌟 **Overview**  
MCloudNet integrates **multi-layer satellite cloud images**, **ground-based meteorological data**, and **advanced deep learning techniques** to improve PV power prediction, particularly in **data-scarce rural micro-grids**. The framework effectively captures **cloud motion vectors, occlusion coefficients, and cloud layer interactions**, leading to superior forecasting accuracy and interpretability.  

### 🚀 **Key Features**  
✅ **Multi-Layer Cloud Feature Extraction** – Separates high-, middle-, and low-altitude cloud layers for enhanced cloud dynamics modeling.  
✅ **Multi-Modal Learning** – Fuses satellite imagery and meteorological data to optimize forecasting accuracy.  
✅ **Robust Transferability** – Achieves high performance even in **new PV stations with limited historical data**, leveraging satellite-based generalization.  
✅ **Energy & Economic Impact** – Successfully deployed in **50+ photovoltaic stations** across underdeveloped regions, reducing **60 million kWh** of curtailment power and generating **24 million CNY** in economic benefits.  

## 📖 **Paper & Citation**  
If you find our work useful, please consider citing:  

```
@article{MCloudNet2025,
  title={MCloudNet: Multi-Layer Cloud Modeling for Ultra-Short-Term PV Forecasting},
  author={Your Name et al.},
  journal={IJCAI 2025},
  year={2025}
}
```

## 🛠 **Installation & Usage**  
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/YourOrg/MCloudNet.git
cd MCloudNet
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Inference  
```python
python predict.py --input sample_data
```

## 📊 **Datasets**  
MCloudNet is trained on **real-world PV datasets**, including:  
🔹 **Local Meteorological Data (LMD)** – 15-min weather observations.  
🔹 **Numerical Weather Prediction (NWP)** – High-resolution forecast data.  
🔹 **Satellite Cloud Imagery (Himawari-8)** – Cloud-top temperature and dynamics.



## ⚡ **Impact & Applications**  
🔹 **Micro-Grid Energy Optimization** – Enables reliable forecasting for off-grid PV systems in remote regions.  
🔹 **Sustainable Development** – Supports clean energy expansion aligned with **SDG 7, SDG 9, and SDG 13**.  
🔹 **Scalability** – Can be deployed in new PV sites **without extensive retraining**, making it highly adaptable for emerging markets.


**🔗 [GitHub Repository](https://github.com/YourOrg/MCloudNet) | 🌍 [Project Website](#)**  
