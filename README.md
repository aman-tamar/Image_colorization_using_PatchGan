Overview:-
  This project implements deep learning-based image colorization that automatically adds color to grayscale images. The model uses a generative approach to produce realistic and vibrant colorizations.

Features:-
  Automatic Colorization: Convert grayscale images to color using deep learning
  
  Web Interface: Easy-to-use Flask-based web application
  
  Batch Processing: Colorize multiple images at once
  
  Real-time Preview: See results instantly
  
  Model Optimization: Traced PyTorch model for faster inference

 Project Structure:-
 Image-Colorization-Project/
├── app/                    # fastapi backend application
├── frontend/               # Frontend web interface
├── model/                  # Model architecture definitions
├── Image_Colorization_Project.ipynb  # Main training notebook
├── inference.ipynb         # Inference notebook
├── requirements.txt        # Python dependencies
└── model_performance.txt  # Model performance metrics

Technical Details:-
  Architecture:-
    Generator: U-Net based architecture with skip connections
    
    Loss Functions: Perceptual loss + L1 loss + GAN loss
    
    Input: L channel from LAB color space
    
    Output: AB channels from LAB color space

Training:-
  The model was trained on KOKO_dataset with:
  
  10 epochs
  
  Batch size: 16
  
  Learning rate: 0.0002
  
  Adam optimizer

Note:- 
  The model files are not included in this repository due to GitHub's file size limits
  
  For production use, consider using the traced model (generator_traced.pt) for faster inference
  
  Results may vary depending on input image quality and content

