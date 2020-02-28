# Clone the repo
!git clone https://github.com/mkocabas/VIBE.git
# Install dependencies
%cd VIBE/
!pip install torch numpy
!pip install -r requirements.txt
# Download pretrained weights and SMPL data
!sh prepare_data.sh