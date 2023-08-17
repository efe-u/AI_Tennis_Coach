# AI_Tennis_Coach
Your virtual tennis coach powered by mediapipe.

# Installation and How to Use
1) Install requirements.txt

```
git clone https://github.com/efe-u/AI_Tennis_Coach.git
pip install -r requirements.txt
``` 
2) Install [pose_landmarker_heavy](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task). It should be placed in the among the files of the project.

3) Rename pose_landmarker_heavy.task as pose_landmarker.

4) Check that all following directories exist.
   - extractions
   - images
   - segmentations
   - results

If not, please create them. You may also create an additional "media" directory to store your videos. As paths are hard-coded at the moment, please make sure you've given the paths of your content correctly and that the directories above are in the project directory.
<br>
<br/>

If all packages are installed successfully running main.py should initiate a demo and yield example outcomes.
