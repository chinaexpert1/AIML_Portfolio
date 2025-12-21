For RECTIFICATION:

The notebooks attempt to answer the questions posed. 


There is something wrong that I can't identify I need feedback. But I submitted:

techtrack/modules/rectification/augmentation.py
techtrack/modules/rectification/hard_negative_mining.py
techtrack/modules/utils/loss.py
Notebooks to submit:

techtrack/notebooks/demo_augmenation.ipynb
techtrack/notebooks/demo_hard_negatives.ipynb


There is an empty detections folder where I keep my detections locally.








FOR INFERENCE:

[comment]: <> (Task 6: Include instructions to run the service)

Step 1: Start Inference

Navigate your CLI to the "techtrack" folder and enter:

python app.py

The model will now wait for streaming to begin at port 23000

Step 2: Start the UDP streaming

In two separate terminal windows, enter

ffplay udp:127.0.0.1:23000

in one of them, then in the other:

ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000

Step 3: Check saved images as inference begins

The model will start to output saved images for every 10th frame, with bounding boxes drawn in.

Use CTRL-C in the CLI to stop inference, or the folder will be filled with these images as it goes all the way through the video (which has thousands of frames).


Note: The model was not containerized so you need to run it in an environemnt with opencv-python installed
The instruction did not require containerization so I didn't do it.