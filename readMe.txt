I'm attaching zip file of all essential documents.
unzip the file into a folder.
open command prompt or terminal from the project directory

Technique:  Arnold Transform --> 2L dwt --> dct --> embed --> inverse dct --> inverse dwt 
Detailed expplanation is provided in Report.pdf

Execution commands :

python digitalWatermarking.py

It gives three options to perform watermarking

1. No attack --> watermark embedding and extracting without any attack
2. Salt pepper attack --> watermark embedding and extracting with salt pepper attack
3. Rotation attack --> watermark embedding and extracting with Rotation attack

coverImage.jpg --> cover Image 
watermark.png --> original watermark to be embedded
scrambledWatermark.jpg --> scrambled watermark from arnold transform
WatermarkedImage.png --> Image with watermark
rescrambledwm.jpg --> recovered image with watermark
extractedWatermark.png --> extracted watermark fromm inverse arnold transform
