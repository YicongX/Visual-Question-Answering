# Visual-Question-Answering

In this project we address the task of Visual Question Answering (VQA): given an image and a question about it, our goal is to provide an answer to this question. While open-ended question answering would demand text generation, VQA is traditionally tackled as a classification problem; we select the answer out of a (large) pool of possible answers.

![image](https://user-images.githubusercontent.com/72159394/216913592-0cc53d50-4450-45fc-ac69-d08b3f0dcc6a.png)

Our goal is to implement two such answer classifiers. First, we implement a simple baseline model that featurizes the image and the question into 1-d vectors, then concatenates the two representations and feeds them to a linear layer to output a score for each possible answer. However, we noticed that such a late fusion of the two modalities (vision and language) does not allow for sufficient feature interaction and leads to poor results. Lastly, we implemented a Transformer-based model that featurizes the two modalities into sequences, then applies iterative self-/cross-attention on them.

## Setup
We used VQA v1.0 Open-Ended.(https://visualqa.org/vqa_v1_download.html).For reference, we used torch 1.11.0, torchvision 0.12 and transformers==4.12.3

## Baseline Results
VQA model for this project is a classifier over 5217 classes. The baseline featurize the image using a frozen pre-trained ResNet18. For that, we feed the image to ResNet18 except for the last layer which gives the classification logits. This gives us an 1-d feature vector for the image. We also featurize the question into an 1-d vector using a frozen pre-trained RoBERTa, a Transformer-based language model. Finally, we concatenate the two representations and feed to a linear layer. The parameters of this linear layer are trainable. 

![image](https://user-images.githubusercontent.com/72159394/216914814-9b8e9a26-ec82-435d-8d34-a2ea1174626c.png)

For baseline model, flattened image features and question embedding are simply concatenated together as a combined input for linear classifier. Thus the model is not learning the correlation and reasoning between image, question pair and answers but trying to project this combined input feature to the answer distribution. So as the plots shows, top 4 predicted answers aligned with the ground truth answer distribution with answer ???YES??? and ???Other??? as the most frequent predictions.

## Transformer Results
The baseline model has obvious limitations: it condenses the whole image (question) into an 1-d vector and there's no interaction between the visual and textual features. We improved the baseline model with attention layers, using Transformers.

Here is the model we implemented:

![image](https://user-images.githubusercontent.com/72159394/216915392-a7adb590-56f9-48e1-a95a-d8bf652e4e4b.png)

![image](https://user-images.githubusercontent.com/72159394/216915609-64076360-de0c-42c1-ba93-79030cc46849.png)

Most frequent ground truth results are not as dominant as the baseline model. We can see more less frequent results show up in the histogram and match up with ground truth distribution  for example class ???white???, ???red???, ???blue???, ???2???. The final validation accuracy has 13.9% increase compared to baseline model.


